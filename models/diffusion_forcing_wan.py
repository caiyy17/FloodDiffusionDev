import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tools.t5 import T5EncoderModel
from .tools.wan_model import WanModel

CROSS_MODULE_REGISTRY = {}


def register_cross_module(cls):
    CROSS_MODULE_REGISTRY[cls.__name__] = cls
    return cls


@register_cross_module
class T5TextCrossModule(nn.Module):
    """Cross-attention module for T5 text conditioning."""

    def __init__(
        self,
        len=512,
        dim=4096,
        t5_size="xxl",
        checkpoint_path=None,
        tokenizer_path=None,
        drop_out=0.1,
        input_keys={
            "text": "text",
            "text_end": "text_end",
        },
        output_keys=None,
    ):
        assert (checkpoint_path is not None and tokenizer_path is not None), "T5 checkpoint and tokenizer paths must be provided."
        super().__init__()
        self.cross_len = len
        self.cross_dim = dim
        self.cross_attn_norm = True
        self.drop_out = drop_out
        self.input_keys = input_keys

        self.text_encoder = T5EncoderModel(
            text_len=len,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            shard_fn=None,
            t5_size=t5_size,
        )
        self.text_cache = {}

    def encode(self, text_list, device):
        """Encode text list with cache. Returns List[Tensor]."""
        # Deduplicate uncached texts
        texts_to_encode = []
        for text in text_list:
            if text not in self.text_cache and text not in texts_to_encode:
                texts_to_encode.append(text)

        # Batch encode deduplicated texts
        if texts_to_encode:
            self.text_encoder.model.to(device)
            encoded = self.text_encoder(texts_to_encode, device)
            for text, feature in zip(texts_to_encode, encoded):
                self.text_cache[text] = feature.cpu()

        # Collect from cache
        return [self.text_cache[text].to(device) for text in text_list]

    def get_context(
        self, x, valid_len, seq_len, device, param_dtype, training=False
    ):
        """
        Get cross-attention context from input dict.

        Returns:
            context: List[Tensor]
            metadata: dict, may contain 'full_text'
        """
        text_key = self.input_keys.get("text", "text")
        text_end_key = self.input_keys.get("text_end", "text_end")
        metadata = {}

        assert text_key in x, f"Input key '{text_key}' not found in input."
        text_list = x[text_key]

        if isinstance(text_list[0], list):
            # Multi-segment text (stream mode)
            full_text = []
            all_context = []
            text_end_list = x[text_end_key]

            for i in range(len(valid_len)):
                if training and np.random.rand() <= self.drop_out:
                    single_text_list = [""]
                    single_text_end_list = [0, valid_len[i]]
                else:
                    single_text_list = text_list[i]
                    single_text_end_list = [0] + [
                        min(t, valid_len[i]) for t in text_end_list[i]
                    ]
                single_text_length_list = [
                    t - b
                    for t, b in zip(
                        single_text_end_list[1:], single_text_end_list[:-1]
                    )
                ]

                full_text.append(
                    " ////////// ".join(
                        [
                            f"{u} //dur:{t}"
                            for u, t in zip(
                                single_text_list, single_text_length_list
                            )
                        ]
                    )
                )

                single_text_context = self.encode(single_text_list, device)
                single_text_context = [
                    u.to(param_dtype) for u in single_text_context
                ]
                for u, duration in zip(
                    single_text_context, single_text_length_list
                ):
                    all_context.extend([u for _ in range(duration)])
                all_context.extend(
                    [
                        single_text_context[-1]
                        for _ in range(
                            seq_len - single_text_end_list[-1]
                        )
                    ]
                )
            metadata["full_text"] = full_text
            return all_context, metadata
        else:
            # Single text per sample
            full_text = [u for u in text_list]
            metadata["full_text"] = full_text
            if training:
                text_list = [
                    ("" if np.random.rand() <= self.drop_out else u)
                    for u in text_list
                ]
            else:
                text_list = [u for u in text_list]
            context = self.encode(text_list, device)
            context = [u.to(param_dtype) for u in context]

            return context, metadata

    def get_null_context(self, batch_size, device, param_dtype):
        """Get null/empty context for classifier-free guidance."""
        null_ctx = self.encode([""] * batch_size, device)
        return [u.to(param_dtype) for u in null_ctx]

    # --- Streaming state management ---

    def init_stream(self, batch_size):
        self.stream_condition_list = [[] for _ in range(batch_size)]

    def update_stream(self, x, device, param_dtype, first_chunk, chunk_size):
        """Add new context for a streaming step."""
        text_key = self.input_keys.get("text", "text")
        if text_key in x:
            new_ctx = self.encode(x[text_key], device)
        else:
            new_ctx = self.encode(
                [""] * len(self.stream_condition_list), device
            )
        new_ctx = [u.to(param_dtype) for u in new_ctx]

        for i in range(len(self.stream_condition_list)):
            if first_chunk:
                self.stream_condition_list[i].extend(
                    [new_ctx[i]] * chunk_size
                )
            else:
                self.stream_condition_list[i].extend([new_ctx[i]])

    def get_stream_context(self, end_index, seq_len):
        """Get context for current streaming window."""
        context = []
        for i in range(len(self.stream_condition_list)):
            context.extend(
                self.stream_condition_list[i][:end_index][-seq_len:]
            )
        return context

    def trim_stream(self, trim_len):
        """Trim stream state when wrapping around."""
        for i in range(len(self.stream_condition_list)):
            self.stream_condition_list[i] = self.stream_condition_list[i][
                trim_len:
            ]


class DiffForcingWanModel(nn.Module):
    def __init__(
        self,
        input_dim=256,
        hidden_dim=1024,
        ffn_dim=2048,
        freq_dim=256,
        num_heads=8,
        num_layers=8,
        time_embedding_scale=1.0,
        crossmodules=[
            {
                "name": "T5TextCrossModule",
                "len": 512,
                "dim": 4096,
            }
        ],
        prediction_type="vel",  # "vel", "x0", "noise"
        causal=False,
        schedule_config={
            "schedule_type": "triangular",
            "noise_type": "linear",
            "chunk_size": 5,
            "steps": 10
        },
        cfg_scale=5.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.time_embedding_scale = time_embedding_scale
        self.schedule_config = schedule_config
        self.cfg_scale = cfg_scale
        self.prediction_type = prediction_type
        self.causal = causal
        # Cross-attention modules
        self.cross_modules = nn.ModuleList()
        for cm_cfg in crossmodules:
            cfg = dict(cm_cfg)
            name = cfg.pop("name")
            cls = CROSS_MODULE_REGISTRY[name]
            self.cross_modules.append(cls(**cfg))

        self.model = WanModel(
            patch_size=(1, 1, 1),
            cross_len=tuple(cm.cross_len for cm in self.cross_modules),
            cross_dim=tuple(cm.cross_dim for cm in self.cross_modules),
            cross_attn_norm=tuple(
                cm.cross_attn_norm for cm in self.cross_modules
            ),
            in_dim=self.input_dim,
            dim=self.hidden_dim,
            ffn_dim=self.ffn_dim,
            freq_dim=self.freq_dim,
            out_dim=self.input_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            window_size=(-1, -1),
            qk_norm=True,
            eps=1e-6,
            causal=self.causal,
        )
        self.param_dtype = torch.float32

    def preprocess(self, x):
        # (bs, T, C) -> (bs, C, T, 1, 1)
        for i in range(len(x)):
            x[i] = x[i].permute(1, 0)[:, :, None, None]
        return x

    def postprocess(self, x):
        # (bs, C, T, 1, 1) ->  (bs, T, C)
        for i in range(len(x)):
            x[i] = x[i].permute(1, 0, 2, 3).contiguous().view(x[i].size(1), -1)
        return x

    def _get_time_steps(self, device, valid_len):
        time_steps = [] # (B,)
        time_schedules = [] # (B, T)
        time_schedules_derivative = [] # (B, T)
        
        if self.schedule_config["schedule_type"] == "uniform":
            for i in range(len(valid_len)):
                t = np.random.uniform(0, 1)
                time_steps.append(torch.tensor(t, device=device))
                single_time_schedules = torch.ones(valid_len[i], device=device) * t
                time_schedules_derivative.append(torch.ones(valid_len[i], device=device) / self.schedule_config["steps"])
                time_schedules.append(single_time_schedules)
        elif self.schedule_config["schedule_type"] == "triangular":
            for i in range(len(valid_len)):
                t = np.random.uniform(0, 1)
                time_steps.append(torch.tensor(t, device=device))
                max_time = valid_len[i] / self.schedule_config["chunk_size"]
                single_time_schedules = torch.clamp(
                    - torch.arange(valid_len[i], device=device)
                    / self.schedule_config["chunk_size"]
                    + t * max_time,
                    min=0.0,
                    max=1.0,
                )
                time_schedules_derivative.append(torch.ones(valid_len[i], device=device) / self.schedule_config["steps"])
                time_schedules.append(single_time_schedules)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_config['schedule_type']}")
        return time_steps, time_schedules, time_schedules_derivative
    
    def _get_noise_levels(self, device, valid_len, time_schedules):
        """Get noise levels"""
        noise_level = [] # (B, T)
        noise_level_derivative = [] # (B, T)
        for i in range(len(valid_len)):
            t = time_schedules[i]
            if self.schedule_config["noise_type"] == "linear":
                single_noise_level = (1 - t).to(device)
                noise_level.append(single_noise_level)
                noise_level_derivative.append(-torch.ones_like(single_noise_level).to(device))
            elif self.schedule_config["noise_type"] == "exponential":
                exponent = self.schedule_config.get("exponent", 2.0)
                if exponent > 1.0:
                    single_noise_level = (1 - t) ** exponent
                    noise_level.append(single_noise_level.to(device))
                    noise_level_derivative.append(-exponent * (1 - t) ** (exponent - 1) .to(device))
                elif exponent == 1.0:
                    single_noise_level = (1 - t)
                    noise_level.append(single_noise_level.to(device))
                    noise_level_derivative.append(-torch.ones_like(single_noise_level).to(device))
                elif exponent < 1.0:
                    single_noise_level = 1 - (t ** (1 / exponent))
                    noise_level.append(single_noise_level.to(device))
                    noise_level_derivative.append(- (1 / exponent) * (t ** ((1 / exponent) - 1)).to(device))
        return noise_level, noise_level_derivative
    
    def _get_window(self, valid_len,time_steps):
        """Get input and output window indices"""
        input_start_index = [] # (B,)
        input_end_index = [] # (B,)
        output_start_index = [] # (B,)
        output_end_index = [] # (B,)
        for i in range(len(time_steps)):
            t = time_steps[i].item()
            end_index = int(t * valid_len[i]) + 1
            end_index = min(valid_len[i], end_index)
            input_start_index.append(0)
            input_end_index.append(end_index)
            output_start_index.append(max(0, end_index - self.schedule_config["chunk_size"]))
            output_end_index.append(end_index)
        return input_start_index, input_end_index, output_start_index, output_end_index

    def add_noise(self, x, noise_level):
        """Add noise
        Args:
            x: (B, T, D)
            noise_level: (B, T)
        """
        noisy_x = [] # (B, T, D)
        noise = [] # (B, T, D)
        for i in range(len(x)):
            noise_i = torch.randn_like(x[i])
            # noise_level: (B, T) -> (B, T, 1)
            noise_level_i = noise_level[i].unsqueeze(-1)
            noisey_x_i = x[i] * (1 - noise_level_i) + noise_level_i * noise_i
            noisy_x.append(noisey_x_i)
            noise.append(noise_i)
        return noisy_x, noise

    def _get_all_contexts(self, x, seq_len, device, training=False, extra_len=0):
        """Get contexts from all cross modules.
        Returns:
            all_contexts: List[List[Tensor]], one per cross module
            metadata: dict, merged from all cross modules
        """
        all_contexts = []
        metadata = {}
        for cm in self.cross_modules:
            ctx, meta = cm.get_context(
                x, seq_len, device, self.param_dtype,
                training=training, extra_len=extra_len,
            )
            all_contexts.append(ctx)
            metadata.update(meta)
        return all_contexts, metadata

    def _get_all_null_contexts(self, batch_size, device):
        """Get null contexts from all cross modules for CFG."""
        return [
            cm.get_null_context(batch_size, device, self.param_dtype)
            for cm in self.cross_modules
        ]

    def forward(self, x):
        feature_original = x["feature"]  # (B, T, C)
        feature_length = x["feature_length"]  # (B,)
        batch_size, seq_len, _ = feature_original.shape
        device = feature_original.device
        feature = []
        valid_len = []
        for i in range(batch_size):
            length = min(feature_length[i].item(), seq_len)
            valid_len.append(length)
            feature.append(feature_original[i, :length, :])

        # get time steps and noise levels
        time_steps, time_schedules, _ = self._get_time_steps(device, valid_len)  # (B,)
        noise_level, noise_level_derivative = self._get_noise_levels(device, valid_len, time_schedules)  # (B, T)
        input_start_index, input_end_index, output_start_index, output_end_index = self._get_window(valid_len, time_steps)

        # Add noise to entire sequence
        noisy_feature, noise = self.add_noise(feature, noise_level)  # (B, T, D)
        feature = self.preprocess(feature)  # (B, C, T, 1, 1)
        noisy_feature = self.preprocess(noisy_feature)  # (B, C, T, 1, 1)
        noise = self.preprocess(noise)  # (B, C, T, 1, 1)

        feature_ref = []
        noise_ref = []
        noisy_feature_input = []
        for i in range(batch_size):
            feature_ref.append(feature[i, :, output_start_index[i]:output_end_index[i], ...])
            noise_ref.append(noise[i, :, output_start_index[i]:output_end_index[i], ...])
            noisy_feature_input.append(noisy_feature[i, :, input_start_index[i]:input_end_index[i], ...])

        # Get contexts from cross modules
        all_contexts, _ = self._get_all_contexts(
            x, seq_len, device, training=True,
        )

        # Through WanModel
        predicted_result = self.model(
            noisy_feature_input,
            noise_level * self.time_embedding_scale,
            all_contexts,
            seq_len,
            y=None,
        )  # (B, C, T, 1, 1)

        loss = 0.0
        for b in range(batch_size):
            if self.prediction_type == "vel":
                vel = (noise_ref[b] - feature_ref[b]) * noise_level_derivative[b, None, :, None, None]# (C, input_length, 1, 1)
                squared_error = (
                    predicted_result[b][:, output_start_index[b]:output_end_index[b], ...] - vel
                ) ** 2
            elif self.prediction_type == "x0":
                squared_error = (
                    predicted_result[b][:, output_start_index[b]:output_end_index[b], ...]
                    - feature_ref[b]
                ) ** 2
            elif self.prediction_type == "noise":
                squared_error = (
                    predicted_result[b][:, output_start_index[b]:output_end_index[b], ...]
                    - noise_ref[b]
                ) ** 2
            sample_loss = squared_error.mean()
            loss += sample_loss
        loss = loss / batch_size
        loss_dict = {"total": loss, "mse": loss}
        return loss_dict

    def generate(self, x, schedule_config={}):
        """
        Generation - Diffusion Forcing inference
        Uses triangular noise schedule, progressively generating from left to right

        Generation process:
        1. Start from t=0, gradually increase t
        2. Each t corresponds to a noise schedule: clean on left, noisy on right, gradient in middle
        3. After each denoising step, t increases slightly and continues
        """
        self.schedule_config.update(schedule_config)
        feature_length = x["feature_length"]
        batch_size = len(feature_length)
        seq_len = max(feature_length).item()

        if num_denoise_steps is None:
            num_denoise_steps = self.noise_steps
        assert num_denoise_steps % self.chunk_size == 0

        device = next(self.parameters()).device

        # Initialize entire sequence as pure noise
        generated = torch.randn(
            batch_size, seq_len + self.chunk_size, self.input_dim, device=device
        )
        generated = self.preprocess(generated)  # (B, C, T, 1, 1)

        # Calculate total number of time steps needed
        max_t = 1 + (seq_len - 1) / self.chunk_size

        # Step size for each advancement
        dt = 1 / num_denoise_steps
        total_steps = int(max_t / dt)

        # Get contexts from cross modules
        all_contexts, metadata = self._get_all_contexts(
            x, seq_len, device, training=False, extra_len=self.chunk_size,
        )
        generated_length = metadata.get("generated_length", feature_length)
        full_text = metadata.get("full_text", x.get("text", [""] * batch_size))

        # Get null contexts for CFG
        all_null_contexts = self._get_all_null_contexts(batch_size, device)

        # Progressively advance from t=0 to t=max_t
        for step in range(total_steps):
            # Current time step
            t = step * dt
            start_index = max(0, int(self.chunk_size * (t - 1)) + 1)
            end_index = int(self.chunk_size * t) + 1
            time_steps = torch.full((batch_size,), t, device=device)

            # Calculate current noise schedule
            noise_level, _ = self._get_noise_levels(
                device, seq_len + self.chunk_size, time_steps
            )  # (B, T)

            # Predict noise through WanModel
            noisy_input = []
            for i in range(batch_size):
                noisy_input.append(generated[i, :, :end_index, ...])

            predicted_result = self.model(
                noisy_input,
                noise_level * self.time_embedding_scale,
                all_contexts,
                seq_len + self.chunk_size,
                y=None,
            )  # (B, C, T, 1, 1)

            # Adjust using CFG
            if self.cfg_scale != 1.0:
                predicted_result_null = self.model(
                    noisy_input,
                    noise_level * self.time_embedding_scale,
                    all_null_contexts,
                    seq_len + self.chunk_size,
                    y=None,
                )  # (B, C, T, 1, 1)
                predicted_result = [
                    self.cfg_scale * pv - (self.cfg_scale - 1) * pvn
                    for pv, pvn in zip(predicted_result, predicted_result_null)
                ]

            for i in range(batch_size):
                predicted_result_i = predicted_result[i]  # (C, input_length, 1, 1)
                if self.prediction_type == "vel":
                    predicted_vel = predicted_result_i[:, start_index:end_index, ...]
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt
                elif self.prediction_type == "x0":
                    predicted_vel = (
                        predicted_result_i[:, start_index:end_index, ...]
                        - generated[i, :, start_index:end_index, ...]
                    ) / (
                        noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt
                elif self.prediction_type == "noise":
                    predicted_vel = (
                        generated[i, :, start_index:end_index, ...]
                        - predicted_result_i[:, start_index:end_index, ...]
                    ) / (
                        1
                        + dt
                        - noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt

        generated = self.postprocess(generated)  # (B, T, C)
        y_hat_out = []
        for i in range(batch_size):
            # cut off the padding
            single_generated = generated[i, : generated_length[i], :]
            y_hat_out.append(single_generated)
        out = {}
        out["generated"] = y_hat_out
        out["text"] = full_text

        return out

    @torch.no_grad()
    def stream_generate(self, x, num_denoise_steps=None):
        """
        Streaming generation - Diffusion Forcing inference
        Uses triangular noise schedule, progressively generating from left to right

        Generation process:
        1. Start from t=0, gradually increase t
        2. Each t corresponds to a noise schedule: clean on left, noisy on right, gradient in middle
        3. After each denoising step, t increases slightly and continues
        """
        feature_length = x["feature_length"]
        batch_size = len(feature_length)
        seq_len = max(feature_length).item()

        if num_denoise_steps is None:
            num_denoise_steps = self.noise_steps
        assert num_denoise_steps % self.chunk_size == 0

        device = next(self.parameters()).device

        # Initialize entire sequence as pure noise
        generated = torch.randn(
            batch_size, seq_len + self.chunk_size, self.input_dim, device=device
        )
        generated = self.preprocess(generated)  # (B, C, T, 1, 1)

        # Calculate total number of time steps needed
        max_t = 1 + (seq_len - 1) / self.chunk_size

        # Step size for each advancement
        dt = 1 / num_denoise_steps
        total_steps = int(max_t / dt)

        # Get contexts from cross modules
        all_contexts, metadata = self._get_all_contexts(
            x, seq_len, device, training=False, extra_len=self.chunk_size,
        )
        generated_length = metadata.get("generated_length", feature_length)

        # Get null contexts for CFG
        all_null_contexts = self._get_all_null_contexts(batch_size, device)

        commit_index = 0
        # Progressively advance from t=0 to t=max_t
        for step in range(total_steps):
            # Current time step
            t = step * dt
            start_index = max(0, int(self.chunk_size * (t - 1)) + 1)
            end_index = int(self.chunk_size * t) + 1
            time_steps = torch.full((batch_size,), t, device=device)

            # Calculate current noise schedule
            noise_level, _ = self._get_noise_levels(
                device, seq_len + self.chunk_size, time_steps
            )  # (B, T)

            # Predict noise through WanModel
            noisy_input = []
            for i in range(batch_size):
                noisy_input.append(generated[i, :, :end_index, ...])

            predicted_result = self.model(
                noisy_input,
                noise_level * self.time_embedding_scale,
                all_contexts,
                seq_len + self.chunk_size,
                y=None,
            )  # (B, C, T, 1, 1)

            # Adjust using CFG
            if self.cfg_scale != 1.0:
                predicted_result_null = self.model(
                    noisy_input,
                    noise_level * self.time_embedding_scale,
                    all_null_contexts,
                    seq_len + self.chunk_size,
                    y=None,
                )  # (B, C, T, 1, 1)
                predicted_result = [
                    self.cfg_scale * pv - (self.cfg_scale - 1) * pvn
                    for pv, pvn in zip(predicted_result, predicted_result_null)
                ]

            for i in range(batch_size):
                predicted_result_i = predicted_result[i]  # (C, input_length, 1, 1)
                if self.prediction_type == "vel":
                    predicted_vel = predicted_result_i[:, start_index:end_index, ...]
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt
                elif self.prediction_type == "x0":
                    predicted_vel = (
                        predicted_result_i[:, start_index:end_index, ...]
                        - generated[i, :, start_index:end_index, ...]
                    ) / (
                        noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt
                elif self.prediction_type == "noise":
                    predicted_vel = (
                        generated[i, :, start_index:end_index, ...]
                        - predicted_result_i[:, start_index:end_index, ...]
                    ) / (
                        1
                        + dt
                        - noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt

            if commit_index < start_index:
                output = generated[:, :, commit_index:start_index, ...]
                output = self.postprocess(output)  # (B, T, C)
                y_hat_out = []
                for i in range(batch_size):
                    if commit_index < generated_length[i]:
                        y_hat_out.append(
                            output[i, : generated_length[i] - commit_index, ...]
                        )
                    else:
                        y_hat_out.append(None)

                out = {}
                out["generated"] = y_hat_out
                yield out
                commit_index = start_index

        output = generated[:, :, commit_index:, ...]
        output = self.postprocess(output)  # (B, T_remain, C)
        y_hat_out = []
        for i in range(batch_size):
            if commit_index < generated_length[i]:
                y_hat_out.append(output[i, : generated_length[i] - commit_index, ...])
            else:
                y_hat_out.append(None)
        out = {}
        out["generated"] = y_hat_out
        yield out

    def init_generated(self, seq_len, batch_size=1, num_denoise_steps=None):
        self.seq_len = seq_len
        self.batch_size = batch_size
        if num_denoise_steps is None:
            self.num_denoise_steps = self.noise_steps
        else:
            self.num_denoise_steps = num_denoise_steps
        assert self.num_denoise_steps % self.chunk_size == 0
        self.dt = 1 / self.num_denoise_steps
        self.current_step = 0
        self.generated = torch.randn(
            self.batch_size, self.seq_len * 2 + self.chunk_size, self.input_dim
        )
        self.generated = self.preprocess(self.generated)  # (B, C, T, 1, 1)
        self.commit_index = 0

        # Initialize streaming state for all cross modules
        for cm in self.cross_modules:
            cm.init_stream(self.batch_size)

    @torch.no_grad()
    def stream_generate_step(self, x, first_chunk=True):
        """
        Streaming generation step - Diffusion Forcing inference
        Uses triangular noise schedule, progressively generating from left to right

        Generation process:
        1. Start from t=0, gradually increase t
        2. Each t corresponds to a noise schedule: clean on left, noisy on right, gradient in middle
        3. After each denoising step, t increases slightly and continues
        """

        device = next(self.parameters()).device
        if first_chunk:
            self.generated = self.generated.to(device)

        # Update streaming state for all cross modules
        for cm in self.cross_modules:
            cm.update_stream(
                x, device, self.param_dtype, first_chunk, self.chunk_size
            )

        # Get null contexts for CFG
        all_null_contexts = self._get_all_null_contexts(self.batch_size, device)

        end_step = (
            (self.commit_index + self.chunk_size)
            * self.num_denoise_steps
            / self.chunk_size
        )
        while self.current_step < end_step:
            current_time = self.current_step * self.dt
            start_index = max(0, int(self.chunk_size * (current_time - 1)) + 1)
            end_index = int(self.chunk_size * current_time) + 1
            time_steps = torch.full((self.batch_size,), current_time, device=device)

            noise_level, _ = self._get_noise_levels(device, end_index, time_steps)
            noise_level = noise_level[:, -self.seq_len :]  # (B, T)

            # Predict noise through WanModel
            noisy_input = []
            for i in range(self.batch_size):
                noisy_input.append(
                    self.generated[i, :, :end_index, ...][:, -self.seq_len :]
                )  # (C, T, 1, 1)

            # Get streaming context from all cross modules
            all_stream_contexts = [
                cm.get_stream_context(end_index, self.seq_len)
                for cm in self.cross_modules
            ]

            predicted_result = self.model(
                noisy_input,
                noise_level * self.time_embedding_scale,
                all_stream_contexts,
                min(end_index, self.seq_len),
                y=None,
            )  # (B, C, T, 1, 1)

            # Adjust using CFG
            if self.cfg_scale != 1.0:
                predicted_result_null = self.model(
                    noisy_input,
                    noise_level * self.time_embedding_scale,
                    all_null_contexts,
                    min(end_index, self.seq_len),
                    y=None,
                )  # (B, C, T, 1, 1)
                predicted_result = [
                    self.cfg_scale * pv - (self.cfg_scale - 1) * pvn
                    for pv, pvn in zip(predicted_result, predicted_result_null)
                ]

            for i in range(self.batch_size):
                predicted_result_i = predicted_result[i]  # (C, input_length, 1, 1)
                if end_index > self.seq_len:
                    predicted_result_i = torch.cat(
                        [
                            torch.zeros(
                                predicted_result_i.shape[0],
                                end_index - self.seq_len,
                                predicted_result_i.shape[2],
                                predicted_result_i.shape[3],
                                device=device,
                            ),
                            predicted_result_i,
                        ],
                        dim=1,
                    )
                if self.prediction_type == "vel":
                    predicted_vel = predicted_result_i[:, start_index:end_index, ...]
                    self.generated[i, :, start_index:end_index, ...] += (
                        predicted_vel * self.dt
                    )
                elif self.prediction_type == "x0":
                    predicted_vel = (
                        predicted_result_i[:, start_index:end_index, ...]
                        - self.generated[i, :, start_index:end_index, ...]
                    ) / (
                        noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    self.generated[i, :, start_index:end_index, ...] += (
                        predicted_vel * self.dt
                    )
                elif self.prediction_type == "noise":
                    predicted_vel = (
                        self.generated[i, :, start_index:end_index, ...]
                        - predicted_result_i[:, start_index:end_index, ...]
                    ) / (
                        1
                        + self.dt
                        - noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    self.generated[i, :, start_index:end_index, ...] += (
                        predicted_vel * self.dt
                    )
            self.current_step += 1
        output = self.generated[:, :, self.commit_index : self.commit_index + 1, ...]
        output = self.postprocess(output)  # (B, 1, C)
        out = {}
        out["generated"] = output
        self.commit_index += 1

        if self.commit_index == self.seq_len * 2:
            self.generated = torch.cat(
                [
                    self.generated[:, :, self.seq_len :, ...],
                    torch.randn(
                        self.batch_size,
                        self.input_dim,
                        self.seq_len,
                        1,
                        1,
                        device=device,
                    ),
                ],
                dim=2,
            )
            self.current_step -= self.seq_len * self.num_denoise_steps / self.chunk_size
            self.commit_index -= self.seq_len
            for cm in self.cross_modules:
                cm.trim_stream(self.seq_len)
        return out
