import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tools.t5 import T5EncoderModel
from .tools.wan_model import WanModel

CROSS_MODULE_REGISTRY = {}
TIME_SCHEDULER_REGISTRY = {}

def register_cross_module(cls):
    CROSS_MODULE_REGISTRY[cls.__name__] = cls
    return cls

def register_time_scheduler(cls):
    TIME_SCHEDULER_REGISTRY[cls.__name__] = cls
    return cls

@register_time_scheduler
class UniformTimeScheduler:
    def __init__(self, config):
        self.steps = config["steps"]
        self.noise_type = config.get("noise_type", "linear")
        self.exponent = config.get("exponent", 2.0)

    def get_total_steps(self, seq_len):
        return self.steps

    def get_time_steps(self, device, valid_len, current_step=None):
        time_steps = []
        if current_step is None:
            for i in range(len(valid_len)):
                time_steps.append(torch.tensor(np.random.uniform(0, 1), device=device, dtype=torch.float64))
        elif isinstance(current_step, int):
            for i in range(len(valid_len)):
                t = current_step * (1 / self.steps)
                time_steps.append(torch.tensor(t, device=device, dtype=torch.float64))
        elif isinstance(current_step, list):
            for i in range(len(valid_len)):
                t = current_step[i] * (1 / self.steps)
                time_steps.append(torch.tensor(t, device=device, dtype=torch.float64))
        return time_steps

    def get_time_schedules(self, device, valid_len, time_steps):
        time_schedules = []
        time_schedules_derivative = []
        for i in range(len(valid_len)):
            t = time_steps[i].item()
            time_schedules.append(torch.ones(valid_len[i], device=device) * t)
            time_schedules_derivative.append(torch.ones(valid_len[i], device=device) / self.steps)
        return time_schedules, time_schedules_derivative

    def get_windows(self, valid_len, time_steps):
        input_start, input_end, output_start, output_end = [], [], [], []
        for i in range(len(time_steps)):
            input_start.append(0)
            input_end.append(valid_len[i])
            output_start.append(0)
            output_end.append(valid_len[i])
        return input_start, input_end, output_start, output_end

    def get_noise_levels(self, device, valid_len, time_schedules):
        noise_level = []
        noise_level_derivative = []
        for i in range(len(valid_len)):
            t = time_schedules[i]
            if self.noise_type == "linear":
                nl = (1 - t).to(device)
                nld = (-torch.ones_like(nl)).to(device)
            elif self.noise_type == "exponential":
                if self.exponent > 1.0:
                    nl = ((1 - t) ** self.exponent).to(device)
                    nld = (-self.exponent * (1 - t) ** (self.exponent - 1)).to(device)
                elif self.exponent == 1.0:
                    nl = (1 - t).to(device)
                    nld = (-torch.ones_like(nl)).to(device)
                else:
                    nl = (1 - t ** (1 / self.exponent)).to(device)
                    nld = (-(1 / self.exponent) * t ** (1 / self.exponent - 1)).to(device)
            else:
                raise ValueError(f"Unknown noise type: {self.noise_type}")
            noise_level.append(nl)
            noise_level_derivative.append(nld)
        return noise_level, noise_level_derivative

@register_time_scheduler
class TriangularTimeScheduler:
    def __init__(self, config):
        self.steps = config["steps"]
        self.chunk_size = config["chunk_size"]
        self.noise_type = config.get("noise_type", "linear")
        self.exponent = config.get("exponent", 2.0)

    def get_total_steps(self, seq_len):
        return int(self.steps * seq_len / self.chunk_size)

    def get_time_steps(self, device, valid_len, current_step=None):
        time_steps = []
        if current_step is None:
            for i in range(len(valid_len)):
                max_time = valid_len[i] / self.chunk_size
                time_steps.append(torch.tensor(np.random.uniform(0, max_time), device=device))
        elif isinstance(current_step, int):
            for i in range(len(valid_len)):
                t = current_step * (1 / self.steps)
                time_steps.append(torch.tensor(t, device=device))
        elif isinstance(current_step, list):
            for i in range(len(valid_len)):
                t = current_step[i] * (1 / self.steps)
                time_steps.append(torch.tensor(t, device=device))
        return time_steps

    def get_time_schedules(self, device, valid_len, time_steps):
        time_schedules = []
        time_schedules_derivative = []
        for i in range(len(valid_len)):
            t = time_steps[i].item()
            time_schedules = torch.clamp(
                -torch.arange(valid_len[i], device=device) / self.chunk_size + t,
                min=0.0,
                max=1.0,
            )
            time_schedules_next = torch.clamp(
                -torch.arange(valid_len[i], device=device) / self.chunk_size + t + (1 / self.steps),
                min=0.0,
                max=1.0,
            )
            time_schedules_derivative = torch.clamp((time_schedules_next - time_schedules), min=0.0, max=1.0)
            time_schedules.append(time_schedules)
            time_schedules_derivative.append(time_schedules_derivative)
        return time_schedules, time_schedules_derivative

    def get_windows(self, valid_len, time_steps):
        # suppose self.steps % self.chunk_size == 0
        # so the index can be added by a small amount to get the correct window
        # we choose 0.5 * (1 / self.steps) here
        input_start, input_end, output_start, output_end = [], [], [], []
        for i in range(len(time_steps)):
            t = time_steps[i].item()
            start_index = max(0, math.floor((t - 1) * self.chunk_size + 0.5 * (1 / self.steps)) + 1)
            end_index = min(valid_len[i], math.floor(t * self.chunk_size + 0.5 * (1 / self.steps)) + 1)
            input_start.append(0)
            input_end.append(end_index)
            output_start.append(start_index)
            output_end.append(end_index)
        return input_start, input_end, output_start, output_end

    def get_noise_levels(self, device, valid_len, time_schedules):
        noise_level = []
        noise_level_derivative = []
        for i in range(len(valid_len)):
            t = time_schedules[i]
            if self.noise_type == "linear":
                nl = (1 - t).to(device)
                nld = (-torch.ones_like(nl)).to(device)
            elif self.noise_type == "exponential":
                if self.exponent > 1.0:
                    nl = ((1 - t) ** self.exponent).to(device)
                    nld = (-self.exponent * (1 - t) ** (self.exponent - 1)).to(device)
                elif self.exponent == 1.0:
                    nl = (1 - t).to(device)
                    nld = (-torch.ones_like(nl)).to(device)
                else:
                    nl = (1 - t ** (1 / self.exponent)).to(device)
                    nld = (-(1 / self.exponent) * t ** (1 / self.exponent - 1)).to(device)
            else:
                raise ValueError(f"Unknown noise type: {self.noise_type}")
            noise_level.append(nl)
            noise_level_derivative.append(nld)
        return noise_level, noise_level_derivative

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

    def update_stream(self, x, device, param_dtype):
        """Add new context for a streaming step."""
        text_key = self.input_keys.get("text", "text")
        if text_key in x:
            new_ctx = self.encode(x[text_key], device)
        else:
            raise ValueError(f"Input key '{text_key}' not found in input.")
        new_ctx = [u.to(param_dtype) for u in new_ctx]

        for i in range(len(self.stream_condition_list)):
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
        causal=False,
        prediction_type="vel",  # "vel", "x0", "noise"
        crossmodules=[
            {
                "name": "T5TextCrossModule",
                "len": 512,
                "dim": 4096,
            }
        ],
        schedule_config={
            "schedule_name": "TriangularTimeScheduler",
            "noise_type": "linear",
            "chunk_size": 5,
            "steps": 10,
            "extra_len": 5,
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
        self.time_scheduler = TIME_SCHEDULER_REGISTRY[schedule_config["schedule_name"]](schedule_config)
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

    def _get_all_contexts(self, x, valid_len, seq_len, device, training=False):
        """Get contexts from all cross modules.
        Returns:
            all_contexts: List[List[Tensor]], one per cross module
            metadata: dict, merged from all cross modules
        """
        all_contexts = []
        metadata = {}
        for cm in self.cross_modules:
            ctx, meta = cm.get_context(
                x, valid_len, seq_len, device, self.param_dtype,
                training=training,
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
        time_steps = self.time_scheduler.get_time_steps(device, valid_len)  # (B,)
        time_schedules, _ = self.time_scheduler.get_time_schedules(device, valid_len, time_steps)  # (B, T)
        noise_level, noise_level_derivative = self.time_scheduler.get_noise_levels(device, valid_len, time_schedules)  # (B, T)
        input_start_index, input_end_index, output_start_index, output_end_index = self.time_scheduler.get_windows(valid_len, time_steps)

        # Add noise to entire sequence
        noisy_feature, noise = self.add_noise(feature, noise_level)  # (B, T, D)
        feature = self.preprocess(feature)  # (B, C, T, 1, 1)
        noisy_feature = self.preprocess(noisy_feature)  # (B, C, T, 1, 1)
        noise = self.preprocess(noise)  # (B, C, T, 1, 1)

        feature_ref = []
        noise_ref = []
        noisy_feature_input = []
        for i in range(batch_size):
            feature_ref.append(feature[i][:, output_start_index[i]:output_end_index[i], ...])
            noise_ref.append(noise[i][:, output_start_index[i]:output_end_index[i], ...])
            noisy_feature_input.append(noisy_feature[i][:, input_start_index[i]:input_end_index[i], ...])

        # Get contexts from cross modules
        all_contexts, _ = self._get_all_contexts(
            x, valid_len, seq_len, device, training=True,
        )

        # Pad noise_level list into [B, seq_len] tensor for WanModel
        time_schedules_padded = torch.zeros(batch_size, seq_len, device=device)
        for i in range(batch_size):
            ts = time_schedules[i][:input_end_index[i]]
            time_schedules_padded[i, :len(ts)] = ts

        # Through WanModel
        predicted_result = self.model(
            noisy_feature_input,
            time_schedules_padded * self.time_embedding_scale,
            all_contexts,
            seq_len,
            y=None,
        )  # (B, C, T, 1, 1)

        loss = 0.0
        for b in range(batch_size):
            if self.prediction_type == "vel":
                nld = noise_level_derivative[b][output_start_index[b]:output_end_index[b]]
                vel = (noise_ref[b] - feature_ref[b]) * nld[None, :, None, None]  # (C, output_length, 1, 1)
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
        self.time_scheduler = TIME_SCHEDULER_REGISTRY[self.schedule_config["schedule_name"]](self.schedule_config)
        extra_len = self.schedule_config.get("extra_len", 0)
        feature_length = x["feature_length"]  # (B,)
        batch_size = len(feature_length)
        seq_len = max(feature_length).item() + extra_len
        device = next(self.parameters()).device

        valid_len = []
        for i in range(batch_size):
            length = min(feature_length[i].item(), seq_len)
            valid_len.append(length)
        generated_len = [seq_len for _ in range(batch_size)]

        # Initialize entire sequence as pure noise, generate use seq_len for all samples
        generated = torch.randn(batch_size, seq_len, self.input_dim, device=device)
        generated = [generated[i] for i in range(batch_size)]
        generated = self.preprocess(generated)  # list of (C, total_len, 1, 1)

        # Get contexts from cross modules
        all_contexts, metadata = self._get_all_contexts(
            x, generated_len, seq_len, device, training=False,
        )
        full_text = metadata.get("full_text", x.get("text", [""] * batch_size))

        # Get null contexts for CFG
        all_null_contexts = self._get_all_null_contexts(batch_size, device)

        total_steps = self.time_scheduler.get_total_steps(seq_len)
        # Progressively advance from t=0 to t=max_t
        for step in range(total_steps):
            # get time steps and noise levels
            time_steps = self.time_scheduler.get_time_steps(device, generated_len, step)  # (B,)
            time_schedules, time_schedules_derivative = self.time_scheduler.get_time_schedules(device, generated_len, time_steps)  # (B, T)
            noise_level, noise_level_derivative = self.time_scheduler.get_noise_levels(device, generated_len, time_schedules)  # (B, T)
            input_start_index, input_end_index, output_start_index, output_end_index = self.time_scheduler.get_windows(generated_len, time_steps)

            # Predict through WanModel
            noisy_input = []
            for i in range(batch_size):
                noisy_input.append(generated[i][:, input_start_index[i]:input_end_index[i], ...])  # (C, T, 1, 1)

            # Pad noise_level list into [B, seq_len] tensor for WanModel
            time_schedules_padded = torch.zeros(batch_size, seq_len, device=device)
            for i in range(batch_size):
                ts = time_schedules[i][:input_end_index[i]]
                time_schedules_padded[i, :len(ts)] = ts

            predicted_result = self.model(
                noisy_input,
                time_schedules_padded * self.time_embedding_scale,
                all_contexts,
                seq_len,
                y=None,
            )  # list of (C, T, 1, 1)

            # Adjust using CFG
            if self.cfg_scale != 1.0:
                predicted_result_null = self.model(
                    noisy_input,
                    time_schedules_padded * self.time_embedding_scale,
                    all_null_contexts,
                    seq_len,
                    y=None,
                )
                predicted_result = [
                    self.cfg_scale * pv - (self.cfg_scale - 1) * pvn
                    for pv, pvn in zip(predicted_result, predicted_result_null)
                ]

            for i in range(batch_size):
                predicted_result_i = predicted_result[i]  # (C, seq_len, 1, 1)
                os, oe = output_start_index[i], output_end_index[i]
                dt = time_schedules_derivative[i][os:oe][None, :, None, None]
                if self.prediction_type == "vel":
                    predicted_vel = predicted_result_i[:, os:oe, ...]
                    generated[i][:, os:oe, ...] += predicted_vel * dt
                elif self.prediction_type == "x0":
                    nl = noise_level[i][os:oe][None, :, None, None]
                    predicted_vel = (
                        predicted_result_i[:, os:oe, ...]
                        - generated[i][:, os:oe, ...]
                    ) / nl
                    generated[i][:, os:oe, ...] += predicted_vel * dt
                elif self.prediction_type == "noise":
                    nl = noise_level[i][os:oe][None, :, None, None]
                    predicted_vel = (
                        generated[i][:, os:oe, ...]
                        - predicted_result_i[:, os:oe, ...]
                    ) / (1 + dt - nl)
                    generated[i][:, os:oe, ...] += predicted_vel * dt

        generated = self.postprocess(generated)  # list of (T, C)
        y_hat_out = []
        for i in range(batch_size):
            single_generated = generated[i][:valid_len[i], :]
            y_hat_out.append(single_generated)
        out = {}
        out["generated"] = y_hat_out
        out["text"] = full_text

        return out

    def init_generated(self, seq_len):
        self.seq_len = seq_len
        self.extra_len = self.schedule_config.get("extra_len", 0)
        self.current_step = []
        self.commit_index = 0

        total_buf = self.seq_len * 2 + self.extra_len
        # Initialize entire sequence as pure noise
        generated = torch.randn(1, total_buf, self.input_dim)
        self.generated = self.preprocess([generated])[0]  # list of (C, total_buf, 1, 1)
        # Initialize streaming state for all cross modules
        for cm in self.cross_modules:
            cm.init_stream(1)

    # @torch.no_grad()
    # def stream_generate_step(self, x, first_chunk=True):
    #     """
    #     Streaming generation step - Diffusion Forcing inference
    #     Uses triangular noise schedule, progressively generating from left to right

    #     Generation process:
    #     1. Start from t=0, gradually increase t
    #     2. Each t corresponds to a noise schedule: clean on left, noisy on right, gradient in middle
    #     3. After each denoising step, t increases slightly and continues
    #     """

    #     device = next(self.parameters()).device
    #     if first_chunk:
    #         self.generated = self.generated.to(device)

    #     # Update streaming state for all cross modules
    #     for cm in self.cross_modules:
    #         cm.update_stream(x, device, self.param_dtype, first_chunk)
    #     # Get null contexts for CFG
    #     all_null_contexts = self._get_all_null_contexts(1, device)
    #     while self.current_step < end_step:
    #         # get time steps and noise levels
    #         self.current_step = self.time_scheduler.get_time_steps(device, [self.seq_len * 2], self.current_step)  # (B,)
    #         time_schedules, time_schedules_derivative = self.time_scheduler.get_time_schedules(device, [self.seq_len * 2], self.current_step)  # (B, T)
    #         noise_level, noise_level_derivative = self.time_scheduler.get_noise_levels(device, [self.seq_len * 2], time_schedules)  # (B, T)
    #         input_start_index, input_end_index, output_start_index, output_end_index = self.time_scheduler.get_windows([self.seq_len * 2], self.current_step)

    #         # Predict through WanModel
    #         noisy_input = []
    #         for i in range(1):
    #             noisy_input.append(generated[i][:, input_start_index[i]:input_end_index[i], ...])  # (C, T, 1, 1)

    #         # Pad noise_level list into [B, seq_len] tensor for WanModel
    #         noise_level_padded = torch.zeros(1, seq_len, device=device)
    #         for i in range(1):
    #             nl = noise_level[i][:input_end_index[i]]
    #             noise_level_padded[i, :len(nl)] = nl

    #         predicted_result = self.model(
    #             noisy_input,
    #             noise_level_padded * self.time_embedding_scale,
    #             all_contexts,
    #             seq_len,
    #             y=None,
    #         )  # list of (C, T, 1, 1)

    #         # Adjust using CFG
    #         if self.cfg_scale != 1.0:
    #             predicted_result_null = self.model(
    #                 noisy_input,
    #                 noise_level_padded * self.time_embedding_scale,
    #                 all_null_contexts,
    #                 seq_len,
    #                 y=None,
    #             )
    #             predicted_result = [
    #                 self.cfg_scale * pv - (self.cfg_scale - 1) * pvn
    #                 for pv, pvn in zip(predicted_result, predicted_result_null)
    #             ]

    #         for i in range(1):
    #             predicted_result_i = predicted_result[i]  # (C, seq_len, 1, 1)
    #             os, oe = output_start_index[i], output_end_index[i]
    #             dt = time_schedules_derivative[i][os:oe][None, :, None, None]
    #             if self.prediction_type == "vel":
    #                 predicted_vel = predicted_result_i[:, os:oe, ...]
    #                 generated[i][:, os:oe, ...] += predicted_vel * dt
    #             elif self.prediction_type == "x0":
    #                 nl = noise_level[i][os:oe][None, :, None, None]
    #                 predicted_vel = (
    #                     predicted_result_i[:, os:oe, ...]
    #                     - generated[i][:, os:oe, ...]
    #                 ) / nl
    #                 generated[i][:, os:oe, ...] += predicted_vel * dt
    #             elif self.prediction_type == "noise":
    #                 nl = noise_level[i][os:oe][None, :, None, None]
    #                 predicted_vel = (
    #                     generated[i][:, os:oe, ...]
    #                     - predicted_result_i[:, os:oe, ...]
    #                 ) / (1 + dt - nl)
    #                 generated[i][:, os:oe, ...] += predicted_vel * dt

    #         for i in range(1):
    #             predicted_result_i = predicted_result[i]  # (C, visible_len, 1, 1)
    #             if end_index > self.seq_len:
    #                 predicted_result_i = torch.cat(
    #                     [
    #                         torch.zeros(
    #                             predicted_result_i.shape[0],
    #                             end_index - self.seq_len,
    #                             predicted_result_i.shape[2],
    #                             predicted_result_i.shape[3],
    #                             device=device,
    #                         ),
    #                         predicted_result_i,
    #                     ],
    #                     dim=1,
    #                 )
    #             nl = noise_level_full[i][start_index:end_index][None, :, None, None]
    #             if self.prediction_type == "vel":
    #                 predicted_vel = predicted_result_i[:, start_index:end_index, ...]
    #                 self.generated[0][:, start_index:end_index, ...] += (
    #                     predicted_vel * dt
    #                 )
    #             elif self.prediction_type == "x0":
    #                 predicted_vel = (
    #                     predicted_result_i[:, start_index:end_index, ...]
    #                     - self.generated[i][:, start_index:end_index, ...]
    #                 ) / nl
    #                 self.generated[i][:, start_index:end_index, ...] += (
    #                     predicted_vel * dt
    #                 )
    #             elif self.prediction_type == "noise":
    #                 predicted_vel = (
    #                     self.generated[i][:, start_index:end_index, ...]
    #                     - predicted_result_i[:, start_index:end_index, ...]
    #                 ) / (1 + dt - nl)
    #                 self.generated[i][:, start_index:end_index, ...] += (
    #                     predicted_vel * dt
    #                 )
    #         self.current_step += 1

    #     output = [
    #         self.generated[i][:, self.commit_index : self.commit_index + 1, ...]
    #         for i in range(self.batch_size)
    #     ]
    #     output = self.postprocess(output)  # list of (1, C)
    #     out = {}
    #     out["generated"] = output
    #     self.commit_index += 1

    #     if self.commit_index == self.seq_len * 2:
    #         for i in range(self.batch_size):
    #             self.generated[i] = torch.cat(
    #                 [
    #                     self.generated[i][:, self.seq_len :, ...],
    #                     torch.randn(
    #                         self.input_dim, self.seq_len, 1, 1, device=device,
    #                     ),
    #                 ],
    #                 dim=1,
    #             )
    #         self.current_step -= int(self.seq_len * steps / chunk_size)
    #         self.commit_index -= self.seq_len
    #         for cm in self.cross_modules:
    #             cm.trim_stream(self.seq_len)
    #     return out
