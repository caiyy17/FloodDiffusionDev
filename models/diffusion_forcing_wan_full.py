import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tools.t5 import T5EncoderModel
from .tools.wan_model import WanModel

EPSILON = 1e-5
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
        self.chunk_size = config["chunk_size"]
        self.noise_type = config.get("noise_type", "linear")
        self.exponent = config.get("exponent", 2.0)

    def get_total_steps(self, seq_len):
        return self.steps

    def get_time_steps(self, device, valid_len, current_step=None):
        time_steps = []
        if current_step is None:
            for i in range(len(valid_len)):
                time_steps.append(torch.tensor(np.random.uniform(0, 1), device=device))
        elif isinstance(current_step, int):
            for i in range(len(valid_len)):
                t = current_step * (1 / self.steps)
                time_steps.append(torch.tensor(t, device=device))
        elif isinstance(current_step, list):
            for i in range(len(valid_len)):
                t = current_step[i] * (1 / self.steps)
                time_steps.append(torch.tensor(t, device=device))
        return time_steps

    def get_time_schedules(self, device, valid_len, time_steps, training=False):
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
    
    def add_noise(self, x, noise_level, training=False, noise=None):
        """Add noise
        Args:
            x: list of (C, T, H, W)
            noise_level: list of (T,)
        """
        noisy_x = []
        noise_output = []
        for i in range(len(x)):
            if noise is not None:
                noise_i = noise[i]
            else:
                noise_i = torch.randn_like(x[i])
            noise_level_i = noise_level[i][None, :, None, None]  # (1, T, 1, 1)
            noisy_x_i = x[i] * (1 - noise_level_i) + noise_level_i * noise_i
            noisy_x.append(noisy_x_i)
            noise_output.append(noise_i)
        return noisy_x, noise_output

    # --- Streaming support ---

    # We assume seq_len is multiple of chunk_size for simplicity
    def get_committable(self, condition_frames):
        """Given total accumulated conditions, return how many frames can be committed and the corresponding step count."""
        wave_index = condition_frames // self.chunk_size
        committable_length = wave_index * self.chunk_size
        committable_steps = wave_index * self.steps
        return committable_length, committable_steps

    def get_step_rollback(self, seq_len):
        """Get the step count to subtract when wrapping the buffer by seq_len."""
        steps = seq_len // self.chunk_size * self.steps
        return steps

@register_time_scheduler
class TriangularTimeScheduler:
    def __init__(self, config):
        self.steps = config["steps"]
        self.chunk_size = config["chunk_size"]
        self.noise_type = config.get("noise_type", "linear")
        self.exponent = config.get("exponent", 2.0)
        self.random_epsilon = config.get("random_epsilon", 0.01) # schedule jittering

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

    def get_time_schedules(self, device, valid_len, time_steps, training=False):
        time_schedules = []
        time_schedules_derivative = []
        for i in range(len(valid_len)):
            t = time_steps[i].item()
            current_time_schedules = torch.clamp(
                -torch.arange(valid_len[i], device=device) / self.chunk_size + t,
                min=0.0,
                max=1.0,
            )
            current_time_schedules_next = torch.clamp(
                -torch.arange(valid_len[i], device=device) / self.chunk_size + t + (1 / self.steps),
                min=0.0,
                max=1.0,
            )
            current_time_schedules_derivative = torch.clamp((current_time_schedules_next - current_time_schedules), min=0.0, max=1.0)
            if training:
                current_time_schedules = torch.clamp(
                    current_time_schedules + torch.randn_like(current_time_schedules) * self.random_epsilon,
                    min=0.0,
                    max=1.0,
                )
            time_schedules.append(current_time_schedules)
            time_schedules_derivative.append(current_time_schedules_derivative)
        return time_schedules, time_schedules_derivative

    def get_windows(self, valid_len, time_steps):
        # for the floating point issue, we can add the start_index by 0.5 / [steps * chunk_size]
        # for convenience, we just choose 0.5 * (1 / (self.steps * self.chunk_size)) here
        input_start, input_end, output_start, output_end = [], [], [], []
        for i in range(len(time_steps)):
            t = time_steps[i].item()
            start_index = max(0, math.floor((t - 1) * self.chunk_size + 0.5 * (1 / (self.steps * self.chunk_size))) + 1)
            end_index = min(valid_len[i], math.floor(t * self.chunk_size + 0.5 * (1 / (self.steps * self.chunk_size))) + 1)
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
    
    def add_noise(self, x, noise_level, training=False, noise=None):
        """Add noise
        Args:
            x: list of (C, T, H, W)
            noise_level: list of (T,)
        """
        noisy_x = []
        noise_output = []
        for i in range(len(x)):
            if noise is not None:
                noise_i = noise[i]
            else:
                noise_i = torch.randn_like(x[i])
            noise_level_i = noise_level[i][None, :, None, None]  # (1, T, 1, 1)
            noisy_x_i = x[i] * (1 - noise_level_i) + noise_level_i * noise_i
            noisy_x.append(noisy_x_i)
            noise_output.append(noise_i)
        return noisy_x, noise_output

    # --- Streaming support ---

    def get_committable(self, total_frames):
        """Given total accumulated conditions, return how many frames can be committed.
        Currently, we suppose steps % chunk_size == 0 for simplicity."""
        committable_length = max(0, total_frames - self.chunk_size + 1)
        committable_steps = total_frames * (self.steps // self.chunk_size)
        return committable_length, committable_steps

    def get_step_rollback(self, seq_len):
        """Get the step count to subtract when wrapping the buffer by seq_len.
        Corresponds to how many steps were consumed by seq_len frames."""
        steps = seq_len * (self.steps // self.chunk_size)
        return steps

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

        if text_key not in x:
            text_list = ["" for _ in range(len(valid_len))]
        else:
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
        """Add one frame of context for a streaming step."""
        text_key = self.input_keys.get("text", "text")
        text_input = x[text_key]
        new_ctx = self.encode(text_input, device)
        new_ctx = [u.to(param_dtype) for u in new_ctx]
        for i in range(len(self.stream_condition_list)):
            self.stream_condition_list[i].append(new_ctx[i])

    def get_stream_context(self, start_index, end_index, seq_len):
        context = []
        for i in range(len(self.stream_condition_list)):
            context.extend(self.stream_condition_list[i][start_index:end_index][ -seq_len : ])
            context.extend([self.stream_condition_list[i][-1]] * max(0, seq_len - (end_index - start_index)))
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
        mean_path=None,
        std_path=None,
        hidden_dim=1024,
        ffn_dim=2048,
        freq_dim=256,
        num_heads=8,
        num_layers=8,
        time_embedding_scale=1.0,
        causal=False,
        rope_channel_split=[1, 0, 0],
        spatial_shape=(1, 1),
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
            "extra_len": 4,
            "random_epsilon": 0.00,
        },
        cfg_config=[
            {
                "scale": 5.0,
                "crossmodule": [True],
            },
            {
                "scale": -4.0,
                "crossmodule": [False],
            }
        ]
    ):
        super().__init__()

        self.mean_path = mean_path
        self.std_path = std_path
        self.input_dim = input_dim
        self.spatial_shape = tuple(spatial_shape)
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.time_embedding_scale = time_embedding_scale
        self.causal = causal
        self.rope_channel_split = rope_channel_split
        self.prediction_type = prediction_type
        self.cfg_config = cfg_config
        self.schedule_config = schedule_config
        self.time_scheduler = TIME_SCHEDULER_REGISTRY[schedule_config["schedule_name"]](schedule_config)
        # Cross-attention modules
        self.cross_modules = nn.ModuleList()
        for cm_cfg in crossmodules:
            cfg = dict(cm_cfg)
            name = cfg.pop("name")
            cls = CROSS_MODULE_REGISTRY[name]
            self.cross_modules.append(cls(**cfg))

        if self.mean_path is not None:
            self.register_buffer(
                "mean", torch.from_numpy(np.load(self.mean_path)).float()
            )
        else:
            self.register_buffer("mean", torch.zeros(input_dim))

        if self.std_path is not None:
            self.register_buffer(
                "std", torch.from_numpy(np.load(self.std_path)).float()
            )
        else:
            self.register_buffer("std", torch.ones(input_dim))

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
            rope_channel_split=self.rope_channel_split,
        )
        self.param_dtype = torch.float32

    def preprocess(self, x):
        """Convert last-channel format to channel-first, padding to 4D (C, T, H, W).
        (T, C) -> (C, T, 1, 1)
        (T, H, C) -> (C, T, H, 1)
        (T, H, W, C) -> (C, T, H, W)
        """
        for i in range(len(x)):
            ndim = x[i].ndim
            if ndim == 2:      # (T, C)
                x[i] = x[i].permute(1, 0)[:, :, None, None]
            elif ndim == 3:    # (T, H, C)
                x[i] = x[i].permute(2, 0, 1)[:, :, :, None]
            elif ndim == 4:    # (T, H, W, C)
                x[i] = x[i].permute(3, 0, 1, 2)
        return x

    def postprocess(self, x):
        """Reverse of preprocess: channel-first 4D back to last-channel, stripping padding dims.
        (C, T, 1, 1) -> (T, C)
        (C, T, H, 1) -> (T, H, C)
        (C, T, H, W) -> (T, H, W, C)
        """
        for i in range(len(x)):
            shape = x[i].shape  # (C, T, H, W)
            if shape[2] == 1 and shape[3] == 1:      # (C, T, 1, 1) -> (T, C)
                x[i] = x[i][:, :, 0, 0].permute(1, 0)
            elif shape[3] == 1:                        # (C, T, H, 1) -> (T, H, C)
                x[i] = x[i][:, :, :, 0].permute(1, 2, 0)
            else:                                      # (C, T, H, W) -> (T, H, W, C)
                x[i] = x[i].permute(1, 2, 3, 0)
        return x

    def _get_all_contexts(self, x, valid_len, seq_len, device, training=False, crossmodule_switch=None):
        """Get contexts from all cross modules.
        Args:
            crossmodule_switch: Optional list of bools, one per cross module.
                True = use real context, False = use null context.
                None = use real context for all.
        Returns:
            all_contexts: List[List[Tensor]], one per cross module
            metadata: dict, merged from all cross modules
        """
        all_contexts = []
        metadata = {}
        batch_size = len(valid_len)
        for idx, cm in enumerate(self.cross_modules):
            if crossmodule_switch is not None and not crossmodule_switch[idx]:
                ctx = cm.get_null_context(batch_size, device, self.param_dtype)
                all_contexts.append(ctx)
            else:
                ctx, meta = cm.get_context(
                    x, valid_len, seq_len, device, self.param_dtype,
                    training=training,
                )
                all_contexts.append(ctx)
                metadata.update(meta)
        return all_contexts, metadata

    def forward(self, x):
        feature_original = x["feature"]  # (B, T, C)
        feature_length = x["feature_length"]  # (B,)
        feature_original = (feature_original - self.mean) / self.std
        batch_size = feature_original.shape[0]
        seq_len = feature_original.shape[1]
        device = feature_original.device
        feature = []
        valid_len = []
        for i in range(batch_size):
            length = min(feature_length[i].item(), seq_len)
            valid_len.append(length)
            feature.append(feature_original[i, :length, ...])

        # Preprocess to (C, T, 1, 1) per sample
        feature = self.preprocess(feature)

        # get time steps and noise levels
        time_steps = self.time_scheduler.get_time_steps(device, valid_len)  # (B,)
        time_schedules, _ = self.time_scheduler.get_time_schedules(device, valid_len, time_steps, training=True)  # (B, T)
        noise_level, noise_level_derivative = self.time_scheduler.get_noise_levels(device, valid_len, time_schedules)  # (B, T)
        input_start_index, input_end_index, output_start_index, output_end_index = self.time_scheduler.get_windows(valid_len, time_steps)
        noisy_feature, noise = self.time_scheduler.add_noise(feature, noise_level, training=True)

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

    def generate(self, x):
        """
        Generation - Diffusion Forcing inference
        Uses triangular noise schedule, progressively generating from left to right

        Generation process:
        1. Start from t=0, gradually increase t
        2. Each t corresponds to a noise schedule: clean on left, noisy on right, gradient in middle
        3. After each denoising step, t increases slightly and continues
        """
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

        # Initialize entire sequence as pure noise
        generated = torch.randn(batch_size, seq_len, *self.spatial_shape, self.input_dim, device=device)
        generated = [generated[i] for i in range(batch_size)]
        generated = self.preprocess(generated)

        # Precompute contexts for each cfg entry
        cfg_contexts_list = []
        metadata = {}
        for cfg_entry in self.cfg_config:
            ctx, meta = self._get_all_contexts(
                x, generated_len, seq_len, device, training=False,
                crossmodule_switch=cfg_entry["crossmodule"],
            )
            cfg_contexts_list.append(ctx)
            metadata.update(meta)
        full_text = metadata.get("full_text", x.get("text", [""] * batch_size))

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

            # Run model for each cfg entry and accumulate
            predicted_result = None
            for cfg_idx, cfg_entry in enumerate(self.cfg_config):
                scale = cfg_entry["scale"]
                pred = self.model(
                    noisy_input,
                    time_schedules_padded * self.time_embedding_scale,
                    cfg_contexts_list[cfg_idx],
                    seq_len,
                    y=None,
                )
                if predicted_result is None:
                    predicted_result = [scale * p for p in pred]
                else:
                    predicted_result = [a + scale * p for a, p in zip(predicted_result, pred)]

            for i in range(batch_size):
                predicted_result_i = predicted_result[i]  # (C, seq_len, 1, 1)
                os, oe = output_start_index[i], output_end_index[i]
                dt = time_schedules_derivative[i][os:oe][None, :, None, None]
                nl = noise_level[i][os:oe][None, :, None, None]
                nld = noise_level_derivative[i][os:oe][None, :, None, None]
                if self.prediction_type == "vel":
                    predicted_vel = predicted_result_i[:, os:oe, ...]
                elif self.prediction_type == "x0":
                    predicted_vel = (
                        generated[i][:, os:oe, ...]
                        - predicted_result_i[:, os:oe, ...]
                    ) / nl * nld
                elif self.prediction_type == "noise":
                    predicted_vel = (
                        predicted_result_i[:, os:oe, ...]
                        - generated[i][:, os:oe, ...]
                    ) / (1 - nl + EPSILON) * nld
                generated[i][:, os:oe, ...] += predicted_vel * dt
                
        generated = self.postprocess(generated)  # list of (T, C)
        y_hat_out = []
        for i in range(batch_size):
            single_generated = generated[i][:valid_len[i], :] * self.std + self.mean
            y_hat_out.append(single_generated)
        out = {}
        out["generated"] = y_hat_out
        out["text"] = full_text

        return out

    def init_generated(self, seq_len, batch_size=1, schedule_config={}):
        """Initialize streaming generation state.

        Args:
            seq_len: Model window size (how many frames WanModel processes per step).
            schedule_config: Optional schedule config overrides.

        Buffer is 2*seq_len. Model window is always buffer[0:seq_len].
        When conditions overflow seq_len, shift buffer by seq_len and restart.
        """
        self.schedule_config.update(schedule_config)
        self.time_scheduler = TIME_SCHEDULER_REGISTRY[self.schedule_config["schedule_name"]](self.schedule_config)

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.buf_len = seq_len * 2
        self.current_step = 0
        self.current_commit = 0
        self.condition_frames = 0

        device = next(self.parameters()).device
        # Initialize entire buffer as pure noise
        generated = torch.randn(batch_size, self.buf_len, *self.spatial_shape, self.input_dim, device=device)
        generated = [generated[i] for i in range(batch_size)]
        self.generated = self.preprocess(generated)

        # Initialize streaming state for all cross modules
        for cm in self.cross_modules:
            cm.init_stream(self.batch_size)


    def _rollback(self):
        """Shift buffer by seq_len when conditions overflow the window."""
        for i in range(self.batch_size):
            self.generated[i][:, :self.seq_len, ...] = self.generated[i][:, self.seq_len:, ...].clone()
            self.generated[i][:, self.seq_len:, ...] = torch.randn_like(
                self.generated[i][:, self.seq_len:, ...]
            )
        self.current_step -= self.time_scheduler.get_step_rollback(self.seq_len)
        self.condition_frames -= self.seq_len
        self.current_commit -= self.seq_len
        for cm in self.cross_modules:
            cm.trim_stream(self.seq_len)

    @torch.no_grad()
    def stream_generate_step(self, x):
        """
        Streaming generation step. Each call provides 1 frame of conditions.
        The scheduler determines committable frames from accumulated conditions.

        Returns:
            dict with "generated": list of one (N, C) tensor, or [] if nothing to commit.
        """
        device = next(self.parameters()).device
        self.generated = [g.to(device) for g in self.generated]

        # 1. Update conditions (1 frame per call)
        for cm in self.cross_modules:
            cm.update_stream(x, device, self.param_dtype)
        self.condition_frames += 1

        # 2. Rollback if conditions overflow the window
        if self.condition_frames > self.buf_len:
            self._rollback()

        # 3. Determine how many frames can be committed
        committable_length, committable_steps = self.time_scheduler.get_committable(self.condition_frames)
        while self.current_step < committable_steps:
            time_steps = self.time_scheduler.get_time_steps(
                device, [self.buf_len], self.current_step)
            time_schedules, time_schedules_derivative = (
                self.time_scheduler.get_time_schedules(
                    device, [self.buf_len], time_steps))
            noise_level, noise_level_derivative = self.time_scheduler.get_noise_levels(
                device, [self.buf_len], time_schedules)
            is_, ie_, os_, oe_ = self.time_scheduler.get_windows(
                [self.buf_len], time_steps)
            noisy_input = [self.generated[i][:, is_[0]:ie_[0], ...][:, -self.seq_len:, ...]
                           for i in range(self.batch_size)]
            ts = time_schedules[0][is_[0]:ie_[0]][-self.seq_len:]
            cut_length = max(0, (ie_[0] - is_[0]) - self.seq_len)

            time_schedules_padded = torch.zeros(self.batch_size, self.seq_len, device=device)
            for i in range(self.batch_size):
                time_schedules_padded[i, :len(ts)] = ts

            # Run model for each cfg entry and accumulate
            predicted_result = None
            for cfg_entry in self.cfg_config:
                scale = cfg_entry["scale"]
                switches = cfg_entry["crossmodule"]
                entry_contexts = []
                for j, cm in enumerate(self.cross_modules):
                    if switches[j]:
                        entry_contexts.append(
                            cm.get_stream_context(is_[0], ie_[0], self.seq_len)
                        )
                    else:
                        entry_contexts.append(
                            cm.get_null_context(self.batch_size, device, self.param_dtype)
                        )
                pred = self.model(
                    noisy_input,
                    time_schedules_padded * self.time_embedding_scale,
                    entry_contexts,
                    self.seq_len,
                    y=None,
                )
                if predicted_result is None:
                    predicted_result = [scale * p for p in pred]
                else:
                    predicted_result = [a + scale * p for a, p in zip(predicted_result, pred)]

            os_idx, oe_idx = os_[0], oe_[0]
            dt = time_schedules_derivative[0][os_idx:oe_idx][None, :, None, None]
            pred_os_idx = os_idx - cut_length
            pred_oe_idx = oe_idx - cut_length

            for i in range(self.batch_size):
                predicted_result_i = predicted_result[i]
                nl = noise_level[0][os_idx:oe_idx][None, :, None, None]
                nld = noise_level_derivative[0][os_idx:oe_idx][None, :, None, None]
                if self.prediction_type == "vel":
                    predicted_vel = predicted_result_i[:, pred_os_idx:pred_oe_idx, ...]
                elif self.prediction_type == "x0":
                    predicted_vel = (
                        self.generated[i][:, os_idx:oe_idx, ...]
                        - predicted_result_i[:, pred_os_idx:pred_oe_idx, ...]
                    ) / nl * nld
                elif self.prediction_type == "noise":
                    predicted_vel = (
                        predicted_result_i[:, pred_os_idx:pred_oe_idx, ...]
                        - self.generated[i][:, os_idx:oe_idx, ...]
                    ) / (1 - nl + EPSILON) * nld
                self.generated[i][:, os_idx:oe_idx, ...] += predicted_vel * dt
            self.current_step += 1

        # 5. Extract newly committed frames
        if self.current_commit < committable_length:
            output = [self.generated[i][:, self.current_commit:committable_length, ...]
                       for i in range(self.batch_size)]
            output = self.postprocess(output)
            output = [o * self.std + self.mean for o in output]
            self.current_commit = committable_length
            return {"generated": output}
        else:
            empty = [torch.zeros(self.input_dim, 0, *self.spatial_shape, device=device)
                     for _ in range(self.batch_size)]
            empty = self.postprocess(empty)
            return {"generated": empty}
