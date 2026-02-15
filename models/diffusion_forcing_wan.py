import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tools.t5 import T5EncoderModel
from .tools.wan_model import WanModel

EPSILON = 1e-5

class TriangularTimeScheduler:
    def __init__(self, config):
        self.steps = config["steps"]
        self.chunk_size = config["chunk_size"]
        self.noise_type = config.get("noise_type", "linear")
        self.exponent = config.get("exponent", 2.0)
        self.random_epsilon = config.get("random_epsilon", 0.00) # schedule jittering

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
        alpha = []
        alpha_derivative = []
        beta = []
        beta_derivative = []
        for i in range(len(valid_len)):
            t = time_schedules[i]
            if self.noise_type == "linear":
                alpha_i = t.to(device)
                alpha_derivative_i = torch.ones_like(alpha_i).to(device)
                beta_i = (1 - t).to(device)
                beta_derivative_i = (-torch.ones_like(beta_i)).to(device)
            elif self.noise_type == "exponential":
                if self.exponent > 1.0:
                    beta_i = ((1 - t) ** self.exponent).to(device)
                    beta_derivative_i = (-self.exponent * (1 - t) ** (self.exponent - 1)).to(device)
                    alpha_i = (1 - beta_i).to(device)
                    alpha_derivative_i = (-beta_derivative_i).to(device)
                elif self.exponent == 1.0:
                    alpha_i = t.to(device)
                    alpha_derivative_i = torch.ones_like(alpha_i).to(device)
                    beta_i = (1 - t).to(device)
                    beta_derivative_i = (-torch.ones_like(beta_i)).to(device)
                else:
                    alpha_i = (t ** (1 / self.exponent)).to(device)
                    alpha_derivative_i = ((1 / self.exponent) * t ** (1 / self.exponent - 1)).to(device)
                    beta_i = (1 - alpha_i).to(device)
                    beta_derivative_i = (-alpha_derivative_i).to(device)
            else:
                raise ValueError(f"Unknown noise type: {self.noise_type}")
            alpha.append(torch.clamp(alpha_i, min=0.0, max=1.0))
            alpha_derivative.append(alpha_derivative_i)
            beta.append(torch.clamp(beta_i, min=0.0, max=1.0))
            beta_derivative.append(beta_derivative_i)
        return alpha, alpha_derivative, beta, beta_derivative
    
    def add_noise(self, x, alpha, beta, input_start, input_end, output_start, output_end, training=False, noise=None):
        """Add noise and slice into input/reference regions.
        Args:
            x: list of (C, T, H, W), x0 in training, xt in inference
            alpha: list of (T,)
            beta: list of (T,)
            input_start/input_end: per-sample input window indices
            output_start/output_end: per-sample output window indices
        Returns:
            x0: list of (C, output_len, H, W)
            eps: list of (C, output_len, H, W)
            xt: list of (C, input_len, H, W)
        """
        x0 = []
        eps = []
        xt = []
        if training:
            for i in range(len(x)):
                if noise is not None:
                    noise_i = noise[i]
                else:
                    noise_i = torch.randn_like(x[i])
                alpha_i = alpha[i][None, :, None, None]  # (1, T, 1, 1)
                beta_i = beta[i][None, :, None, None]  # (1, T, 1, 1)
                noisy_x_i = x[i] * alpha_i + noise_i * beta_i  # (C, T, H, W)
                x0.append(x[i][:, output_start[i]:output_end[i], ...])
                eps.append(noise_i[:, output_start[i]:output_end[i], ...])
                xt.append(noisy_x_i[:, input_start[i]:input_end[i], ...])
        else:
            for i in range(len(x)):
                xt.append(x[i][:, input_start[i]:input_end[i], ...])
        return x0, eps, xt

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
        self.len = len
        self.dim = dim
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
        prediction_type="vel",  # "vel", "x0", "eps"
        text_config={
            "len": 512,
            "dim": 4096,
        },
        schedule_config={
            "noise_type": "linear",
            "chunk_size": 5,
            "steps": 10,
            "extra_len": 4,
            "random_epsilon": 0.00,
        },
        cfg_config={
            "text_scale": 5.0,
            "null_scale": -4.0,
        },
        input_keys={
            "feature": "feature",
            "feature_length": "feature_length",
            "text": "text",
            "text_end": "text_end",
        },
    ):
        super().__init__()
        self.input_keys = input_keys

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
        self.time_scheduler = TriangularTimeScheduler(schedule_config)
        # Cross-attention module (text)
        self.text_module = T5TextCrossModule(**text_config)

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
            text_len=self.text_module.len,
            text_dim=self.text_module.dim,
            cross_attn_norm=self.text_module.cross_attn_norm,
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

    def _extract_inputs(self, x):
        """Extract inputs from x using input_keys mapping."""
        inputs = {}
        for internal_key, external_key in self.input_keys.items():
            if external_key in x:
                inputs[internal_key] = x[external_key]
        return inputs

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

    def forward(self, x):
        x = self._extract_inputs(x)
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
        alpha, alpha_derivative, beta, beta_derivative = self.time_scheduler.get_noise_levels(device, valid_len, time_schedules)  # (B, T)
        input_start_index, input_end_index, output_start_index, output_end_index = self.time_scheduler.get_windows(valid_len, time_steps)
        x0, eps, xt = self.time_scheduler.add_noise(
            feature, alpha, beta, input_start_index, input_end_index, output_start_index, output_end_index, training=True
        )

        # Get context from text cross module
        context, _ = self.text_module.get_context(
            x, valid_len, seq_len, device, self.param_dtype, training=True,
        )

        # Pad noise_level list into [B, seq_len] tensor for WanModel
        time_schedules_padded = torch.zeros(batch_size, seq_len, device=device)
        for i in range(batch_size):
            ts = time_schedules[i][input_start_index[i]:input_end_index[i]]
            time_schedules_padded[i, :len(ts)] = ts

        # Through WanModel
        predicted_result = self.model(
            xt,
            time_schedules_padded * self.time_embedding_scale,
            context,
            seq_len,
            y=None,
        )  # (B, C, T, 1, 1)

        loss = 0.0
        for b in range(batch_size):
            pred_os = output_start_index[b] - input_start_index[b]
            pred_oe = output_end_index[b] - input_start_index[b]
            alpha_d = alpha_derivative[b][output_start_index[b]:output_end_index[b]]
            beta_d = beta_derivative[b][output_start_index[b]:output_end_index[b]]
            if self.prediction_type == "vel":
                vel = x0[b] * alpha_d[None, :, None, None] + eps[b] * beta_d[None, :, None, None]  # (C, output_length, 1, 1)
                squared_error = (
                    predicted_result[b][:, pred_os:pred_oe, ...] - vel
                ) ** 2
            elif self.prediction_type == "x0":
                squared_error = (
                    predicted_result[b][:, pred_os:pred_oe, ...]
                    - x0[b]
                ) ** 2
            elif self.prediction_type == "eps":
                squared_error = (
                    predicted_result[b][:, pred_os:pred_oe, ...]
                    - eps[b]
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
        x = self._extract_inputs(x)
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

        # Precompute text and null contexts for CFG
        text_context, metadata = self.text_module.get_context(
            x, generated_len, seq_len, device, self.param_dtype, training=False,
        )
        null_context = self.text_module.get_null_context(batch_size, device, self.param_dtype)
        full_text = metadata["full_text"]

        total_steps = self.time_scheduler.get_total_steps(seq_len)
        # Progressively advance from t=0 to t=max_t
        for step in range(total_steps):
            # get time steps and noise levels
            time_steps = self.time_scheduler.get_time_steps(device, generated_len, step)  # (B,)
            time_schedules, time_schedules_derivative = self.time_scheduler.get_time_schedules(device, generated_len, time_steps)  # (B, T)
            alpha, alpha_derivative, beta, beta_derivative = self.time_scheduler.get_noise_levels(device, generated_len, time_schedules)  # (B, T)
            input_start_index, input_end_index, output_start_index, output_end_index = self.time_scheduler.get_windows(generated_len, time_steps)
            _, _, xt = self.time_scheduler.add_noise(
                generated, alpha, beta, input_start_index, input_end_index, output_start_index, output_end_index, training=False
            )

            # Pad noise_level list into [B, seq_len] tensor for WanModel
            time_schedules_padded = torch.zeros(batch_size, seq_len, device=device)
            for i in range(batch_size):
                ts = time_schedules[i][input_start_index[i]:input_end_index[i]]
                time_schedules_padded[i, :len(ts)] = ts

            # CFG: text_scale * pred_text + null_scale * pred_null
            pred_text = self.model(
                xt,
                time_schedules_padded * self.time_embedding_scale,
                text_context, seq_len, y=None,
            )
            pred_null = self.model(
                xt,
                time_schedules_padded * self.time_embedding_scale,
                null_context, seq_len, y=None,
            )
            predicted_result = [
                self.cfg_config["text_scale"] * pt + self.cfg_config["null_scale"] * pn
                for pt, pn in zip(pred_text, pred_null)
            ]

            for i in range(batch_size):
                os, oe = output_start_index[i], output_end_index[i]
                pred_os = os - input_start_index[i]
                pred_oe = oe - input_start_index[i]
                predicted_result_i = predicted_result[i][:, pred_os:pred_oe, ...]
                generated_i = generated[i][:, os:oe, ...]
                dt = time_schedules_derivative[i][os:oe][None, :, None, None]
                alpha_val = alpha[i][os:oe][None, :, None, None]
                alpha_d = alpha_derivative[i][os:oe][None, :, None, None]
                beta_val = beta[i][os:oe][None, :, None, None]
                beta_d = beta_derivative[i][os:oe][None, :, None, None]
                if self.prediction_type == "vel":
                    vel = predicted_result_i
                elif self.prediction_type == "x0":
                    vel = predicted_result_i * (-beta_d / beta_val * alpha_val + alpha_d) + generated_i * (beta_d / beta_val)
                elif self.prediction_type == "eps":
                    vel = predicted_result_i * (-alpha_d / (alpha_val + dt) * beta_val + beta_d) + generated_i * (alpha_d / (alpha_val + dt))
                generated[i][:, os:oe, ...] += vel * dt

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
        self.time_scheduler = TriangularTimeScheduler(self.schedule_config)

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

        # Initialize streaming state for cross module
        self.text_module.init_stream(self.batch_size)


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
        self.text_module.trim_stream(self.seq_len)

    @torch.no_grad()
    def stream_generate_step(self, x):
        """
        Streaming generation step. Each call provides 1 frame of conditions.
        The scheduler determines committable frames from accumulated conditions.

        Returns:
            dict with "generated": list of one (N, C) tensor, or [] if nothing to commit.
        """
        x = self._extract_inputs(x)
        device = next(self.parameters()).device
        self.generated = [g.to(device) for g in self.generated]

        # 1. Update conditions (1 frame per call)
        self.text_module.update_stream(x, device, self.param_dtype)
        self.condition_frames += 1

        # 2. Rollback if conditions overflow the window
        if self.condition_frames > self.buf_len:
            self._rollback()

        # 3. Determine how many frames can be committed
        committable_length, committable_steps = self.time_scheduler.get_committable(self.condition_frames)
        while self.current_step < committable_steps:
            time_steps = self.time_scheduler.get_time_steps(
                device, [self.buf_len] * self.batch_size, self.current_step)
            time_schedules, time_schedules_derivative = (
                self.time_scheduler.get_time_schedules(
                    device, [self.buf_len] * self.batch_size, time_steps))
            alpha, alpha_derivative, beta, beta_derivative = self.time_scheduler.get_noise_levels(
                device, [self.buf_len] * self.batch_size, time_schedules)
            is_, ie_, os_, oe_ = self.time_scheduler.get_windows(
                [self.buf_len] * self.batch_size, time_steps)
            _, _, xt = self.time_scheduler.add_noise(
                self.generated, alpha, beta, is_, ie_, os_, oe_, training=False
            )

            noisy_input = [xt[i][:, -self.seq_len:, ...] for i in range(self.batch_size)]
            ts = time_schedules[0][is_[0]:ie_[0]][-self.seq_len:]
            cut_length = max(0, (ie_[0] - is_[0]) - self.seq_len)

            time_schedules_padded = torch.zeros(self.batch_size, self.seq_len, device=device)
            for i in range(self.batch_size):
                time_schedules_padded[i, :len(ts)] = ts

            # CFG: text_scale * pred_text + null_scale * pred_null
            text_context = self.text_module.get_stream_context(
                is_[0], ie_[0], self.seq_len
            )
            null_context = self.text_module.get_null_context(
                self.batch_size, device, self.param_dtype
            )
            pred_text = self.model(
                noisy_input,
                time_schedules_padded * self.time_embedding_scale,
                text_context, self.seq_len, y=None,
            )
            pred_null = self.model(
                noisy_input,
                time_schedules_padded * self.time_embedding_scale,
                null_context, self.seq_len, y=None,
            )
            predicted_result = [
                self.cfg_config["text_scale"] * pt + self.cfg_config["null_scale"] * pn
                for pt, pn in zip(pred_text, pred_null)
            ]

            os_idx, oe_idx = os_[0], oe_[0]
            dt = time_schedules_derivative[0][os_idx:oe_idx][None, :, None, None]
            pred_os_idx = os_idx - is_[0] - cut_length
            pred_oe_idx = oe_idx - is_[0] - cut_length
            alpha_val = alpha[0][os_idx:oe_idx][None, :, None, None]
            alpha_d = alpha_derivative[0][os_idx:oe_idx][None, :, None, None]
            beta_val = beta[0][os_idx:oe_idx][None, :, None, None]
            beta_d = beta_derivative[0][os_idx:oe_idx][None, :, None, None]

            for i in range(self.batch_size):
                predicted_result_i = predicted_result[i][:, pred_os_idx:pred_oe_idx, ...]
                generated_i = self.generated[i][:, os_idx:oe_idx, ...]
                if self.prediction_type == "vel":
                    vel = predicted_result_i
                elif self.prediction_type == "x0":
                    vel = predicted_result_i * (-beta_d / beta_val * alpha_val + alpha_d) + generated_i * (beta_d / beta_val)
                elif self.prediction_type == "eps":
                    vel = predicted_result_i * (-alpha_d / (alpha_val + dt) * beta_val + beta_d) + generated_i * (alpha_d / (alpha_val + dt))
                self.generated[i][:, os_idx:oe_idx, ...] += vel * dt
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
