"""Debug script: test streaming generation across rollback boundary.
Prints detailed state before/after rollback and compares with non-streaming output."""
import os
import sys
import math
import torch
import numpy as np
from lightning import seed_everything

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils.initialize import instantiate, load_config
from torch_ema import ExponentialMovingAverage

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    cfg = load_config()
    seed_everything(cfg.seed)

    # Load model
    model = instantiate(
        target=cfg.model.target, cfg=None, hfstyle=False, **cfg.model.params
    )
    checkpoint = torch.load(cfg.test_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if "ema_state" in checkpoint:
        ema = ExponentialMovingAverage(model.parameters(), decay=cfg.model.ema_decay)
        ema.load_state_dict(checkpoint["ema_state"])
        ema.copy_to(model.parameters())
    model.to(device)
    model.eval()

    scheduler = model.time_scheduler
    chunk_size = scheduler.chunk_size
    steps = scheduler.steps
    seq_len = 30
    buf_len = seq_len * 2
    batch_size = 1

    text = "walk in a circle."
    total_frames = 80  # enough to trigger rollback at frame 61

    os.makedirs("tmp", exist_ok=True)
    f = open("tmp/debug_rollback.txt", "w")

    def log(msg):
        f.write(msg + "\n")
        f.flush()

    log(f"=== Rollback Debug ===")
    log(f"chunk_size={chunk_size}, steps={steps}, seq_len={seq_len}, buf_len={buf_len}")
    log(f"Rollback triggers at condition_frames > {buf_len}")
    log(f"Total frames to generate: {total_frames}")
    log(f"")

    # ========== STREAMING GENERATION ==========
    log(f"{'='*80}")
    log(f"STREAMING GENERATION")
    log(f"{'='*80}")

    seed_everything(cfg.seed)
    model.init_generated(seq_len, batch_size=batch_size)

    ik = model.input_keys
    all_stream_frames = []

    with torch.no_grad():
        for frame_i in range(total_frames):
            condition_frame = frame_i + 1

            # State before this step
            cf_before = model.condition_frames
            cs_before = model.current_step
            cc_before = model.current_commit

            x_input = {ik["text"]: [text]}
            x_input = model._extract_inputs(x_input)
            model.text_module.update_stream(x_input, device, model.param_dtype)
            model.condition_frames += 1

            # Check rollback
            rollback_happened = False
            if model.condition_frames > model.buf_len:
                # Log state BEFORE rollback
                log(f"\n{'#'*80}")
                log(f"ROLLBACK at condition_frame={condition_frame}")
                log(f"  BEFORE rollback:")
                log(f"    condition_frames={model.condition_frames}, current_step={model.current_step}, current_commit={model.current_commit}")

                # Log generated buffer state before rollback
                gen_first_half = model.generated[0][:, :seq_len, ...]
                gen_second_half = model.generated[0][:, seq_len:, ...]
                log(f"    generated[0:seq_len] mean={gen_first_half.mean().item():.6f}, std={gen_first_half.std().item():.6f}")
                log(f"    generated[seq_len:buf_len] mean={gen_second_half.mean().item():.6f}, std={gen_second_half.std().item():.6f}")

                # Check committed vs non-committed regions
                log(f"    committed region [0:{model.current_commit}] mean={model.generated[0][:, :model.current_commit].mean().item():.6f}")
                if model.current_commit < buf_len:
                    log(f"    non-committed region [{model.current_commit}:{model.condition_frames}] mean={model.generated[0][:, model.current_commit:model.condition_frames].mean().item():.6f}")

                model._rollback()
                rollback_happened = True

                log(f"  AFTER rollback:")
                log(f"    condition_frames={model.condition_frames}, current_step={model.current_step}, current_commit={model.current_commit}")
                gen_first_half = model.generated[0][:, :seq_len, ...]
                gen_second_half = model.generated[0][:, seq_len:, ...]
                log(f"    generated[0:seq_len] mean={gen_first_half.mean().item():.6f}, std={gen_first_half.std().item():.6f}")
                log(f"    generated[seq_len:buf_len] mean={gen_second_half.mean().item():.6f}, std={gen_second_half.std().item():.6f}")
                log(f"{'#'*80}\n")

            committable_length, committable_steps = scheduler.get_committable(model.condition_frames)

            # Log around rollback boundary
            near_rollback = (55 <= condition_frame <= 70)
            if near_rollback:
                log(f"condition_frame={condition_frame}, condition_frames={model.condition_frames}, "
                    f"current_step={model.current_step}, committable_steps={committable_steps}, "
                    f"committable_length={committable_length}, current_commit={model.current_commit}"
                    f"{' [ROLLBACK]' if rollback_happened else ''}")

            step_count = 0
            while model.current_step < committable_steps:
                step = model.current_step
                time_steps = scheduler.get_time_steps(
                    device, [model.buf_len] * batch_size, step)
                time_schedules, time_schedules_derivative = scheduler.get_time_schedules(
                    device, [model.buf_len] * batch_size, time_steps)
                noise_level, noise_level_derivative = scheduler.get_noise_levels(
                    device, [model.buf_len] * batch_size, time_schedules)
                is_, ie_, os_, oe_ = scheduler.get_windows(
                    [model.buf_len] * batch_size, time_steps)

                _, _, xt = scheduler.add_noise(
                    model.generated, noise_level, is_, ie_, os_, oe_, training=False)

                noisy_input = [xt[i][:, -seq_len:, ...] for i in range(batch_size)]
                ts = time_schedules[0][is_[0]:ie_[0]][-seq_len:]
                cut_length = max(0, (ie_[0] - is_[0]) - seq_len)

                os_idx, oe_idx = os_[0], oe_[0]
                pred_os_idx = os_idx - is_[0] - cut_length
                pred_oe_idx = oe_idx - is_[0] - cut_length

                if near_rollback and step_count == 0:
                    log(f"  first_step={step}, t={time_steps[0].item():.4f}, "
                        f"is={is_[0]}, ie={ie_[0]}, os={os_idx}, oe={oe_idx}, cut={cut_length}, "
                        f"pred_os={pred_os_idx}, pred_oe={pred_oe_idx}")
                    ts_out = time_schedules[0][os_idx:oe_idx]
                    nl_out = noise_level[0][os_idx:oe_idx]
                    log(f"    ts_output: {ts_out.tolist()}")
                    log(f"    noise_level_output: {nl_out.tolist()}")

                # Run model step
                time_schedules_padded = torch.zeros(batch_size, seq_len, device=device)
                for i in range(batch_size):
                    time_schedules_padded[i, :len(ts)] = ts

                text_context = model.text_module.get_stream_context(
                    is_[0], ie_[0], seq_len)
                null_context = model.text_module.get_null_context(
                    batch_size, device, model.param_dtype)
                pred_text = model.model(
                    noisy_input,
                    time_schedules_padded * model.time_embedding_scale,
                    text_context, seq_len, y=None)
                pred_null = model.model(
                    noisy_input,
                    time_schedules_padded * model.time_embedding_scale,
                    null_context, seq_len, y=None)
                predicted_result = [
                    model.cfg_config["text_scale"] * pt + model.cfg_config["null_scale"] * pn
                    for pt, pn in zip(pred_text, pred_null)]

                dt = time_schedules_derivative[0][os_idx:oe_idx][None, :, None, None]
                nl = noise_level[0][os_idx:oe_idx][None, :, None, None]
                nld = noise_level_derivative[0][os_idx:oe_idx][None, :, None, None]

                for i in range(batch_size):
                    predicted_result_i = predicted_result[i]
                    if model.prediction_type == "vel":
                        predicted_vel = predicted_result_i[:, pred_os_idx:pred_oe_idx, ...]
                    elif model.prediction_type == "x0":
                        predicted_vel = (
                            model.generated[i][:, os_idx:oe_idx, ...]
                            - predicted_result_i[:, pred_os_idx:pred_oe_idx, ...]
                        ) / nl * nld
                    elif model.prediction_type == "noise":
                        predicted_vel = (
                            predicted_result_i[:, pred_os_idx:pred_oe_idx, ...]
                            - model.generated[i][:, os_idx:oe_idx, ...]
                        ) / (1 - nl + dt) * nld
                    model.generated[i][:, os_idx:oe_idx, ...] += predicted_vel * dt
                model.current_step += 1
                step_count += 1

            # Commit
            if model.current_commit < committable_length:
                output = [model.generated[i][:, model.current_commit:committable_length, ...]
                          for i in range(batch_size)]
                output = model.postprocess(output)
                output = [o * model.std + model.mean for o in output]
                new_frames = committable_length - model.current_commit
                for o in output:
                    for fi in range(o.shape[0]):
                        all_stream_frames.append(o[fi].cpu())
                if near_rollback:
                    log(f"  COMMIT [{model.current_commit}:{committable_length}] = {new_frames} frames, "
                        f"output mean={output[0].mean().item():.4f}, std={output[0].std().item():.4f}")
                model.current_commit = committable_length

    log(f"\nTotal stream frames: {len(all_stream_frames)}")

    # ========== NON-STREAMING GENERATION ==========
    log(f"\n{'='*80}")
    log(f"NON-STREAMING GENERATION (for comparison)")
    log(f"{'='*80}")

    seed_everything(cfg.seed)
    with torch.no_grad():
        gen_len = total_frames
        x_gen = {
            model.input_keys["feature_length"]: torch.tensor([gen_len]),
            model.input_keys["text"]: [text],
        }
        out = model.generate(x_gen)
        nonstream_frames = out["generated"][0]  # (T, C)
        log(f"Non-streaming output shape: {nonstream_frames.shape}")
        log(f"Non-streaming output mean={nonstream_frames.mean().item():.4f}, std={nonstream_frames.std().item():.4f}")

    # ========== COMPARE ==========
    log(f"\n{'='*80}")
    log(f"FRAME-BY-FRAME COMPARISON (stream vs non-stream)")
    log(f"{'='*80}")

    stream_tensor = torch.stack(all_stream_frames, dim=0)  # (T, C)
    min_len = min(len(all_stream_frames), nonstream_frames.shape[0])
    log(f"Stream frames: {len(all_stream_frames)}, Non-stream frames: {nonstream_frames.shape[0]}")
    log(f"Comparing first {min_len} frames")

    nonstream_cpu = nonstream_frames.cpu()

    for i in range(min_len):
        s = stream_tensor[i]
        ns = nonstream_cpu[i]
        diff = (s - ns).abs().mean().item()
        if i < 10 or (55 <= i <= 70) or i >= min_len - 5:
            log(f"  Frame {i:3d}: stream_mean={s.mean().item():8.4f}, nonstream_mean={ns.mean().item():8.4f}, "
                f"abs_diff={diff:.4f}")

    # Per-frame diff summary
    diffs = [(stream_tensor[i] - nonstream_cpu[i]).abs().mean().item() for i in range(min_len)]
    log(f"\nDiff summary:")
    log(f"  Mean diff: {np.mean(diffs):.4f}")
    log(f"  Max diff: {np.max(diffs):.4f} at frame {np.argmax(diffs)}")
    log(f"  Min diff: {np.min(diffs):.4f}")

    # Check for discontinuity around rollback (frame ~56)
    if len(all_stream_frames) > 60:
        log(f"\nStream output continuity around rollback boundary:")
        for i in range(max(0, 50), min(len(all_stream_frames), 65)):
            s = stream_tensor[i]
            if i > 0:
                prev = stream_tensor[i-1]
                frame_diff = (s - prev).abs().mean().item()
                log(f"  Frame {i}: mean={s.mean().item():8.4f}, delta_from_prev={frame_diff:.4f}")
            else:
                log(f"  Frame {i}: mean={s.mean().item():8.4f}")

    f.close()
    print(f"Rollback debug written to tmp/debug_rollback.txt")
    print(f"Stream: {len(all_stream_frames)} frames, Non-stream: {nonstream_frames.shape[0]} frames")


if __name__ == "__main__":
    main()
