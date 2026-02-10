"""Trace old generate loop step by step, save intermediates."""
import sys, torch, numpy as np
sys.path.insert(0, "/home/cyy/Documents/Projects/FloodDiffusion")
from models.diffusion_forcing_wan import DiffForcingWanModel

device = torch.device("cuda")
model = DiffForcingWanModel(
    checkpoint_path="deps/t5_umt5-xxl-enc-bf16/models_t5_umt5-xxl-enc-bf16.pth",
    tokenizer_path="deps/t5_umt5-xxl-enc-bf16/google/umt5-xxl",
    input_dim=4, noise_steps=10,
)
ckpt = torch.load("outputs/20251106_063218_ldf/step_step=50000.ckpt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model = model.to(device).eval()

seq_len = 50
chunk_size = model.chunk_size
noise_steps = model.noise_steps
batch_size = 1
total_len = seq_len + chunk_size  # 55

torch.manual_seed(42); np.random.seed(42); torch.cuda.manual_seed(42)
generated = torch.randn(batch_size, total_len, model.input_dim, device=device)
generated = model.preprocess(generated)

text_ctx = model.encode_text_with_cache(["a person walks forward and then sits down"], device)
text_ctx = [u.to(model.param_dtype) for u in text_ctx]
text_null = model.encode_text_with_cache([""], device)
text_null = [u.to(model.param_dtype) for u in text_null]

max_t = 1 + (seq_len - 1) / chunk_size
dt = 1 / noise_steps
total_steps = int(max_t / dt)

trace = []
with torch.no_grad():
    for step in range(total_steps):
        t = step * dt
        start_index = max(0, int(chunk_size * (t - 1)) + 1)
        end_index = int(chunk_size * t) + 1
        time_steps = torch.full((batch_size,), t, device=device)
        noise_level = model._get_noise_levels(device, total_len, time_steps)

        noisy_input = [generated[0, :, :end_index, ...]]
        pred = model.model(noisy_input, noise_level * model.time_embedding_scale,
                           text_ctx, total_len, y=None)
        if model.cfg_scale != 1.0:
            pred_null = model.model(noisy_input, noise_level * model.time_embedding_scale,
                                    text_null, total_len, y=None)
            pred = [model.cfg_scale * pv - (model.cfg_scale - 1) * pvn
                    for pv, pvn in zip(pred, pred_null)]

        predicted_vel = pred[0][:, start_index:end_index, ...]
        generated[0, :, start_index:end_index, ...] += predicted_vel * dt

        if step < 5 or step % 20 == 0 or step == total_steps - 1:
            trace.append({
                "step": step, "t": t,
                "start": start_index, "end": end_index,
                "noise_level_slice": noise_level[0, :min(end_index+2, total_len)].cpu(),
                "noise_level_full_passed": True,
                "pred_vel_norm": predicted_vel.norm().item(),
                "gen_slice": generated[0, :, start_index:end_index, ...].clone().cpu(),
            })

# Final output
generated_out = model.postprocess(generated)
result = generated_out[0, :seq_len, :]

torch.save({"trace": trace, "generated": result.cpu(), "total_steps": total_steps}, "/tmp/old_trace.pt")
print(f"Old: total_steps={total_steps}, max_t={max_t}, dt={dt}, total_len={total_len}")
for t in trace:
    print(f"  step={t['step']:3d} t={t['t']:.4f} start={t['start']:2d} end={t['end']:2d} "
          f"nl={t['noise_level_slice'][:6].tolist()} vel_norm={t['pred_vel_norm']:.4f}")
