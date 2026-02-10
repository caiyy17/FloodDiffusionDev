"""Trace step-by-step and compare intermediates between old and new."""
import subprocess, sys, torch, tempfile, os

PYTHON = "/home/cyy/miniforge3/envs/motion_gen/bin/python"
DEPS = "/home/cyy/Documents/Projects/FloodDiffusion/deps"

OLD_SCRIPT = f'''
import sys, torch
sys.path.insert(0, "/home/cyy/Documents/Projects/FloodDiffusion")
from models.diffusion_forcing_wan import DiffForcingWanModel

device = torch.device("cuda")
model = DiffForcingWanModel(
    checkpoint_path="{DEPS}/t5_umt5-xxl-enc-bf16/models_t5_umt5-xxl-enc-bf16.pth",
    tokenizer_path="{DEPS}/t5_umt5-xxl-enc-bf16/google/umt5-xxl",
    input_dim=4, noise_steps=10,
)
ckpt = torch.load(
    "/home/cyy/Documents/Projects/FloodDiffusion/outputs/20251106_063218_ldf/step_step=50000.ckpt",
    map_location="cpu", weights_only=False,
)
model.load_state_dict(ckpt["state_dict"])
model = model.to(device).eval()

seq_len = 50
chunk_size = model.chunk_size
noise_steps = model.noise_steps
batch_size = 1
total_len = seq_len + chunk_size

torch.manual_seed(42)
torch.cuda.manual_seed(42)
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

        if step < 3 or step % 10 == 0 or step == total_steps - 1:
            trace.append({{
                "step": step, "t": t,
                "start": start_index, "end": end_index,
                "noise_level": noise_level[0, :min(end_index+2, total_len)].cpu().float(),
                "pred_norm": pred[0].norm().item(),
                "vel_norm": predicted_vel.norm().item(),
                "gen_norm": generated[0].norm().item(),
            }})

generated_out = model.postprocess(generated)
result = generated_out[0, :seq_len, :]
torch.save({{"trace": trace, "result": result.cpu(), "total_steps": total_steps}}, sys.argv[1])
print(f"Old: total_steps={{total_steps}}")
'''

NEW_SCRIPT = f'''
import sys, torch
sys.path.insert(0, "/home/cyy/Documents/Projects/FloodDiffusionDev")
from models.diffusion_forcing_wan import DiffForcingWanModel

device = torch.device("cuda")
model = DiffForcingWanModel(
    input_dim=4,
    crossmodules=[dict(
        name="T5TextCrossModule",
        len=512, dim=4096,
        checkpoint_path="{DEPS}/t5_umt5-xxl-enc-bf16/models_t5_umt5-xxl-enc-bf16.pth",
        tokenizer_path="{DEPS}/t5_umt5-xxl-enc-bf16/google/umt5-xxl",
    )],
)
ckpt = torch.load(
    "/home/cyy/Documents/Projects/FloodDiffusionDev/outputs/20251106_063218_ldf/step_step=50000_converted.ckpt",
    map_location="cpu", weights_only=False,
)
model.load_state_dict(ckpt["state_dict"])
model = model.to(device).eval()

seq_len_raw = 50
extra_len = 5
seq_len = seq_len_raw + extra_len  # 55
chunk_size = model.schedule_config["chunk_size"]
noise_steps = model.schedule_config["steps"]
batch_size = 1

torch.manual_seed(42)
torch.cuda.manual_seed(42)
generated = torch.randn(batch_size, seq_len, model.input_dim, device=device)
generated = [generated[i] for i in range(batch_size)]
generated = model.preprocess(generated)

# Get contexts
x = {{
    "feature_length": torch.tensor([seq_len_raw]),
    "text": ["a person walks forward and then sits down"],
}}
generated_len = [seq_len]
all_contexts, metadata = model._get_all_contexts(x, generated_len, seq_len, device, training=False)
all_null_contexts = model._get_all_null_contexts(batch_size, device)

total_steps = model.time_scheduler.get_total_steps(seq_len)
trace = []

with torch.no_grad():
    for step in range(total_steps):
        time_steps = model.time_scheduler.get_time_steps(device, generated_len, step)
        time_schedules, time_schedules_derivative = model.time_scheduler.get_time_schedules(device, generated_len, time_steps)
        noise_level, noise_level_derivative = model.time_scheduler.get_noise_levels(device, generated_len, time_schedules)
        input_start_index, input_end_index, output_start_index, output_end_index = model.time_scheduler.get_windows(generated_len, time_steps)

        noisy_input = [generated[i][:, input_start_index[i]:input_end_index[i], ...] for i in range(batch_size)]

        noise_level_padded = torch.zeros(batch_size, seq_len, device=device)
        for i in range(batch_size):
            nl = noise_level[i][:input_end_index[i]]
            noise_level_padded[i, :len(nl)] = nl.float()

        pred = model.model(noisy_input, noise_level_padded * model.time_embedding_scale,
                           all_contexts, seq_len, y=None)
        if model.cfg_scale != 1.0:
            pred_null = model.model(noisy_input, noise_level_padded * model.time_embedding_scale,
                                    all_null_contexts, seq_len, y=None)
            pred = [model.cfg_scale * pv - (model.cfg_scale - 1) * pvn
                    for pv, pvn in zip(pred, pred_null)]

        for i in range(batch_size):
            os_i, oe_i = output_start_index[i], output_end_index[i]
            dt = time_schedules_derivative[i][os_i:oe_i][None, :, None, None]
            predicted_vel = pred[i][:, os_i:oe_i, ...]
            generated[i][:, os_i:oe_i, ...] += predicted_vel * dt

        if step < 3 or step % 10 == 0 or step == total_steps - 1:
            trace.append({{
                "step": step, "t": time_steps[0].item(),
                "start": output_start_index[0], "end": output_end_index[0],
                "noise_level": noise_level_padded[0, :min(input_end_index[0]+2, seq_len)].cpu(),
                "pred_norm": pred[0].norm().item(),
                "vel_norm": predicted_vel.norm().item(),
                "gen_norm": generated[0].norm().item(),
            }})

generated_out = model.postprocess(generated)
result = generated_out[0][:seq_len_raw, :]
torch.save({{"trace": trace, "result": result.cpu(), "total_steps": total_steps}}, sys.argv[1])
print(f"New: total_steps={{total_steps}}")
'''

def run_script(script, out_path):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        f.flush()
        result = subprocess.run([PYTHON, f.name, out_path], capture_output=True, text=True)
        os.unlink(f.name)
    print(result.stdout.strip())
    if result.returncode != 0:
        lines = result.stderr.strip().split('\n')
        print("ERROR:", '\n'.join(lines[-10:]))
        return False
    return True

old_path = "/tmp/trace_old.pt"
new_path = "/tmp/trace_new.pt"

print("=== Running old model ===")
ok1 = run_script(OLD_SCRIPT, old_path)
print("\n=== Running new model ===")
ok2 = run_script(NEW_SCRIPT, new_path)

if ok1 and ok2:
    old_data = torch.load(old_path, weights_only=False)
    new_data = torch.load(new_path, weights_only=False)

    print(f"\n=== Trace Comparison (old={old_data['total_steps']} steps, new={new_data['total_steps']} steps) ===")
    print(f"{'step':>5} | {'t':>8} | {'window':>10} | {'nl_diff':>10} | {'pred_diff':>10} | {'vel_diff':>10} | {'gen_diff':>10}")

    for ot, nt in zip(old_data['trace'], new_data['trace']):
        if ot['step'] != nt['step']:
            break
        min_nl = min(len(ot['noise_level']), len(nt['noise_level']))
        nl_diff = (ot['noise_level'][:min_nl] - nt['noise_level'][:min_nl]).abs().max().item()
        pred_diff = abs(ot['pred_norm'] - nt['pred_norm'])
        vel_diff = abs(ot['vel_norm'] - nt['vel_norm'])
        gen_diff = abs(ot['gen_norm'] - nt['gen_norm'])
        w_match = "OK" if ot['start'] == nt['start'] and ot['end'] == nt['end'] else "DIFF"
        print(f"{ot['step']:5d} | {ot['t']:8.4f} | [{ot['start']:2d}:{ot['end']:2d}] {w_match:>4} | {nl_diff:10.2e} | {pred_diff:10.2e} | {vel_diff:10.2e} | {gen_diff:10.2e}")

    diff = (old_data['result'] - new_data['result']).abs()
    print(f"\nFinal result max_diff: {diff.max():.6e}")
