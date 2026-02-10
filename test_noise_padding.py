"""Test: does noise_level padding cause the difference?
Run new model with FULL noise_level (like old) vs padded noise_level."""
import subprocess, sys, torch, tempfile, os

PYTHON = "/home/cyy/miniforge3/envs/motion_gen/bin/python"
DEPS = "/home/cyy/Documents/Projects/FloodDiffusion/deps"

# New model, but compute noise_level the OLD way (full, no padding)
SCRIPT_FULL_NL = f'''
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

seq_len = 55
chunk_size = 5
noise_steps = 10
batch_size = 1

torch.manual_seed(42)
torch.cuda.manual_seed(42)
generated = torch.randn(batch_size, seq_len, model.input_dim, device=device)
generated = [generated[i] for i in range(batch_size)]
generated = model.preprocess(generated)

x = {{
    "feature_length": torch.tensor([50]),
    "text": ["a person walks forward and then sits down"],
}}
generated_len = [seq_len]
all_contexts, metadata = model._get_all_contexts(x, generated_len, seq_len, device, training=False)
all_null_contexts = model._get_all_null_contexts(batch_size, device)

dt = 1.0 / noise_steps
max_t = 1 + (50 - 1) / chunk_size
total_steps = int(max_t / dt)  # 108 like old

use_full_nl = (sys.argv[2] == "full")

with torch.no_grad():
    for step in range(total_steps):
        t = step * dt
        start_index = max(0, int(chunk_size * (t - 1)) + 1)
        end_index = int(chunk_size * t) + 1

        noisy_input = [generated[0][:, :end_index, ...]]

        if use_full_nl:
            # OLD way: compute full noise_level for all positions
            noise_level = torch.clamp(
                1 + torch.arange(seq_len, device=device).float() / chunk_size - t,
                min=0.0, max=1.0,
            ).unsqueeze(0)  # [1, 55]
        else:
            # NEW way: padded
            time_steps = [torch.tensor(t, device=device, dtype=torch.float64)]
            time_schedules, _ = model.time_scheduler.get_time_schedules(device, generated_len, time_steps)
            noise_level_list, _ = model.time_scheduler.get_noise_levels(device, generated_len, time_schedules)
            noise_level = torch.zeros(batch_size, seq_len, device=device)
            nl = noise_level_list[0][:end_index]
            noise_level[0, :len(nl)] = nl.float()

        pred = model.model(noisy_input, noise_level * model.time_embedding_scale,
                           all_contexts, seq_len, y=None)
        if model.cfg_scale != 1.0:
            pred_null = model.model(noisy_input, noise_level * model.time_embedding_scale,
                                    all_null_contexts, seq_len, y=None)
            pred = [model.cfg_scale * pv - (model.cfg_scale - 1) * pvn
                    for pv, pvn in zip(pred, pred_null)]

        predicted_vel = pred[0][:, start_index:end_index, ...]
        generated[0][:, start_index:end_index, ...] += predicted_vel * dt

generated_out = model.postprocess(generated)
result = generated_out[0][:50, :]
torch.save(result.cpu(), sys.argv[1])
mode = "FULL noise_level" if use_full_nl else "PADDED noise_level"
print(f"New ({{mode}}): norm={{result.norm():.10f}}")
'''

def run(script, out_path, extra_args=""):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        f.flush()
        r = subprocess.run([PYTHON, f.name, out_path] + extra_args.split(),
                          capture_output=True, text=True)
        os.unlink(f.name)
    print(r.stdout.strip())
    if r.returncode != 0:
        print("ERROR:", r.stderr[-500:])
        return False
    return True

# Also run old model for comparison
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
torch.manual_seed(42); torch.cuda.manual_seed(42)
x = {{"feature_length": torch.tensor([50]), "text": ["a person walks forward and then sits down"]}}
with torch.no_grad():
    out = model.generate(x)
result = out["generated"][0].cpu()
torch.save(result, sys.argv[1])
print(f"Old: norm={{result.norm():.10f}}")
'''

print("=== Old model ===")
ok0 = run(OLD_SCRIPT, "/tmp/np_old.pt")
print("\n=== New model, FULL noise_level ===")
ok1 = run(SCRIPT_FULL_NL, "/tmp/np_full.pt", "full")
print("\n=== New model, PADDED noise_level ===")
ok2 = run(SCRIPT_FULL_NL, "/tmp/np_padded.pt", "padded")

if ok0 and ok1 and ok2:
    old = torch.load("/tmp/np_old.pt", weights_only=True)
    full = torch.load("/tmp/np_full.pt", weights_only=True)
    padded = torch.load("/tmp/np_padded.pt", weights_only=True)

    print(f"\n=== Results ===")
    print(f"Old vs New(full):    max_diff={( old - full).abs().max():.6e}")
    print(f"Old vs New(padded):  max_diff={(old - padded).abs().max():.6e}")
    print(f"Full vs Padded:      max_diff={(full - padded).abs().max():.6e}")
