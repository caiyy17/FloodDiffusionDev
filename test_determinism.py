"""Test if flash attention non-determinism causes output variation."""
import subprocess, sys, torch, tempfile, os

PYTHON = "/home/cyy/miniforge3/envs/motion_gen/bin/python"
DEPS = "/home/cyy/Documents/Projects/FloodDiffusion/deps"

SCRIPT = f'''
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

torch.manual_seed(42)
torch.cuda.manual_seed(42)
x = {{
    "feature_length": torch.tensor([50]),
    "text": ["a person walks forward and then sits down"],
}}
with torch.no_grad():
    out = model.generate(x)
result = out["generated"][0].cpu()
torch.save(result, sys.argv[1])
print(f"norm={{result.norm():.10f}}")
'''

def run(out_path):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(SCRIPT)
        f.flush()
        r = subprocess.run([PYTHON, f.name, out_path], capture_output=True, text=True)
        os.unlink(f.name)
    print(r.stdout.strip())
    return r.returncode == 0

print("Run 1:")
ok1 = run("/tmp/det_run1.pt")
print("Run 2:")
ok2 = run("/tmp/det_run2.pt")

if ok1 and ok2:
    r1 = torch.load("/tmp/det_run1.pt", weights_only=True)
    r2 = torch.load("/tmp/det_run2.pt", weights_only=True)
    diff = (r1 - r2).abs()
    print(f"\nSame code, two runs:")
    print(f"  max_diff: {diff.max():.6e}")
    print(f"  mean_diff: {diff.mean():.6e}")
    if diff.max() == 0:
        print("  Perfectly deterministic")
    else:
        print("  Non-deterministic (flash attention)")
