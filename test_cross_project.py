"""Cross-project comparison: FloodDiffusion (old) vs FloodDiffusionDev (new)."""
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
print(f"Old: shape={{result.shape}}, norm={{result.norm():.6f}}")
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

torch.manual_seed(42)
torch.cuda.manual_seed(42)

x = {{
    "feature_length": torch.tensor([50]),
    "text": ["a person walks forward and then sits down"],
}}
with torch.no_grad():
    out = model.generate(x, schedule_config={{"extra_len": 5}})
result = out["generated"][0].cpu()
torch.save(result, sys.argv[1])
print(f"New: shape={{result.shape}}, norm={{result.norm():.6f}}")
'''

def run_script(script, out_path):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        f.flush()
        result = subprocess.run(
            [PYTHON, f.name, out_path],
            capture_output=True, text=True,
        )
        os.unlink(f.name)
    print(result.stdout.strip())
    if result.returncode != 0:
        lines = result.stderr.strip().split('\n')
        print("ERROR:", '\n'.join(lines[-10:]))
        return False
    return True

old_path = "/tmp/cross_old_result.pt"
new_path = "/tmp/cross_new_result.pt"

print("=== Running old model (FloodDiffusion) ===")
ok1 = run_script(OLD_SCRIPT, old_path)

print("\n=== Running new model (FloodDiffusionDev) ===")
ok2 = run_script(NEW_SCRIPT, new_path)

if ok1 and ok2:
    old_result = torch.load(old_path, weights_only=True)
    new_result = torch.load(new_path, weights_only=True)
    print(f"\n=== Comparison ===")
    print(f"Old shape: {old_result.shape}, New shape: {new_result.shape}")
    min_len = min(old_result.shape[0], new_result.shape[0])
    old_r = old_result[:min_len]
    new_r = new_result[:min_len]
    diff = (old_r - new_r).abs()
    print(f"max_diff: {diff.max():.6e}")
    print(f"mean_diff: {diff.mean():.6e}")
    if diff.max() < 1e-3:
        print("PASS!")
    else:
        print(f"FAIL: outputs differ")
