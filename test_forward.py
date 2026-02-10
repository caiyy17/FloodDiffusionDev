import sys
import torch
import torch.nn as nn

from models.diffusion_forcing_wan import (
    DiffForcingWanModel,
    register_cross_module,
    CROSS_MODULE_REGISTRY,
)


@register_cross_module
class DummyCrossModule(nn.Module):
    """Dummy cross module for testing (no T5 needed)."""

    def __init__(self, len=16, dim=64, **kwargs):
        super().__init__()
        self.cross_len = len
        self.cross_dim = dim
        self.cross_attn_norm = True

    def encode(self, text_list, device):
        return [
            torch.randn(self.cross_len, self.cross_dim, device=device)
            for _ in text_list
        ]

    def get_context(self, x, valid_len, seq_len, device, param_dtype, training=False):
        batch_size = len(valid_len)
        context = [
            torch.randn(self.cross_len, self.cross_dim, device=device, dtype=param_dtype)
            for _ in range(batch_size)
        ]
        return context, {}

    def get_null_context(self, batch_size, device, param_dtype):
        return [
            torch.randn(self.cross_len, self.cross_dim, device=device, dtype=param_dtype)
            for _ in range(batch_size)
        ]


# Create model with small dims for fast testing
device = torch.device("cuda:0")
model = DiffForcingWanModel(
    input_dim=4,
    hidden_dim=128,
    ffn_dim=256,
    freq_dim=64,
    num_heads=4,
    num_layers=2,
    crossmodules=[{"name": "DummyCrossModule", "len": 16, "dim": 64}],
).to(device)

# Create dummy input
batch_size = 2
seq_len = 20
x = {
    "feature": torch.randn(batch_size, seq_len, 4, device=device),
    "feature_length": torch.tensor([15, 20]),
    "text": ["hello world", "test text"],
}

# Test forward
print("Testing forward...")
try:
    loss_dict = model(x)
    print(f"Forward succeeded! Loss: {loss_dict}")
    # Test backward
    loss_dict["total"].backward()
    grad_norms = {name: p.grad.norm().item() for name, p in model.named_parameters() if p.grad is not None}
    print(f"Backward succeeded! {len(grad_norms)} params have gradients")
except Exception as e:
    import traceback
    traceback.print_exc()

# Test generate
print("\nTesting generate...")
model.zero_grad()
try:
    with torch.no_grad():
        out = model.generate(x)
    print(f"Generate succeeded! Output keys: {list(out.keys())}")
    for i, g in enumerate(out["generated"]):
        print(f"  generated[{i}] shape: {g.shape}")
except Exception as e:
    import traceback
    traceback.print_exc()
