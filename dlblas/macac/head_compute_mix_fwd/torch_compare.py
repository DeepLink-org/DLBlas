import torch
import time

# Config matching the MACA kernel
batch_size = 16
n1 = 16384
mhc_mult = 4
mhc_pre_eps = 1e-2

# Use MACA device
device = "cuda:0"

# Create inputs (matching C++ test init pattern)
input_mix = torch.tensor([[(i * 7) % 127 / 10.0 - 6.0 for i in range(batch_size * n1 * mhc_mult)]], 
                         dtype=torch.float32, device=device).reshape(batch_size, n1, mhc_mult)
mhc_scale = torch.tensor([1.5], dtype=torch.float32, device=device)
mhc_base = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float32, device=device)

# Warmup
for _ in range(10):
    output = torch.sigmoid(input_mix * mhc_scale + mhc_base) + mhc_pre_eps
torch.cuda.synchronize()

# Benchmark
test_count = 500
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(test_count):
    output = torch.sigmoid(input_mix * mhc_scale + mhc_base) + mhc_pre_eps
end.record()
torch.cuda.synchronize()

torch_time = start.elapsed_time(end) / test_count
print(f"Torch operator average time: {torch_time:.6f} ms")
print(f"<torch_time>{torch_time:.6f} ms</torch_time>")

# Verify correctness against reference
ref = torch.sigmoid(input_mix * mhc_scale + mhc_base) + mhc_pre_eps
max_diff = (output - ref).abs().max().item()
print(f"Max diff (self-consistency): {max_diff:.10f}")
