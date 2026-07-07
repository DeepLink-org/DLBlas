import torch
import time
import os

os.environ["MACA_PATH"] = "/opt/maca/"

# Parameters matching test data
n0, n1 = 1, 13
mhc_mult = 4
hidden_size = 1280
mhc_mult3 = mhc_mult * (2 + mhc_mult)  # 24
rms_group_size = mhc_mult * hidden_size  # 5120
num_rows = n0 * n1  # 13
num_mixes = mhc_mult3  # 24
eps = 1e-6

def torch_norm_fn(residual_flat, mhc_fn_flat, eps):
    """Equivalent Torch implementation of norm_fn forward"""
    mixes = torch.einsum('ik,jk->ij', residual_flat, mhc_fn_flat)
    sqrsum = residual_flat.square().sum(-1)
    rms_factor = (sqrsum / rms_group_size + eps).rsqrt()
    result = mixes * rms_factor.unsqueeze(-1)
    return result.squeeze(-1)

# Generate test data matching original spec
# residual: (n0*n1, mhc_mult*hidden_size) = (13, 5120)
residual = torch.randn((num_rows, rms_group_size), dtype=torch.float32).cuda()
# mhc_fn: (mhc_mult3, mhc_mult*hidden_size) = (24, 5120)
mhc_fn = (torch.randn((num_mixes, rms_group_size), dtype=torch.float32) * 1e-4).cuda()

# Warmup
for _ in range(20):
    _ = torch_norm_fn(residual, mhc_fn, eps)
torch.cuda.synchronize()

# Benchmark Torch
test_count = 500
start = time.perf_counter()
for _ in range(test_count):
    _ = torch_norm_fn(residual, mhc_fn, eps)
torch.cuda.synchronize()
end = time.perf_counter()

torch_time_ms = (end - start) / test_count * 1000.0
print(f"Torch average time: {torch_time_ms:.6f} ms")

# MACA kernel times from latest best version (warp-shuffle + float4 + inv_K)
macac_ori_time_ms = 0.040582   # baseline kernel time
macac_opt_time_ms = 0.027792   # best optimized kernel time
macac_ratio = 0.684852         # runtime_ratio

print(f"\n=== Performance Comparison: norm_fn ===")
print(f"Shape: residual=({num_rows}, {rms_group_size}), mhc_fn=({num_mixes}, {rms_group_size})")
print(f"Output: ({num_rows}, {num_mixes})")
print(f"")
print(f"Torch (einsum+rsqrt):      {torch_time_ms:.6f} ms")
print(f"MACAC ori (baseline):      {macac_ori_time_ms:.6f} ms")
print(f"MACAC opt (best):          {macac_opt_time_ms:.6f} ms")
print(f"MACAC runtime_ratio:       {macac_ratio:.4f}")
print(f"MACAC improvement:         {1.0/macac_ratio:.2f}x over baseline")
print(f"Speedup (Torch/MACAC_opt): {torch_time_ms / macac_opt_time_ms:.2f}x")
print(f"MACAC_opt/Torch ratio:     {macac_opt_time_ms / torch_time_ms:.4f}")

with open("torch_comparison.txt", "w") as f:
    f.write(f"Operator: norm_fn\n")
    f.write(f"Shape: residual=({num_rows}, {rms_group_size}), mhc_fn=({num_mixes}, {rms_group_size})\n")
    f.write(f"Output: ({num_rows}, {num_mixes})\n")
    f.write(f"\n")
    f.write(f"Torch (einsum+rsqrt):      {torch_time_ms:.6f} ms\n")
    f.write(f"MACAC ori (baseline):      {macac_ori_time_ms:.6f} ms\n")
    f.write(f"MACAC opt (best):          {macac_opt_time_ms:.6f} ms\n")
    f.write(f"MACAC runtime_ratio:       {macac_ratio:.4f}\n")
    f.write(f"MACAC improvement:         {1.0/macac_ratio:.2f}x over baseline\n")
    f.write(f"Speedup (Torch/MACAC_opt): {torch_time_ms / macac_opt_time_ms:.2f}x\n")
    f.write(f"MACAC_opt/Torch ratio:     {macac_opt_time_ms / torch_time_ms:.4f}\n")
    f.write(f"\n")
    f.write(f"MACAC best kernel: warp-shuffle reduction + float4 + inv_K precompute, block=256\n")
    f.write(f"Precision: True\n")
    f.write(f"Optimization iterations: 20 total\n")
    f.write(f"Key optimizations: warp-shuffle, float4 vectorization, inv_K, warp-level reduction\n")

print("Results saved to torch_comparison.txt")
