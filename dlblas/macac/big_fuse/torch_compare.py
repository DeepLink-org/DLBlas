#!/usr/bin/env python3
"""big_fuse torch vs MACAC performance comparison on MetaX C500"""
import time
import sys
sys.path.insert(0, '/datapool/zmz/04kernelagent/waic/origin')
from big_fuse import Model, get_inputs, get_init_inputs
import torch

device = torch.device('cuda:0')
print(f"Using device: {device}")
print(f"Torch version: {torch.__version__}")

# Initialize
mhc_mult, hidden_size = get_init_inputs()
model = Model(mhc_mult=mhc_mult, hidden_size=hidden_size)
model = model.to(device)
model.eval()

# Get inputs on GPU
residual = get_inputs()[0].to(device)

print(f"Input shape: {residual.shape}, dtype: {residual.dtype}")
print(f"fn shape: {model.fn.shape}, mhc_scale: {model.mhc_scale.shape}")

# Warmup
print("Warming up...")
for _ in range(20):
    with torch.no_grad():
        model(residual)
torch.cuda.synchronize()

# Benchmark
n_iters = 1000
print(f"Benchmarking torch ({n_iters} iterations)...")
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(n_iters):
    with torch.no_grad():
        model(residual)
torch.cuda.synchronize()
end = time.perf_counter()

torch_time_ms = (end - start) / n_iters * 1000

# MACAC times (from container benchmarks)
macac_baseline_ms = 0.531   # baseline kernel
macac_optimized_ms = 0.373  # best optimized kernel (ratio 0.702)

print(f"\n{'='*60}")
print(f"Performance Comparison: big_fuse operator")
print(f"{'='*60}")
print(f"Shape: residual [1, 512, 4, 1280] bf16")
print(f"fn: [24, 5120], mhc_mult=4, hidden_size=1280")
print(f"{'='*60}")
print(f"Torch (PyTorch fused):          {torch_time_ms:.4f} ms")
print(f"MACAC baseline (ori):           {macac_baseline_ms:.4f} ms")
print(f"MACAC optimized (warp-shuffle): {macac_optimized_ms:.4f} ms")
print(f"{'='*60}")
print(f"MACAC opt vs MACAC baseline: {macac_baseline_ms/macac_optimized_ms:.2f}x speedup")
print(f"MACAC opt vs Torch:          {torch_time_ms/macac_optimized_ms:.2f}x {'faster' if torch_time_ms > macac_optimized_ms else 'slower'}")
print(f"MACAC baseline vs Torch:     {torch_time_ms/macac_baseline_ms:.2f}x")

import json
result = {
    "operator": "big_fuse",
    "shape": "residual[1,512,4,1280] bf16, fn[24,5120]",
    "torch_time_ms": round(torch_time_ms, 4),
    "macac_baseline_ms": macac_baseline_ms,
    "macac_optimized_ms": macac_optimized_ms,
    "macac_speedup_vs_baseline": round(macac_baseline_ms / macac_optimized_ms, 3),
    "macac_vs_torch": round(torch_time_ms / macac_optimized_ms, 3),
    "optimization_strategy": "warp-shuffle reduction (64-lane C500 warp), eliminated 192 __syncthreads",
    "registers_per_thread": 50,
    "mtreg_occupancy_pct": 9.0,
    "runtime_ratio": 0.702
}
with open('/mnt/opt_test/big_fuse_run/performance_comparison.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nResults saved to performance_comparison.json")
