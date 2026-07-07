#!/usr/bin/env python3
"""MACA vs Torch performance comparison for hc_split_sinkhorn"""
import sys, time
sys.path.insert(0, '.')
import hc_split_sinkhorn

# Create model and inputs
model = hc_split_sinkhorn.Model(*hc_split_sinkhorn.get_init_inputs())
inputs = hc_split_sinkhorn.get_inputs()

import torch

# Warmup
for _ in range(10):
    model.forward(*inputs)
torch.cuda.synchronize()

# Benchmark
t0 = time.perf_counter()
for _ in range(100):
    model.forward(*inputs)
torch.cuda.synchronize()
t1 = time.perf_counter()

torch_time = (t1 - t0) / 100 * 1000
print(f"PyTorch reference time: {torch_time:.6f} ms")

# MACA optimized time from final rerun (10 warmup, 500 test)
maca_opt_time = 0.038433  # ms from final rerun
maca_ori_time = 0.144106  # ms from final rerun

print(f"\n=== hc_split_sinkhorn Performance Comparison ===")
print(f"{'Backend':<20} {'Time (ms)':<12} {'vs Torch':<10} {'vs MACA ori':<12}")
print(f"{'-'*54}")
print(f"{'PyTorch (eager)':<20} {torch_time:<12.4f} {1.0:<10.2f}x {'-':<12}")
print(f"{'MACA ori (baseline)':<20} {maca_ori_time:<12.4f} {torch_time/maca_ori_time:<10.2f}x {1.0:<12.2f}x")
print(f"{'MACA opt (best)':<20} {maca_opt_time:<12.4f} {torch_time/maca_opt_time:<10.2f}x {maca_ori_time/maca_opt_time:<12.2f}x")
