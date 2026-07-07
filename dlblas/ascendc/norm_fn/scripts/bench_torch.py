#!/usr/bin/env python3
"""PyTorch norm_fn benchmark for AscendC vs torch comparison."""
import time
import torch
import numpy as np
import sys
sys.path.insert(0, '/mnt/data01/zmz/workspace/12agent/waic/origin')

from norm_fn import Model, generate_norm_fn_test_data

N_WARMUP = 10
N_BENCH = 100

# Generate test data
n1, mhc_mult, hidden_size = 13, 4, 1280
residual, fn, normw, out_grad, eps = generate_norm_fn_test_data(n1, mhc_mult, hidden_size, False)

model = Model()
model.eval()

# Warmup
with torch.no_grad():
    for _ in range(N_WARMUP):
        _ = model(residual, fn, None, eps)

# Benchmark
if torch.cuda.is_available():
    torch.cuda.synchronize()

t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(N_BENCH):
        _ = model(residual, fn, None, eps)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t1 = time.perf_counter()

avg_torch_us = (t1 - t0) / N_BENCH * 1e6
print(f"PyTorch avg latency: {avg_torch_us:.2f} us")
print(f"PyTorch total time ({N_BENCH} iters): {(t1-t0)*1000:.4f} ms")
