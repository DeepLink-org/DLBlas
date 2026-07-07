#!/usr/bin/env python3
"""PyTorch norm_fn benchmark on NPU for AscendC vs torch comparison."""
import time
import torch
import numpy as np
import sys
sys.path.insert(0, '/mnt/data01/zmz/workspace/12agent/waic/origin')

from norm_fn import generate_norm_fn_test_data

N_WARMUP = 10
N_BENCH = 100

# Generate test data and move to NPU
n1, mhc_mult, hidden_size = 13, 4, 1280
residual, fn, normw, out_grad, eps = generate_norm_fn_test_data(n1, mhc_mult, hidden_size, False)

residual = residual.to('npu')
fn = fn.to('npu')

# Model implementation directly on NPU tensors
def norm_fn_torch(residual, mhc_fn, mhc_norm_weight, mhc_norm_eps):
    """Direct PyTorch implementation for NPU benchmark."""
    if mhc_norm_weight is not None:
        mhc_fn = mhc_fn * mhc_norm_weight
    residual = residual.flatten(2, 3).float()
    mhc_mult = mhc_fn.shape[0]
    rms_group_size = mhc_fn.shape[-1]
    mixes = torch.einsum(
        'mbk,nbk->mbn',
        residual.view(-1, 1, rms_group_size),
        mhc_fn.view(mhc_mult, 1, rms_group_size),
    )
    sqrsum = residual.view(-1, 1, rms_group_size).square().sum(-1)
    mixes = (mixes * (sqrsum.unsqueeze(-1) / rms_group_size + mhc_norm_eps).rsqrt()).sum(-2)
    return mixes.view(*residual.shape[:2], -1)

# Also test with the Model class
sys.path.insert(0, '/mnt/data01/zmz/workspace/12agent/waic/origin')
from norm_fn import Model
model = Model()
model.eval()

# Method 1: Direct function
torch.npu.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(N_WARMUP):
        _ = norm_fn_torch(residual, fn, None, eps)
        torch.npu.synchronize()
t1 = time.perf_counter()
warmup_time = t1 - t0

torch.npu.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(N_BENCH):
        _ = norm_fn_torch(residual, fn, None, eps)
        torch.npu.synchronize()
t1 = time.perf_counter()

avg_torch_us_direct = (t1 - t0) / N_BENCH * 1e6
print(f"=== NPU Direct Function ===")
print(f"Warmup ({N_WARMUP} iters): {warmup_time*1000:.2f} ms")
print(f"Bench ({N_BENCH} iters): {(t1-t0)*1000:.4f} ms")
print(f"PyTorch NPU avg latency: {avg_torch_us_direct:.2f} us")

# Method 2: Model class
torch.npu.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(N_WARMUP):
        _ = model(residual, fn, None, eps)
        torch.npu.synchronize()
t1 = time.perf_counter()
warmup_time2 = t1 - t0

torch.npu.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(N_BENCH):
        _ = model(residual, fn, None, eps)
        torch.npu.synchronize()
t1 = time.perf_counter()

avg_torch_us_model = (t1 - t0) / N_BENCH * 1e6
print(f"\n=== NPU Model Class ===")
print(f"Warmup ({N_WARMUP} iters): {warmup_time2*1000:.2f} ms")
print(f"Bench ({N_BENCH} iters): {(t1-t0)*1000:.4f} ms")
print(f"PyTorch NPU avg latency (Model): {avg_torch_us_model:.2f} us")
