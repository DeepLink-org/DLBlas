#!/usr/bin/env python3
"""Torch vs MACAC performance comparison for indexer operator."""
import torch
import time
import numpy as np

# Match the MACAC kernel shapes
B, S, H, D, T_total, T_used, TopK = 2, 64, 16, 64, 256, 16, 16
start_pos = 0

# Use bfloat16 like the MACAC kernel
dtype = torch.bfloat16
device = torch.device('cuda:0')

# Create inputs (same pattern as tmp_test.cu)
q = torch.zeros(B, S, H, D, dtype=dtype, device=device)
kv_cache = torch.zeros(B, T_total, D, dtype=dtype, device=device)
weights = torch.zeros(B, S, H, dtype=dtype, device=device)

# Initialize with same pattern as C++ test
q_flat = torch.arange(B * S * H * D, dtype=torch.float32).fmod(127).div(127.0).to(dtype)
q = q_flat.view(B, S, H, D).to(device)

kv_flat = torch.arange(B * T_total * D, dtype=torch.float32).fmod(131).div(131.0).to(dtype)
kv_cache = kv_flat.view(B, T_total, D).to(device)

w_flat = torch.arange(B * S * H, dtype=torch.float32).mul(3).fmod(100).div(100.0).add(0.5).to(dtype)
weights = w_flat.view(B, S, H).to(device)

# Torch reference implementation
def torch_indexer(q, kv_cache, weights, T_used, TopK, start_pos):
    # einsum: "bshd,btd->bsht"
    kv = kv_cache[:, :T_used, :]
    index_score = torch.einsum("bshd,btd->bsht", q, kv)

    # ReLU + weight * softmax_scale
    softmax_scale = D ** -0.5
    index_score = (index_score.relu() * weights.unsqueeze(-1) * (softmax_scale * H ** -0.5)).sum(dim=2)

    # Causal mask
    if start_pos == 0:
        mask = torch.arange(T_used, device=device).repeat(S, 1) >= torch.arange(1, S + 1, device=device).unsqueeze(1) // 4
        index_score += torch.where(mask, float("-inf"), 0)

    # TopK
    _, topk_idxs = index_score.topk(min(TopK, T_used), dim=-1)
    return topk_idxs


# Warmup
print("Warming up torch...")
for _ in range(10):
    _ = torch_indexer(q, kv_cache, weights, T_used, TopK, start_pos)
torch.cuda.synchronize()

# Benchmark torch
print("Running torch benchmark (500 iterations)...")
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(500):
    result = torch_indexer(q, kv_cache, weights, T_used, TopK, start_pos)
torch.cuda.synchronize()
end = time.perf_counter()
torch_time_ms = (end - start) / 500 * 1000
print(f"Torch average time: {torch_time_ms:.6f} ms")
print(f"Torch output shape: {result.shape}, dtype: {result.dtype}")
print(f"Sample torch output: {result[0, :3, :5]}")

# MACAC kernel timing (from run.sh with exec_mode=2 - opt only)
# Read from the test output above: 0.043486 ms for opt kernel
macac_time_ms = 0.043486

print(f"\n{'='*60}")
print(f"PERFORMANCE COMPARISON")
print(f"{'='*60}")
print(f"MACAC kernel (optimized):  {macac_time_ms:.6f} ms")
print(f"Torch reference:           {torch_time_ms:.6f} ms")
print(f"Speedup (MACAC vs Torch):  {torch_time_ms/macac_time_ms:.2f}x")
print(f"Runtime ratio (MACAC/Torch): {macac_time_ms/torch_time_ms:.4f}")
print(f"{'='*60}")
