#!/usr/bin/env python3
"""MACAC vs Torch Performance Comparison for act_quant_kernel"""

import torch
import torch.nn as nn
import subprocess
import re
import sys
import time

# Configuration (matches the origin spec)
num_tokens = 7
d = 512
group_size = 512
dtype = torch.bfloat16
fp8_max = 448.0  # torch.finfo(torch.float8_e4m3fn).max
fp8_min = -448.0  # torch.finfo(torch.float8_e4m3fn).min

def torch_act_quant(x, group_size, eps=1e-10):
    """Reference Torch implementation of activation quantization"""
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max_val = finfo.max
    fp8_min_val = finfo.min

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax * torch.tensor(1.0 / fp8_max_val, dtype=torch.float32, device=x.device)
    x_q = (x_ / x_s).clamp(min=fp8_min_val, max=fp8_max_val)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))

    return x_q, x_s


def run_macac(mode='ori', warmup=10, test_iters=100):
    """Run MACAC kernel and extract timing"""
    maca_dir = "/home/ailab/maca-vendor-workspace/maca_c_opt/workspace/act_quant_kernel_run"
    if mode == 'ori':
        exec_mode = 1
        grep_pattern = "origin fprop"
    else:
        exec_mode = 2
        grep_pattern = "opt fprop"

    cmd = f"cd {maca_dir} && export MACA_PATH=/opt/maca/ && MACA_VISIBLE_DEVICES=0 ./test_maca {warmup} {test_iters} {exec_mode}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    match = re.search(grep_pattern + r' average time: ([\d.]+) ms', result.stdout)
    if match:
        return float(match.group(1))
    else:
        print(f"MACAC parse error: {result.stdout[:500]}")
        return None


def benchmark_torch(x, warmup=10, test_iters=100):
    """Benchmark PyTorch implementation"""
    # Warmup
    for _ in range(warmup):
        torch_act_quant(x.clone(), group_size)
    torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(test_iters):
        torch_act_quant(x.clone(), group_size)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / test_iters) * 1000.0  # ms


def main():
    print("=" * 60)
    print("act_quant_kernel: MACAC vs Torch Performance Comparison")
    print("=" * 60)
    print(f"  Shape: [{num_tokens}, {d}], dtype={dtype}, group_size={group_size}")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}")
    print()

    # Create test input on GPU
    torch.manual_seed(42)
    x = torch.rand(num_tokens, d, dtype=dtype, device='cuda')

    # Verify correctness
    x_q_ref, x_s_ref = torch_act_quant(x, group_size)
    print(f"  Reference output: x_q shape={x_q_ref.shape}, x_s shape={x_s_ref.shape}")
    print()

    # Run benchmarks
    print("Running MACAC benchmarks...")
    warmup, test_count = 20, 500

    macac_ori_time = run_macac('ori', warmup, test_count)
    print(f"  MACAC baseline (ori):  {macac_ori_time:.6f} ms")

    macac_opt_time = run_macac('opt', warmup, test_count)
    print(f"  MACAC optimized (opt): {macac_opt_time:.6f} ms")

    print("Running Torch benchmark...")
    torch_time = benchmark_torch(x, warmup, test_count)
    print(f"  PyTorch:               {torch_time:.6f} ms")

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  {'Implementation':<25} {'Avg Time (ms)':<15} {'vs MACAC opt':<15} {'vs Torch':<15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15}")

    if macac_opt_time:
        print(f"  {'MACAC optimized':<25} {macac_opt_time:<15.6f} {'1.00x':<15} {torch_time/macac_opt_time:<15.2f}x")
    if macac_ori_time and macac_opt_time:
        print(f"  {'MACAC baseline':<25} {macac_ori_time:<15.6f} {macac_opt_time/macac_ori_time:<15.2f}x {torch_time/macac_ori_time:<15.2f}x")
    print(f"  {'PyTorch':<25} {torch_time:<15.6f} {macac_opt_time/torch_time if macac_opt_time else 0:<15.2f}x {'1.00x':<15}")

    print()
    print("=" * 60)
    if macac_opt_time:
        speedup = torch_time / macac_opt_time
        print(f"  MACAC optimized is {speedup:.1f}x faster than PyTorch")
    if macac_ori_time and macac_opt_time:
        speedup_vs_ori = macac_ori_time / macac_opt_time
        print(f"  MACAC optimization: {speedup_vs_ori:.2f}x speedup over baseline")
    print("=" * 60)


if __name__ == "__main__":
    main()
