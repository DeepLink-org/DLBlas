#!/usr/bin/env python3
"""mhc_post Torch vs MACA Kernel Performance Comparison"""

import torch
import time
import numpy as np

torch.manual_seed(42)

# Problem dimensions
n0 = 2
n1 = 4096
h = 1280
mhc_mult = 4

def mhc_post_torch(x, residual, post_layer_mix, comb_res_mix):
    """Reference torch implementation"""
    term2 = torch.einsum('abmn,abmc->abnc', comb_res_mix, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()

def mhc_post_torch_optimized(x, residual, post_layer_mix, comb_res_mix):
    """Optimized torch: use matmul instead of einsum + explicit fused ops"""
    # comb_res_mix: (n0, n1, 4, 4), residual: (n0, n1, 4, h)
    residual_f = residual.float()
    term2 = torch.matmul(comb_res_mix, residual_f)
    # x: (n0, n1, h) -> (n0, n1, 1, h), post_layer_mix: (n0, n1, 4, 1)
    x_f = x.float().unsqueeze(-2)
    result = (x_f * post_layer_mix + term2).bfloat16()
    return result

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")
    print(f"Torch version: {torch.__version__}")

    # Create test data (same pattern as in our C++ test)
    total_bs = n0 * n1
    total_x = total_bs * h
    total_residual = total_bs * mhc_mult * h
    total_plm = total_bs * mhc_mult
    total_crm = total_bs * mhc_mult * mhc_mult

    print(f"\n--- Problem Dimensions ---")
    print(f"n0={n0}, n1={n1}, h={h}, mhc_mult={mhc_mult}")
    print(f"Total output elements: {total_bs * mhc_mult * h:,}")

    # Generate on GPU
    x = torch.randn((n0, n1, h), dtype=torch.bfloat16, device=device)
    residual = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn((n0, n1, mhc_mult, 1), dtype=torch.float32, device=device)
    comb_res_mix = torch.randn((n0, n1, mhc_mult, mhc_mult), dtype=torch.float32, device=device)

    warmup = 10
    iters = 500

    # Warmup
    for _ in range(warmup):
        _ = mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
    torch.cuda.synchronize()

    # Benchmark einsum version
    start = time.perf_counter()
    for _ in range(iters):
        _ = mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
    torch.cuda.synchronize()
    elapsed_einsum = (time.perf_counter() - start) / iters * 1000  # ms
    print(f"\n--- Torch einsum Results ---")
    print(f"Average time: {elapsed_einsum:.6f} ms")

    # Warmup optimized
    for _ in range(warmup):
        _ = mhc_post_torch_optimized(x, residual, post_layer_mix, comb_res_mix)
    torch.cuda.synchronize()

    # Benchmark matmul version
    start = time.perf_counter()
    for _ in range(iters):
        _ = mhc_post_torch_optimized(x, residual, post_layer_mix, comb_res_mix)
    torch.cuda.synchronize()
    elapsed_matmul = (time.perf_counter() - start) / iters * 1000  # ms
    print(f"Average time (matmul): {elapsed_matmul:.6f} ms")

    # Verify correctness between the two torch implementations
    out_einsum = mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
    out_matmul = mhc_post_torch_optimized(x, residual, post_layer_mix, comb_res_mix)
    max_diff = (out_einsum.float() - out_matmul.float()).abs().max().item()
    print(f"Max diff between einsum and matmul: {max_diff:.6f}")
    print(f"Match: {max_diff < 0.01}")

    # MACA kernel timing (from our test)
    maca_ori_time = 0.167706  # ms (from final rerun)
    maca_opt_time = 0.161676  # ms (from final rerun)

    print(f"\n--- Performance Comparison ---")
    print(f"{'Implementation':<30} {'Time (ms)':<15} {'vs MACA opt':<15}")
    print(f"{'-'*60}")
    print(f"{'MACA original (baseline)':<30} {maca_ori_time:<15.6f} {maca_ori_time/maca_opt_time:<15.2f}x")
    print(f"{'MACA optimized (best)':<30} {maca_opt_time:<15.6f} {1.0:<15.2f}x")
    print(f"{'Torch einsum':<30} {elapsed_einsum:<15.6f} {elapsed_einsum/maca_opt_time:<15.2f}x")
    print(f"{'Torch matmul':<30} {elapsed_matmul:<15.6f} {elapsed_matmul/maca_opt_time:<15.2f}x")

    # Save results
    result = {
        'n0': n0, 'n1': n1, 'h': h, 'mhc_mult': mhc_mult,
        'torch_einsum_ms': elapsed_einsum,
        'torch_matmul_ms': elapsed_matmul,
        'maca_ori_ms': maca_ori_time,
        'maca_opt_ms': maca_opt_time,
        'maca_speedup': maca_ori_time / maca_opt_time,
        'torch_einsum_vs_maca_opt': elapsed_einsum / maca_opt_time,
        'torch_matmul_vs_maca_opt': elapsed_matmul / maca_opt_time,
    }

    import json
    with open('/home/ailab/opt_test/mhc_post_run/torch_comparison.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to torch_comparison.json")

if __name__ == '__main__':
    main()
