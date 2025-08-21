import torch
import triton
import json
from pathlib import Path
from typing import Tuple
import random

from dlblas.kernels.grouped_gemm.BF16 import k_grouped_gemm
from dlblas.kernels.grouped_gemm.BF16.utils import generate_random_list, row_max_normalization
from dlblas.utils.device_utils import infer_device


def gmm_dw(a, b, batch_sizes):
    """Reference implementation of grouped matrix multiplication"""
    K, M = a.shape
    K_, N = b.shape

    assert a.stride(-1) == 1, "Please make sure A is K-major"
    assert b.stride(-1) == 1, "Please make sure B is K-major"
    assert K == K_, "Please make sure that A and B have the same seqlen"
    num_groups = batch_sizes.shape[0]

    out = a.new_empty(num_groups, M, N)

    group_end = batch_sizes.cumsum(0) - batch_sizes + batch_sizes
    group_start = batch_sizes.cumsum(0) - batch_sizes
    for g, (start, end) in enumerate(zip(group_start, group_end)):
        rhs = b[start:end, :]
        lhs = a[start:end, :]
        out[g] = lhs.T @ rhs
    return out.contiguous()


def grouped_gemm_ref(a, b, batch_sizes):
    """Reference implementation wrapper"""
    return gmm_dw(a, b, batch_sizes.cpu())


def grouped_gemm_kernel(a, b, batch_sizes):
    """Kernel implementation wrapper"""
    return k_grouped_gemm(a, b, batch_sizes)


# Benchmark configurations
matrix_configs = [
    (768*2, 2048),
    (2048, 768), 
    (1536*2, 4096),
    (4096, 1536)
]

groups_range = [64, 128, 256]
configs = [(m, n, groups) for m, n in matrix_configs for groups in groups_range]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['m', 'n', 'groups'],
        x_vals=configs,
        line_arg='provider',
        line_vals=['torch', 'kernel'],
        line_names=['Torch Reference (ms)', 'Fused Kernel (ms)'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='grouped-gemm-performance',
        args={},
    ))
def benchmark(m, n, groups, provider):
    dtype = torch.bfloat16
    device = infer_device()
    
    # Generate random batch sizes
    batch_sizes = torch.Tensor(generate_random_list(groups, groups*5120)).to(device).to(torch.int64).abs()
    K = batch_sizes.sum().item()
    
    # Create input matrices
    a = torch.randn(K, m, dtype=dtype, device=device)
    b = torch.randn(K, n, dtype=dtype, device=device)
    
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: grouped_gemm_ref(a.clone(), b.clone(), batch_sizes.clone()),
            quantiles=quantiles,
        )
    elif provider == 'kernel':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: grouped_gemm_kernel(a.clone(), b.clone(), batch_sizes.clone()),
            quantiles=quantiles,
        )

    return ms, max_ms, min_ms


def validate_correctness():
    """Validate that kernel produces correct results"""
    print("Validating correctness...")
    
    device = infer_device()
    groups = 128
    batch_sizes = torch.Tensor(generate_random_list(groups, groups*5120)).to(device).to(torch.int64).abs()
    K = batch_sizes.sum().item()
    
    for m, n in matrix_configs:
        torch.cuda.empty_cache()
        
        a = torch.randn(K, m, dtype=torch.bfloat16, device=device)
        b = torch.randn(K, n, dtype=torch.bfloat16, device=device)
        
        # Run reference implementation
        out_ref = grouped_gemm_ref(a, b, batch_sizes)
        out_ref = row_max_normalization(out_ref)
        
        # Run kernel implementation
        out_kernel = grouped_gemm_kernel(a, b, batch_sizes)
        out_kernel = row_max_normalization(out_kernel)
        
        # Validate correctness
        try:
            torch.testing.assert_close(out_kernel, out_ref, rtol=0.01, atol=0.01)
            print(f"✓ Validation passed for m={m}, n={n}")
        except AssertionError:
            print(f"✗ Validation failed for m={m}, n={n}")
            return False
    
    print("All validations passed!")
    return True


if __name__ == '__main__':
    # First validate correctness
    if validate_correctness():
        # Then run benchmark
        benchmark.run(print_data=True)
    else:
        print("Benchmark aborted due to validation failures")
