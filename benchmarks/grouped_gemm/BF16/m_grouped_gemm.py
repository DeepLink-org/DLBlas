import torch
import triton
import json
from pathlib import Path
from typing import Tuple
import random

from dlblas.kernels.grouped_gemm.BF16 import m_grouped_gemm
from dlblas.kernels.grouped_gemm.BF16.utils import generate_random_list, row_max_normalization
from dlblas.utils.device_utils import infer_device


def gmm_ref(a, b, batch_sizes, trans_b=False):
    """Reference implementation of grouped matrix multiplication"""
    batch_sizes = batch_sizes.cpu().numpy()

    out = []
    start = 0
    for i, size in enumerate(batch_sizes):
        rhs = b[i, :, :].t() if trans_b else b[i, :, :]
        out.append(a[start:start + size, :] @ rhs)
        start += size
    return torch.cat(out)


def grouped_gemm_ref(a, b, batch_sizes, trans_b=False):
    """Reference implementation wrapper"""
    return gmm_ref(a, b, batch_sizes, trans_b)


def grouped_gemm_kernel(a, b, batch_sizes, trans_b=False):
    """Kernel implementation wrapper"""
    return m_grouped_gemm(a, b, batch_sizes, trans_b)


# Benchmark configurations
matrix_configs = [
    (768*2, 2048),
    (2048, 768), 
    (1536*2, 4096),
    (4096, 1536)
]

groups_range = [64, 128, 256]
trans_b_options = [False, True]
configs = [(n, k, groups, trans_b) for n, k in matrix_configs 
          for groups in groups_range for trans_b in trans_b_options]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['n', 'k', 'groups', 'trans_b'],
        x_vals=configs,
        line_arg='provider',
        line_vals=['torch', 'kernel'],
        line_names=['Torch Reference (ms)', 'Fused Kernel (ms)'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='m-grouped-gemm-performance',
        args={},
    ))
def benchmark(n, k, groups, trans_b, provider):
    dtype = torch.bfloat16
    device = infer_device()
    
    # Generate random batch sizes
    batch_sizes = torch.Tensor(generate_random_list(groups, groups*5120)).to(device).to(torch.int64)
    M = batch_sizes.sum().item()
    
    # Create input matrices
    a = torch.randn(M, k, dtype=dtype, device=device)
    if trans_b:
        b = torch.randn(groups, n, k, dtype=dtype, device=device)
    else:
        b = torch.randn(groups, k, n, dtype=dtype, device=device)
    
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: grouped_gemm_ref(a.clone(), b.clone(), batch_sizes.clone(), trans_b),
            quantiles=quantiles,
        )
    elif provider == 'kernel':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: grouped_gemm_kernel(a.clone(), b.clone(), batch_sizes.clone(), trans_b),
            quantiles=quantiles,
        )

    return ms, max_ms, min_ms


def validate_correctness():
    """Validate that kernel produces correct results"""
    print("Validating correctness...")
    
    device = infer_device()
    groups = 128
    
    for trans_b in [False, True]:
        print(f"Validating trans_b={trans_b}...")
        
        batch_sizes = torch.Tensor(generate_random_list(groups, groups*5120)).to(device).to(torch.int64)
        M = batch_sizes.sum().item()
        
        for n, k in matrix_configs:
            torch.cuda.empty_cache()
            
            a = torch.randn(M, k, dtype=torch.bfloat16, device=device)
            if trans_b:
                b = torch.randn(groups, n, k, dtype=torch.bfloat16, device=device)
            else:
                b = torch.randn(groups, k, n, dtype=torch.bfloat16, device=device)
            
            # Run reference implementation
            out_ref = grouped_gemm_ref(a, b, batch_sizes, trans_b)
            out_ref = row_max_normalization(out_ref)
            
            # Run kernel implementation
            out_kernel = grouped_gemm_kernel(a, b, batch_sizes, trans_b)
            out_kernel = row_max_normalization(out_kernel)
            
            # Validate correctness
            try:
                torch.testing.assert_close(out_kernel, out_ref, rtol=0.01, atol=0.01)
                print(f"✓ Validation passed for n={n}, k={k}, trans_b={trans_b}")
            except AssertionError:
                print(f"✗ Validation failed for n={n}, k={k}, trans_b={trans_b}")
                return False
    
    print("All validations passed!")
    return True


def calculate_tflops(n, k, M, time_ms):
    """Calculate TFLOPs performance"""
    if time_ms > 0:
        return (2 * M * n * k) / (time_ms * 1e-3) / 1e12  # Convert to TFLOPs
    return 0


def measure_single_performance():
    """Measure performance for a single configuration"""
    device = infer_device()
    groups = 128
    trans_b = False
    n, k = matrix_configs[0]  # First configuration
    
    batch_sizes = torch.Tensor(generate_random_list(groups, groups*5120)).to(device).to(torch.int64)
    M = batch_sizes.sum().item()
    
    # Create input matrices
    a = torch.randn(M, k, dtype=torch.bfloat16, device=device)
    if trans_b:
        b = torch.randn(groups, n, k, dtype=torch.bfloat16, device=device)
    else:
        b = torch.randn(groups, k, n, dtype=torch.bfloat16, device=device)
    
    # Warmup
    for _ in range(3):
        _ = grouped_gemm_kernel(a, b, batch_sizes, trans_b)
    
    # Benchmark kernel
    ms = triton.testing.do_bench(
        lambda: grouped_gemm_kernel(a, b, batch_sizes, trans_b),
        quantiles=[0.5],
    )
    
    tflops = calculate_tflops(n, k, M, ms)
    return ms, tflops


if __name__ == '__main__':
    # First validate correctness
    if validate_correctness():
        # Then run benchmark
        benchmark.run(print_data=True)
        
        # Additional performance analysis
        print("\nPerformance Summary:")
        print("=" * 80)
        print(f"{'Config':<25} {'Time (us)':<12} {'TFLOPs':<10}")
        print("-" * 80)
        
        # Measure performance for a sample configuration
        time_us, tflops = measure_single_performance()
        n, k = matrix_configs[0]
        print(f"{f'n={n},k={k},g=128,trans_b=False':<25} {time_us:.2f}{'':<8} {tflops:.1f}{'':<6}")
        
    else:
        print("Benchmark aborted due to validation failures")
