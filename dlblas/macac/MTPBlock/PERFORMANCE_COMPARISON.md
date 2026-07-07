# MTPBlock HC Kernel — MACA vs Torch Performance Comparison

## Test Configuration
- Operator: MTPBlock HC (Hyper-Connection pre computation)
- Input: [1, 8, 4, 512] float32
- Container: metax_gemm_opt
- Date: 2026-06-29

## Results

| Backend | Time (ms) | Speedup vs Baseline | Notes |
|---------|-----------|---------------------|-------|
| MACA C500 Baseline (tmp_ori) | 1.210 | 1.00x | Original with per-element sinf/cosf |
| **MACA C500 Best (iter10)** | **0.095** | **12.74x** | Shared memory weight table precomputation |
| Torch GPU (MetaX C500) | 0.123 | — | PyTorch 2.8.0+metax3.3.0.2 |
| Torch CPU | 0.081 | — | Intel Xeon |

## Key Finding
- **MACA C500 is 1.29x faster than Torch GPU** on this workload
- The breakthrough optimization was precomputing the HC weight table in shared memory
  once per block, eliminating per-element sinf/cosf transcendental function calls
- Original MACA baseline was 7.7x slower than Torch GPU; after optimization, MACA is
  1.29x faster

## Optimization Strategy
- Iteration 10: Precompute HC weight table in shared memory (12.5x speedup)
- The weight table stores HC*HC_D = 4*2048 = 8192 float values computed once per block
- All threads then access the table via shared memory instead of calling sinf/cosf
- Shared memory usage: ~10KB per block, well within 64KB limit

## Files
- Best kernel: inc/tmp_use.cuh (commit 272e88e)
- Baseline: inc/tmp_ori.cuh
- Test harness: src/tmp_test.cu
- Full log: ITERATIONS.md
