# mhc_post MACA Kernel Performance Report

## Operator Info
- **Name**: mhc_post (multi-head convolution post-processing)
- **Formula**: `output = bf16(x_f32 * post_layer_mix + einsum('abmn,abmc->abnc', comb_res_mix, residual_f32))`
- **Family**: matmul + elementwise fusion (batched small matmul + broadcast mul + add + cast)
- **dtype**: bf16 inputs (x, residual, output), fp32 intermediates (post_layer_mix, comb_res_mix)
- **Shape**: n0=2, n1=4096, h=1280, mhc_mult=4
- **Output elements**: 41,943,040 (42M)

## Optimization Summary
- **Strategy**: __ldg() read-only cache + 2D grid launch + crm/plm register hoisting + increased grid coverage
- **Iterations**: 9 candidate rounds
- **Best version**: Iteration 9 (2D grid + hoist crm/plm + __ldg())

## Performance Results (500 iterations)

| Metric | Value |
|--------|-------|
| MACA baseline (ori) | 0.167706 ms |
| MACA optimized (best) | 0.161676 ms |
| MACA speedup | 1.037x (3.7%) |
| Torch einsum | 3.852051 ms |
| Torch vs MACA opt | 23.83x slower |

## Correctness
- `<precision>True</precision>` — output matches baseline within tolerance (atol=0.01)
- Memory bandwidth: ~1,232 GB/s achieved (66.86% of peak HBM bandwidth)

## Bottleneck Analysis (from trace-report)
- **Classification**: Memory-bound
- **L2C hit rate**: 0.58% (working set ~180MB >> L2 cache)
- **HBM usage**: 66.86%
- **Effective occupancy**: 8.00%
- **Primary bottleneck**: HBM bandwidth; kernel is fundamentally memory-bound with 42M output elements

## Key Optimizations Applied
1. `__ldg()` for all read-only inputs (x, residual, post_layer_mix, comb_res_mix)
2. 2D grid launch (256 x 32) for improved SM coverage
3. crm/plm hoisted outside h-loop to reduce redundant loads
4. Compact arithmetic with fused multiply-add chains

## Comparison Conclusion
The MACA custom kernel achieves **23.8x speedup** over torch einsum by:
- Single fused kernel vs multi-kernel launch
- No intermediate tensor allocations
- Direct bf16 memory access with optimized memory pattern
- Compiler-level FMA fusion

The optimization is fundamentally bounded by HBM bandwidth (66.86% utilized).
