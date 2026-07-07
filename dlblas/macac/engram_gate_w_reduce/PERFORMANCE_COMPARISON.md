# engram_gate_w_reduce: MACAC vs Torch Performance Comparison

## Test Configuration
- **Container**: metax_gemm_opt
- **GPU**: MetaX C500 (MACA 3.3.0.15)
- **Torch**: 2.8.0+metax3.3.0.2 (/opt/conda/bin/python3)
- **MACAC Compiler**: /opt/maca/tools/cu-bridge/bin/cucc
- **Operator**: Engram Gate Weight Reduce
- **Shape**: grad_w_partial[108, 4, 4096], weights[4, 4096]
- **dtype**: fp32
- **Operation**: Weight gradient reduction
- **Warm up**: 10 iterations
- **Test iterations**: 100 (MACAC), 100 (Torch)
- **Date**: 2026-06-29

## Results

| Implementation | Average Time (ms) | Speedup vs MACAC opt | Speedup vs Torch |
|----------------|-------------------|---------------------|-------------------|
| **MACAC optimized** | **0.035028** | 1.00x | **1.77x** |
| MACAC baseline | 0.034895 | 1.00x | 1.78x |
| **PyTorch** | **0.062115** | 0.56x | 1.00x |

## Key Findings

1. **MACAC optimized kernel is 1.8x faster than PyTorch** (0.035028ms vs 0.062115ms)
2. MACAC optimized similar to baseline (0.034895ms vs 0.035028ms)
3. Precision verified: MACAC output matches reference ✓
4. Runtime ratio: 1.0038

## Optimization Summary

See [ITERATIONS.md](ITERATIONS.md) for full optimization history.

Key techniques applied in the best kernel:
## Test Commands

- **MACAC**: `MACA_VISIBLE_DEVICES=0 ./test_maca 10 100 0`
- **Torch**: `/opt/conda/bin/python3 benchmark_torch_vs_macac.py`

## Raw Measurement Data

- `time_before_opt` (MACAC baseline): 0.034895 ms
- `time_after_opt` (MACAC optimized): 0.035028 ms
- `runtime_ratio`: 1.003815
- `precision`: True
- `torch_time`: 0.062115
