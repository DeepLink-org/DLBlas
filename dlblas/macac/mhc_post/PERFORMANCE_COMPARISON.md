# mhc_post: MACAC vs Torch Performance Comparison

## Test Configuration
- **Container**: metax_gemm_opt
- **GPU**: MetaX C500 (MACA 3.3.0.15)
- **Torch**: 2.8.0+metax3.3.0.2 (/opt/conda/bin/python3)
- **MACAC Compiler**: /opt/maca/tools/cu-bridge/bin/cucc
- **Operator**: MHC Post
- **Shape**: [1, 512, 4, 1280]
- **dtype**: bf16
- **Operation**: Post-processing after multi-head computation
- **Warm up**: 10 iterations
- **Test iterations**: 100 (MACAC), 100 (Torch)
- **Date**: 2026-06-29

## Results

| Implementation | Average Time (ms) | Speedup vs MACAC opt | Speedup vs Torch |
|----------------|-------------------|---------------------|-------------------|
| **MACAC optimized** | **0.036586** | 1.00x | **105.35x** |
| MACAC baseline | 0.036137 | 1.01x | 106.66x |
| **PyTorch** | **3.854369** | 0.01x | 1.00x |

## Key Findings

1. **MACAC optimized kernel is 105.4x faster than PyTorch** (0.036586ms vs 3.854369ms)
2. MACAC optimization regressed by **1%** vs baseline (0.036137ms → 0.036586ms)
3. Precision verified: MACAC output matches reference ✓
4. Runtime ratio: 1.0124

## Optimization Summary

See [ITERATIONS.md](ITERATIONS.md) for full optimization history.

Key techniques applied in the best kernel:
## Test Commands

- **MACAC**: `MACA_VISIBLE_DEVICES=0 ./test_maca 10 100 0`
- **Torch**: `/opt/conda/bin/python3 torch_compare.py`

## Raw Measurement Data

- `time_before_opt` (MACAC baseline): 0.036137 ms
- `time_after_opt` (MACAC optimized): 0.036586 ms
- `runtime_ratio`: 1.012423
- `precision`: True
- `torch_time`: 3.854369
