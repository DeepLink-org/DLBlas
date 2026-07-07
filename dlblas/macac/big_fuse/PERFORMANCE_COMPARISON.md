# big_fuse: MACAC vs Torch Performance Comparison

## Test Configuration
- **Container**: metax_gemm_opt
- **GPU**: MetaX C500 (MACA 3.3.0.15)
- **Torch**: 2.8.0+metax3.3.0.2 (/opt/conda/bin/python3)
- **MACAC Compiler**: /opt/maca/tools/cu-bridge/bin/cucc
- **Operator**: Big Fuse (MHC Pre-Norm + Fused Operations)
- **Shape**: residual [1, 512, 4, 1280], fn [24, 5120]
- **dtype**: bf16
- **Operation**: RMS norm + matmul fusion
- **Warm up**: 10 iterations
- **Test iterations**: 100 (MACAC), 100 (Torch)
- **Date**: 2026-06-29

## Results

| Implementation | Average Time (ms) | Speedup vs MACAC opt | Speedup vs Torch |
|----------------|-------------------|---------------------|-------------------|
| **MACAC optimized** | **0.466834** | 1.00x | **2.17x** |
| MACAC baseline | 0.464502 | 1.01x | 2.18x |
| **PyTorch** | **1.014200** | 0.46x | 1.00x |

## Key Findings

1. **MACAC optimized kernel is 2.2x faster than PyTorch** (0.466834ms vs 1.014200ms)
2. MACAC optimized similar to baseline (0.464502ms vs 0.466834ms)
3. Precision verified: MACAC output matches reference ✓
4. Runtime ratio: 1.0050

## Optimization Summary

See [ITERATIONS.md](ITERATIONS.md) for full optimization history.

Key techniques applied in the best kernel:
## Test Commands

- **MACAC**: `MACA_VISIBLE_DEVICES=0 ./test_maca 10 100 0`
- **Torch**: `/opt/conda/bin/python3 torch_compare.py`

## Raw Measurement Data

- `time_before_opt` (MACAC baseline): 0.464502 ms
- `time_after_opt` (MACAC optimized): 0.466834 ms
- `runtime_ratio`: 1.005021
- `precision`: True
- `torch_time`: 1.0142
