# engram_hash: MACAC vs Torch Performance Comparison

## Test Configuration
- **Container**: metax_gemm_opt
- **GPU**: MetaX C500 (MACA 3.3.0.15)
- **Torch**: 2.8.0+metax3.3.0.2 (/opt/conda/bin/python3)
- **MACAC Compiler**: /opt/maca/tools/cu-bridge/bin/cucc
- **Operator**: Engram Hash
- **Shape**: [2, 4096, 16]
- **dtype**: fp32
- **Operation**: Hash-based embedding lookup
- **Warm up**: 10 iterations
- **Test iterations**: 100 (MACAC), 100 (Torch)
- **Date**: 2026-06-29

## Results

| Implementation | Average Time (ms) | Speedup vs MACAC opt | Speedup vs Torch |
|----------------|-------------------|---------------------|-------------------|
| **MACAC optimized** | **0.045268** | 1.00x | **22.73x** |
| MACAC baseline | 0.067011 | 0.68x | 15.36x |
| **PyTorch** | **1.029027** | 0.04x | 1.00x |

## Key Findings

1. **MACAC optimized kernel is 22.7x faster than PyTorch** (0.045268ms vs 1.029027ms)
2. MACAC optimization achieved **48% improvement over baseline** (0.067011ms → 0.045268ms)
3. Precision verified: MACAC output matches reference ✓
4. Runtime ratio: 0.6755 (32.4% improvement)

## Optimization Summary

See [ITERATIONS.md](ITERATIONS.md) for full optimization history.

Key techniques applied in the best kernel:
## Test Commands

- **MACAC**: `MACA_VISIBLE_DEVICES=0 ./test_maca 10 100 0`
- **Torch**: `/opt/conda/bin/python3 torch_compare.py`

## Raw Measurement Data

- `time_before_opt` (MACAC baseline): 0.067011 ms
- `time_after_opt` (MACAC optimized): 0.045268 ms
- `runtime_ratio`: 0.675543
- `precision`: True
- `torch_time`: 1.029027
