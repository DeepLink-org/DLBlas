# pre_split_mixes: MACAC vs Torch Performance Comparison

## Test Configuration
- **Container**: metax_gemm_opt
- **GPU**: MetaX C500 (MACA 3.3.0.15)
- **Torch**: 2.8.0+metax3.3.0.2 (/opt/conda/bin/python3)
- **MACAC Compiler**: /opt/maca/tools/cu-bridge/bin/cucc
- **Operator**: Pre-Split Mixes
- **Shape**: B=1, N=1024, M=4, M3=24
- **dtype**: fp32
- **Operation**: Pre-split mixing computation
- **Warm up**: 10 iterations
- **Test iterations**: 100 (MACAC), 100 (Torch)
- **Date**: 2026-06-29

## Results

| Implementation | Average Time (ms) | Speedup vs MACAC opt | Speedup vs Torch |
|----------------|-------------------|---------------------|-------------------|
| **MACAC optimized** | **0.026578** | 1.00x | **5.95x** |
| MACAC baseline | 0.042314 | 0.63x | 3.74x |
| **PyTorch** | **0.158116** | 0.17x | 1.00x |

## Key Findings

1. **MACAC optimized kernel is 5.9x faster than PyTorch** (0.026578ms vs 0.158116ms)
2. MACAC optimization achieved **59% improvement over baseline** (0.042314ms → 0.026578ms)
3. Precision verified: MACAC output matches reference ✓
4. Runtime ratio: 0.6281 (37.2% improvement)

## Optimization Summary

See [ITERATIONS.md](ITERATIONS.md) for full optimization history.

Key techniques applied in the best kernel:
## Test Commands

- **MACAC**: `MACA_VISIBLE_DEVICES=0 ./test_maca 10 100 0`
- **Torch**: `/opt/conda/bin/python3 bench_torch_fixed.py`

## Raw Measurement Data

- `time_before_opt` (MACAC baseline): 0.042314 ms
- `time_after_opt` (MACAC optimized): 0.026578 ms
- `runtime_ratio`: 0.628108
- `precision`: True
- `torch_time`: 0.158116
