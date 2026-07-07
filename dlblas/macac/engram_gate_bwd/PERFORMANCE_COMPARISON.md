# engram_gate_bwd: MACAC vs Torch Performance Comparison

## Test Configuration
- **Container**: metax_gemm_opt
- **GPU**: MetaX C500 (MACA 3.3.0.15)
- **Torch**: 2.8.0+metax3.3.0.2 (/opt/conda/bin/python3)
- **MACAC Compiler**: /opt/maca/tools/cu-bridge/bin/cucc
- **Operator**: Engram Gate Backward
- **Shape**: [2, 4096, 16]
- **dtype**: fp32
- **Operation**: Gradient computation for engram gating
- **Warm up**: 10 iterations
- **Test iterations**: 100 (MACAC), 100 (Torch)
- **Date**: 2026-06-29

## Results

| Implementation | Average Time (ms) | Speedup vs MACAC opt | Speedup vs Torch |
|----------------|-------------------|---------------------|-------------------|
| **MACAC optimized** | **0.029542** | 1.00x | **29.23x** |
| MACAC baseline | 0.040241 | 0.73x | 21.46x |
| **PyTorch** | **0.863382** | 0.03x | 1.00x |

## Key Findings

1. **MACAC optimized kernel is 29.2x faster than PyTorch** (0.029542ms vs 0.863382ms)
2. MACAC optimization achieved **36% improvement over baseline** (0.040241ms → 0.029542ms)
3. Precision verified: MACAC output matches reference ✓
4. Runtime ratio: 0.7341 (26.6% improvement)

## Optimization Summary

See [ITERATIONS.md](ITERATIONS.md) for full optimization history.

Key techniques applied in the best kernel:
## Test Commands

- **MACAC**: `MACA_VISIBLE_DEVICES=0 ./test_maca 10 100 0`
- **Torch**: `/opt/conda/bin/python3 bench_torch.py`

## Raw Measurement Data

- `time_before_opt` (MACAC baseline): 0.040241 ms
- `time_after_opt` (MACAC optimized): 0.029542 ms
- `runtime_ratio`: 0.734143
- `precision`: True
- `torch_time`: 0.863382
