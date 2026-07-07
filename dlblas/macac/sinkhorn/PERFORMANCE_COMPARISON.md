# sinkhorn: MACAC vs Torch Performance Comparison

## Test Configuration
- **Container**: metax_gemm_opt
- **GPU**: MetaX C500 (MACA 3.3.0.15)
- **Torch**: 2.8.0+metax3.3.0.2 (/opt/conda/bin/python3)
- **MACAC Compiler**: /opt/maca/tools/cu-bridge/bin/cucc
- **Operator**: Sinkhorn
- **Shape**: [B, M, N]
- **dtype**: fp32
- **Operation**: Sinkhorn normalization algorithm
- **Warm up**: 10 iterations
- **Test iterations**: 100 (MACAC), 100 (Torch)
- **Date**: 2026-06-29

## Results

| Implementation | Average Time (ms) | Speedup vs MACAC opt | Speedup vs Torch |
|----------------|-------------------|---------------------|-------------------|
| **MACAC optimized** | **0.039191** | 1.00x | **18.34x** |
| MACAC baseline | 0.164723 | 0.24x | 4.36x |
| **PyTorch** | **0.718897** | 0.05x | 1.00x |

## Key Findings

1. **MACAC optimized kernel is 18.3x faster than PyTorch** (0.039191ms vs 0.718897ms)
2. MACAC optimization achieved **320% improvement over baseline** (0.164723ms → 0.039191ms)
3. Precision verified: MACAC output matches reference ✓
4. Runtime ratio: 0.2379 (76.2% improvement)

## Optimization Summary

See [ITERATIONS.md](ITERATIONS.md) for full optimization history.

Key techniques applied in the best kernel:
## Test Commands

- **MACAC**: `MACA_VISIBLE_DEVICES=0 ./test_maca 10 100 0`
- **Torch**: `/opt/conda/bin/python3 bench_torch.py`

## Raw Measurement Data

- `time_before_opt` (MACAC baseline): 0.164723 ms
- `time_after_opt` (MACAC optimized): 0.039191 ms
- `runtime_ratio`: 0.237921
- `precision`: True
- `torch_time`: 0.718897
