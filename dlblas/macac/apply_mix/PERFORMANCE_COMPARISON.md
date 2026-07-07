# apply_mix: MACAC vs Torch Performance Comparison

## Test Configuration
- **Container**: metax_gemm_opt
- **GPU**: MetaX C500 (MACA 3.3.0.15)
- **Torch**: 2.8.0+metax3.3.0.2 (/opt/conda/bin/python3)
- **MACAC Compiler**: /opt/maca/tools/cu-bridge/bin/cucc
- **Operator**: Apply Mix
- **Shape**: [2, 1024, 4, 1280], mix=[2, 1024, 4, 1]
- **dtype**: bf16 input, fp32 mix
- **Operation**: Element-wise multiply + broadcast
- **Warm up**: 10 iterations
- **Test iterations**: 100 (MACAC), 100 (Torch)
- **Date**: 2026-06-29

## Results

| Implementation | Average Time (ms) | Speedup vs MACAC opt | Speedup vs Torch |
|----------------|-------------------|---------------------|-------------------|
| **MACAC optimized** | **0.042842** | 1.00x | **13.39x** |
| MACAC baseline | 0.064765 | 0.66x | 8.86x |
| **PyTorch** | **0.573822** | 0.07x | 1.00x |

## Key Findings

1. **MACAC optimized kernel is 13.4x faster than PyTorch** (0.042842ms vs 0.573822ms)
2. MACAC optimization achieved **51% improvement over baseline** (0.064765ms → 0.042842ms)
3. Precision verified: MACAC output matches reference ✓
4. Runtime ratio: 0.6615 (33.9% improvement)

## Optimization Summary

See [ITERATIONS.md](ITERATIONS.md) for full optimization history.

Key techniques applied in the best kernel:
## Test Commands

- **MACAC**: `MACA_VISIBLE_DEVICES=0 ./test_maca 10 100 0`
- **Torch**: `/opt/conda/bin/python3 bench_torch.py`

## Raw Measurement Data

- `time_before_opt` (MACAC baseline): 0.064765 ms
- `time_after_opt` (MACAC optimized): 0.042842 ms
- `runtime_ratio`: 0.661488
- `precision`: True
- `torch_time`: 0.573822
