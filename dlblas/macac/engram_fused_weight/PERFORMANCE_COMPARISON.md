# engram_fused_weight: MACAC vs Torch Performance Comparison

## Test Configuration
- **Container**: metax_gemm_opt
- **GPU**: MetaX C500 (MACA 3.3.0.15)
- **Torch**: 2.8.0+metax3.3.0.2 (/opt/conda/bin/python3)
- **MACAC Compiler**: /opt/maca/tools/cu-bridge/bin/cucc
- **Operator**: Engram Fused Weight
- **Shape**: [4, 128]
- **dtype**: bf16 input → fp32 output
- **Operation**: wh.float() * we.float()
- **Warm up**: 10 iterations
- **Test iterations**: 100 (MACAC), 100 (Torch)
- **Date**: 2026-06-29

## Results

| Implementation | Average Time (ms) | Speedup vs MACAC opt | Speedup vs Torch |
|----------------|-------------------|---------------------|-------------------|
| **MACAC optimized** | **0.023406** | 1.00x | **1.46x** |
| MACAC baseline | 0.033359 | 0.70x | 1.03x |
| **PyTorch** | **0.034247** | 0.68x | 1.00x |

## Key Findings

1. **MACAC optimized kernel is 1.5x faster than PyTorch** (0.023406ms vs 0.034247ms)
2. MACAC optimization achieved **43% improvement over baseline** (0.033359ms → 0.023406ms)
3. Precision verified: MACAC output matches reference ✓
4. Runtime ratio: 0.7016 (29.8% improvement)

## Optimization Summary

See [ITERATIONS.md](ITERATIONS.md) for full optimization history.

Key techniques applied in the best kernel:
## Test Commands

- **MACAC**: `MACA_VISIBLE_DEVICES=0 ./test_maca 10 100 0`
- **Torch**: `/opt/conda/bin/python3 bench_torch.py`

## Raw Measurement Data

- `time_before_opt` (MACAC baseline): 0.033359 ms
- `time_after_opt` (MACAC optimized): 0.023406 ms
- `runtime_ratio`: 0.701635
- `precision`: True
- `torch_time`: 0.034247
