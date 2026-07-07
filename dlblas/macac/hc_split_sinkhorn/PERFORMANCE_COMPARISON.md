# hc_split_sinkhorn: MACAC vs Torch Performance Comparison

## Test Configuration
- **Container**: metax_gemm_opt
- **GPU**: MetaX C500 (MACA 3.3.0.15)
- **Torch**: 2.8.0+metax3.3.0.2 (/opt/conda/bin/python3)
- **MACAC Compiler**: /opt/maca/tools/cu-bridge/bin/cucc
- **Operator**: HC Split Sinkhorn
- **Shape**: [B, M, HC, D]
- **dtype**: fp32
- **Operation**: Sinkhorn normalization with head-split pattern
- **Warm up**: 10 iterations
- **Test iterations**: 100 (MACAC), 100 (Torch)
- **Date**: 2026-06-29

## Results

| Implementation | Average Time (ms) | Speedup vs MACAC opt | Speedup vs Torch |
|----------------|-------------------|---------------------|-------------------|
| **MACAC optimized** | **0.040991** | 1.00x | **37.30x** |
| MACAC baseline | 0.151043 | 0.27x | 10.12x |
| **PyTorch** | **1.528823** | 0.03x | 1.00x |

## Key Findings

1. **MACAC optimized kernel is 37.3x faster than PyTorch** (0.040991ms vs 1.528823ms)
2. MACAC optimization achieved **268% improvement over baseline** (0.151043ms → 0.040991ms)
3. Precision verified: MACAC output matches reference ✓
4. Runtime ratio: 0.2714 (72.9% improvement)

## Optimization Summary

See [ITERATIONS.md](ITERATIONS.md) for full optimization history.

## Test Commands

- **MACAC**: `MACA_VISIBLE_DEVICES=0 ./test_maca 10 100 0`
- **Torch**: `/opt/conda/bin/python3 bench_torch.py`

## Raw Measurement Data

- `time_before_opt` (MACAC baseline): 0.151043 ms
- `time_after_opt` (MACAC optimized): 0.040991 ms
- `runtime_ratio`: 0.271385
- `precision`: True
- `torch_time`: 1.528823
