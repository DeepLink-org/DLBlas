# sparse_attn: MACAC vs Torch Performance Comparison

## Test Configuration
- **Container**: metax_gemm_opt
- **GPU**: MetaX C500 (MACA 3.3.0.15)
- **Torch**: 2.8.0+metax3.3.0.2 (/opt/conda/bin/python3)
- **MACAC Compiler**: /opt/maca/tools/cu-bridge/bin/cucc
- **Operator**: Sparse Attention
- **Shape**: B=2, M=16, N=32, H=8, D=64, TopK=16
- **dtype**: bf16
- **Operation**: Sparse attention with top-k selection
- **Warm up**: 10 iterations
- **Test iterations**: 100 (MACAC), 100 (Torch)
- **Date**: 2026-06-29

## Results

| Implementation | Average Time (ms) | Speedup vs MACAC opt | Speedup vs Torch |
|----------------|-------------------|---------------------|-------------------|
| **MACAC optimized** | **0.048714** | 1.00x | **9.02x** |
| MACAC baseline | 0.085645 | 0.57x | 5.13x |
| **PyTorch** | **0.439599** | 0.11x | 1.00x |

## Key Findings

1. **MACAC optimized kernel is 9.0x faster than PyTorch** (0.048714ms vs 0.439599ms)
2. MACAC optimization achieved **76% improvement over baseline** (0.085645ms → 0.048714ms)
3. Precision verified: MACAC output matches reference ✓
4. Runtime ratio: 0.5688 (43.1% improvement)

## Optimization Summary

See [ITERATIONS.md](ITERATIONS.md) for full optimization history.

Key techniques applied in the best kernel:
## Test Commands

- **MACAC**: `MACA_VISIBLE_DEVICES=0 ./test_maca 10 100 0`
- **Torch**: `/opt/conda/bin/python3 bench_sparse_attn.py`

## Raw Measurement Data

- `time_before_opt` (MACAC baseline): 0.085645 ms
- `time_after_opt` (MACAC optimized): 0.048714 ms
- `runtime_ratio`: 0.568794
- `precision`: True
- `torch_time`: 0.439599
