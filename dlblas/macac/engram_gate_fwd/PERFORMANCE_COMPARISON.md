# engram_gate_fwd: MACAC vs Torch Performance Comparison

## Test Configuration
- **Container**: metax_gemm_opt
- **GPU**: MetaX C500 (MACA 3.3.0.15)
- **Torch**: 2.8.0+metax3.3.0.2 (/opt/conda/bin/python3)
- **MACAC Compiler**: /opt/maca/tools/cu-bridge/bin/cucc
- **Operator**: Engram Gate Forward
- **Shape**: num_tokens=4096, max_ngram_size=3
- **dtype**: fp32
- **Operation**: Forward engram gating with ngram embedding lookup
- **Warm up**: 10 iterations
- **Test iterations**: 100 (MACAC), 100 (Torch)
- **Date**: 2026-06-29

## Results

| Implementation | Average Time (ms) | Speedup vs MACAC opt | Speedup vs Torch |
|----------------|-------------------|---------------------|-------------------|
| **MACAC optimized** | **0.487742** | 1.00x | **28.05x** |
| MACAC baseline | 0.764763 | 0.64x | 17.89x |
| **PyTorch** | **13.681100** | 0.04x | 1.00x |

## Key Findings

1. **MACAC optimized kernel is 28.0x faster than PyTorch** (0.487742ms vs 13.681100ms)
2. MACAC optimization achieved **57% improvement over baseline** (0.764763ms → 0.487742ms)
3. Precision verified: MACAC output matches reference ✓
4. Runtime ratio: 0.6378 (36.2% improvement)

## Optimization Summary

See [ITERATIONS.md](ITERATIONS.md) for full optimization history.

Key techniques applied in the best kernel:
## Test Commands

- **MACAC**: `MACA_VISIBLE_DEVICES=0 ./test_maca 10 100 0`
- **Torch**: `/opt/conda/bin/python3 bench_torch.py`

## Raw Measurement Data

- `time_before_opt` (MACAC baseline): 0.764763 ms
- `time_after_opt` (MACAC optimized): 0.487742 ms
- `runtime_ratio`: 0.637768
- `precision`: True
- `torch_time`: 13.6811
