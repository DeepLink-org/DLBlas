# head_compute_mix_bwd: MACAC vs Torch Performance Comparison

## Test Configuration
- **Container**: metax_gemm_opt
- **GPU**: MetaX C500 (MACA 3.3.0.15)
- **Torch**: 2.8.0+metax3.3.0.2 (/opt/conda/bin/python3)
- **MACAC Compiler**: /opt/maca/tools/cu-bridge/bin/cucc
- **Operator**: Head Compute Mix Backward
- **Shape**: [16, 16384, 4]
- **dtype**: fp32
- **Operation**: Backward head mixing computation
- **Warm up**: 10 iterations
- **Test iterations**: 100 (MACAC), 100 (Torch)
- **Date**: 2026-06-29

## Results

| Implementation | Average Time (ms) | Speedup vs MACAC opt | Speedup vs Torch |
|----------------|-------------------|---------------------|-------------------|
| **MACAC optimized** | **0.055060** | 1.00x | **1.82x** |
| MACAC baseline | 0.068856 | 0.80x | 1.46x |
| **PyTorch** | **0.100449** | 0.55x | 1.00x |

## Key Findings

1. **MACAC optimized kernel is 1.8x faster than PyTorch** (0.055060ms vs 0.100449ms)
2. MACAC optimization achieved **25% improvement over baseline** (0.068856ms → 0.055060ms)
3. Precision verified: MACAC output matches reference ✓
4. Runtime ratio: 0.7996 (20.0% improvement)

## Optimization Summary

See [ITERATIONS.md](ITERATIONS.md) for full optimization history.

Key techniques applied in the best kernel:
## Test Commands

- **MACAC**: `MACA_VISIBLE_DEVICES=0 ./test_maca 10 100 0`
- **Torch**: `/opt/conda/bin/python3 bench_torch.py`

## Raw Measurement Data

- `time_before_opt` (MACAC baseline): 0.068856 ms
- `time_after_opt` (MACAC optimized): 0.055060 ms
- `runtime_ratio`: 0.799643
- `precision`: True
- `torch_time`: 0.100449
