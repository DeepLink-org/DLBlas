# expand_kenel_bwd: MACAC vs Torch Performance Comparison

## Test Configuration
- **Container**: metax_gemm_opt
- **GPU**: MetaX C500 (MACA 3.3.0.15)
- **Torch**: 2.8.0+metax3.3.0.2 (/opt/conda/bin/python3)
- **MACAC Compiler**: /opt/maca/tools/cu-bridge/bin/cucc
- **Operator**: Expand Kernel Backward
- **Shape**: [1, 1024, 1280] → [1, 1024, 4, 1280]
- **dtype**: bf16
- **Operation**: Backward pass of tensor expansion
- **Warm up**: 10 iterations
- **Test iterations**: 100 (MACAC), 100 (Torch)
- **Date**: 2026-06-29

## Results

| Implementation | Average Time (ms) | Speedup vs MACAC opt | Speedup vs Torch |
|----------------|-------------------|---------------------|-------------------|
| **MACAC optimized** | **0.062456** | 1.00x | **5.62x** |
| MACAC baseline | 0.099684 | 0.63x | 3.52x |
| **PyTorch** | **0.350910** | 0.18x | 1.00x |

## Key Findings

1. **MACAC optimized kernel is 5.6x faster than PyTorch** (0.062456ms vs 0.350910ms)
2. MACAC optimization achieved **60% improvement over baseline** (0.099684ms → 0.062456ms)
3. Precision verified: MACAC output matches reference ✓
4. Runtime ratio: 0.6265 (37.3% improvement)

## Optimization Summary

See [ITERATIONS.md](ITERATIONS.md) for full optimization history.

## Test Commands

- **MACAC**: `MACA_VISIBLE_DEVICES=0 ./test_maca 10 100 0`
- **Torch**: `/opt/conda/bin/python3 bench_torch.py`

## Raw Measurement Data

- `time_before_opt` (MACAC baseline): 0.099684 ms
- `time_after_opt` (MACAC optimized): 0.062456 ms
- `runtime_ratio`: 0.626544
- `precision`: True
- `torch_time`: 0.35091
