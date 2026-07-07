# act_quant_kernel: MACAC vs Torch Performance Comparison

## Test Configuration
- **Container**: metax_gemm_opt
- **GPU**: MetaX C500 (MACA 3.3.0.15)
- **Torch**: 2.8.0+metax3.3.0.2 (/opt/conda/bin/python3)
- **MACAC Compiler**: /opt/maca/tools/cu-bridge/bin/cucc
- **Operator**: Activation Quantization Kernel (act_quant_kernel)
- **Shape**: [7, 512] (num_tokens=7, d=512)
- **dtype**: bf16 input → fp8_e4m3fn range clamped
- **Operation**: Per-row abs-max reduction + scale/quantize
- **Warm up**: 20 iterations
- **Test iterations**: 500 (MACAC), 500 (Torch)
- **Date**: 2026-06-29

## Results

| Implementation | Average Time (ms) | Speedup vs MACAC opt | Speedup vs Torch |
|----------------|-------------------|---------------------|-------------------|
| **MACAC optimized** | **0.013246** | 1.00x | **10.54x** |
| MACAC baseline | 0.015612 | 0.85x | 8.94x |
| **PyTorch** | **0.139557** | 0.09x | 1.00x |

## Key Findings

1. **MACAC optimized kernel is 10.5x faster than PyTorch** (0.0132ms vs 0.1396ms)
2. MACAC optimization achieved **18% improvement over baseline** (0.0156ms → 0.0132ms)
3. Precision verified: MACAC output matches reference ✓
4. Runtime ratio (mode 0): 0.539

## Optimization Summary

The baseline kernel used shared memory tree reduction with 512 threads per block.
Key techniques applied in the optimized kernel:

1. **G-loop elimination**: Since group_size == D, the per-group loop is redundant
2. **Warp shuffle reduction**: Replace shared memory tree reduction with 64-lane warp shuffle
3. **Vectorized uint32_t loads**: Read 2 bf16 values per memory transaction
4. **block_size=256**: Eliminate 256 idle threads (256 pairs / 256 threads = all active)
5. **__launch_bounds__(256,2)**: Control register allocation for better occupancy
6. **4-warp cross-warp reduction**: Simple thread-0 sequential read of 4 warp maxes

See [ITERATIONS.md](ITERATIONS.md) for full optimization history with 9 iterations.

## Torch Implementation Issue

PyTorch requires multiple kernel launches (reshape→abs→max→clamp→div→reshape×2), each incurring CPU→GPU dispatch overhead that dominates at this tiny problem size (3584 elements). The single fused MACAC kernel amortizes all overhead into one launch.

## Test Commands

- **MACAC**: `MACA_VISIBLE_DEVICES=0 ./test_maca 20 500 <mode>`
- **Torch**: `/opt/conda/bin/python3 bench_torch.py`

## Raw Measurement Data

- `time_before_opt` (MACAC baseline): 0.015612 ms
- `time_after_opt` (MACAC optimized): 0.013246 ms
- `runtime_ratio` (mode 0): 0.539315
- `precision`: True
- `torch_time`: 0.139557 ms
