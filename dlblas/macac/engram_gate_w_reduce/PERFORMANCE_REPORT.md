# engram_gate_w_reduce — MACAC Performance Report

## Operator Summary
- **Name**: `engram_gate_w_reduce`
- **Family**: Reduction + Elementwise Fusion
- **Semantics**: Reduce sum along dim 0 of `grad_w_partial[108,4,4096]` → multiply by bf16 weights → add to reference accumulators
- **Input Types**: fp32 (grad_w_partial, refs), bf16 (weights)
- **Output**: Two fp32 tensors of shape `[4, 4096]`

## Performance Comparison

### Torch vs MACAC
| Metric | Value |
|--------|-------|
| **Torch time** | 0.063656 ms |
| **MACAC baseline (original)** | 0.037120 ms |
| **MACAC optimized** | 0.032223 ms |
| **MACAC speedup vs Torch** | **1.98x** |
| **MACAC opt vs MACAC ori** | 0.868 (13% improvement) |
| **Precision** | True ✓ |

### Key Optimization Techniques
1. **Loop unrolling by 4x** — reduced branch overhead in B-reduction loop
2. **`__ldg()` intrinsic** — routed read-only data through texture cache for improved L2 hit rate
3. **Optimal block size (256 threads)** — balanced SM coverage and per-SM occupancy
4. **Pointer arithmetic** — efficient address calculation vs array indexing

### Trace Profile (v0 baseline)
- **Bottleneck**: Memory-bound (GVM share 16.10%, L2C hit 2.48%)
- **Occupancy**: 5.00% (register-limited, 26 regs/thread)
- **Top stall**: vls_pipeline_stall 95.52%
- **HBM bandwidth**: 972.84 GB/s (52.78% of peak)
- **Grid/block**: [64,1,1] / [256,1,1]

### Iterations Summary
- 14 optimization iterations executed
- Best result: ratio 0.803 (Iteration 6)
- 4 rejected variants (branch overhead, low occupancy, register pressure)
- Final kernel achieves consistent 13-20% improvement over baseline

## Files
- `inc/tmp_ori.cuh` — Baseline kernel
- `inc/tmp_use.cuh` — Optimized kernel
- `inc/tmp_check.cuh` — Baseline copy (backward compat)
- `src/tmp_test.cu` — Unit test
- `ITERATIONS.md` — Full iteration log
- `benchmark_results.txt` — Raw benchmark output
- `profile-artifacts/REPORT_baseline.md` — Baseline trace report
