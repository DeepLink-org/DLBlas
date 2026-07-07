# ITERATIONS.md — engram_gate_w_reduce

## Running Info
- Task path: /root/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_w_reduce_run
- Container: metax_gemm_opt
- Start time: 2026-06-26
- Verify command: export MACA_PATH=/opt/maca/ && bash run.sh 10 100 0

## Target Signature
- Operator: engram_gate_w_reduce
- Family: reduction + elementwise fusion
- dtype: fp32 (inputs/outputs), bf16 (weights)
- Shape: grad_w_partial[108, 4, 4096] → reduce dim 0 → [4, 4096] → mul + add
- Layout: C-contiguous
- Key bottleneck: memory bandwidth (7+ MB read per launch)
- Key assumptions: C500 warp=64, block=256 default, B=108 reduce axis

## Reference Files Read
- references/routing.md
- references/hardware/c500.md
- references/verification.md
- references/case_retrieval.md
- references/operator_families/reduction.md
- references/issues/precision_error.md
- references/issues/reduction_bottleneck.md

## Baseline (Round 0)
- Command: bash run.sh 10 100 0
- <time_before_opt>: 0.036265 ms
- <time_after_opt>: 0.034639 ms
- <runtime_ratio>: 0.955174
- <precision>: True

## Iteration Results Summary

### Iter 1: Process 4 outputs per thread + shared memory idea
- runtime_ratio: 4.559 (FAIL - grid too small)
- Decision: REJECT

### Iter 2: __ldg() for float inputs + 512 threads/block
- runtime_ratio: 0.923
- Decision: KEEP (baseline for comparison)

### Iter 3: 128 threads/block + __ldg
- runtime_ratio: 0.950
- Decision: REJECT (worse than Iter 2)

### Iter 4: 2 outputs per thread + __ldg
- runtime_ratio: 2.676 (FAIL - branch in inner loop)
- Decision: REJECT

### Iter 5: 512 threads + unroll by 4 + __ldg
- runtime_ratio: 0.809
- Decision: KEEP

### Iter 6: 256 threads + unroll by 4 + __ldg (BEST)
- runtime_ratio: 0.803
- Decision: BEST - KEEP

### Iter 7: 256 threads + unroll by 8 + __ldg
- runtime_ratio: 0.885
- Decision: REJECT

### Iter 8: __launch_bounds__ + unroll4 + __ldg
- runtime_ratio: 0.854
- Decision: REJECT

### Iter 9: Unroll by 4 without __ldg
- runtime_ratio: 0.848
- Decision: REJECT (__ldg is beneficial)

### Iter 10: Precompute weights + __ldg only on large tensor
- runtime_ratio: 0.850
- Decision: REJECT

### Iter 11: Unroll by 2 + __ldg
- runtime_ratio: 0.899
- Decision: REJECT

### Iter 12: Array indexing + unroll4 + __ldg
- runtime_ratio: 1.033
- Decision: REJECT

### Iter 13: Refined best kernel (Iter 6 pattern)
- runtime_ratio: 0.864 (1000 iters; noise-limited)
- Decision: KEEP as best

### Iter 14: All C per thread
- runtime_ratio: 6.084 (FAIL - grid too small)
- Decision: REJECT

## Final Result
- Best version: Iteration 6/13 (256 threads, unroll by 4, __ldg, pointer arithmetic)
- Final runtime_ratio: 0.80-0.91 (measurement-noise-limited at ~0.03ms kernel time)
- Absolute improvement: ~3-7 microseconds per kernel launch
- Key techniques: loop unrolling (4x), __ldg() for read-only cache, optimal block size (256)
- Rejected variants: multi-output-per-thread (branch overhead), unroll-8 (register pressure), array indexing (compiler inefficiency), all-C-per-thread (low occupancy)
- Remaining risk: measurement noise at 30us kernel time; further optimization limited by memory bandwidth

## Final Rerun
- Commit: 83c3904
- <time_before_opt>: 0.034322 ms
- <time_after_opt>: 0.031352 ms
- <runtime_ratio>: 0.913478
- <precision>: True
