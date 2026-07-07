# MTPBlock HC Kernel Optimization — ITERATIONS.md

## Run Information
- Task path: /datapool/zmz/04kernelagent/waic/macac/MTPBlock_run
- Container path: /home/ailab/maca-vendor-workspace/maca_c_opt/workspace/MTPBlock_run
- Container: metax_gemm_opt
- Start time: 2026-06-26
- Validation command: `docker exec -w <WORK_DIR> metax_gemm_opt bash -c 'export MACA_PATH=/opt/maca/ && bash run.sh 10 500 0'`

## Target Signature
- Operator: MTPBlock HC (Hyper-Connection pre/post with Sinkhorn normalization)
- Family: softmax (Sinkhorn = doubly-stochastic normalization)
- Dtype: float32
- Shape: input [1, 8, 4, 512], output [1, 8, 512]
- Key operations: RMS normalization, linear projection, sigmoid, Sinkhorn iterations, weighted sum

## Baseline (Round 0)
- `<time_before_opt>1.210468 ms</time_before_opt>`
- `<time_after_opt>1.209510 ms</time_after_opt>`
- `<runtime_ratio>0.999</runtime_ratio>`
- `<precision>True</precision>`

## Iterations

### Iteration 1: __ldg() 优化只读全局内存
- <time_before_opt>: 1.208 ms
- <time_after_opt>: 1.207 ms
- <runtime_ratio>: 0.999
- <precision>: True
- Result: Negligible change, kept for safety

### Iteration 2: Remove unused comb computation
- <time_before_opt>: 1.215 ms
- <time_after_opt>: 1.169 ms
- <runtime_ratio>: 0.962
- <precision>: True
- Result: 3.8% improvement by removing Sinkhorn iterations from hc_pre

### Iteration 3: Reduce linear projection to only pre-relevant elements
- <time_before_opt>: 1.210 ms
- <time_after_opt>: 1.162 ms
- <runtime_ratio>: 0.960
- <precision>: True
- Result: 4% improvement by computing only HC (4) mix elements instead of MIX_HC (24)

### Iteration 4: Remove __ldg() baseline test
- <time_before_opt>: 1.211 ms
- <time_after_opt>: 1.163 ms
- <runtime_ratio>: 0.961
- <precision>: True
- Result: __ldg() has minimal impact, kept for safety

### Iteration 5: Unroll weighted sum inner loop
- <time_before_opt>: 1.210 ms
- <time_after_opt>: 1.164 ms
- <runtime_ratio>: 0.962
- <precision>: True
- Result: Manual unrolling for HC=4, marginal improvement

### Iteration 6: Precompute 4 weights per inner iteration
- <time_before_opt>: 1.209 ms
- <time_after_opt>: 1.019 ms
- <runtime_ratio>: 0.843
- <precision>: True
- Result: **BEST - 15.6% improvement** by grouping 4 weight computations per loop iteration, reducing sinf/cosf call overhead

### Iteration 7: Fuse mix+pre computation
- <time_before_opt>: 1.207 ms
- <time_after_opt>: 1.159 ms
- <runtime_ratio>: 0.961
- <precision>: True
- Result: Similar to iteration 3

### Iteration 8: Direct pre_smem store
- <precision>: False
- Result: Discarded — precision mismatch from shared memory aliasing

### Iteration 9: Final combination
- <time_before_opt>: 1.210 ms
- <time_after_opt>: 1.163 ms
- <runtime_ratio>: 0.961
- <precision>: True
- Result: Combination of __ldg + unrolled sum + reduced mix

## Final Result
- **Best version**: Iteration 6 (commit 3e2b675)
- **Final rerun**:
  - <time_before_opt>: 1.208037 ms
  - <time_after_opt>: 1.019707 ms
  - <runtime_ratio>: 0.844103
  - <precision>: True
- **Speedup**: 1.18x over baseline
- **Key strategy**: Precompute 4 weight values per inner loop iteration to reduce sinf/cosf overhead in the linear projection

## Torch vs MACA Performance Comparison
- Torch CPU: 144.037 ms
- Torch CUDA (MetaX GPU): 0.158 ms
- MACA C500 best: 1.020 ms
- MACA vs Torch CUDA: 0.16x (MACA is 6.25x slower)
- Note: Torch CUDA uses optimized MXBLAS matrix multiply; MACA kernel computes dot products with per-element weight generation (sinf+cosf), which is fundamentally slower than pre-loaded weight matrix multiplication

## Remaining Risks
- The hc_weight computation (sinf+cosf per element) is the primary bottleneck
- Using pre-loaded constant memory for weights could significantly improve performance
- The per-thread rms_r warp shuffle issue limits safe optimization options

### Iteration 10: Shared memory weight table precomputation
- <time_before_opt>: 1.209037 ms
- <time_after_opt>: 0.098079 ms
- <runtime_ratio>: 0.081121
- <precision>: True
- Result: **BREAKTHROUGH - 12.3x speedup!** Computing HC weight table once in shared memory per block eliminates sinf/cosf overhead entirely. Weight lookup replaces per-element transcendental computation.
- MACA (0.098ms) now faster than torch CUDA (0.158ms) by 1.6x!

### Iteration 10: Shared memory weight table precomputation (BREAKTHROUGH)
- <time_before_opt>: 1.206 ms
- <time_after_opt>: 0.097 ms
- <runtime_ratio>: 0.080
- <precision>: True
- Result: **BREAKTHROUGH - 12.5x speedup!** Computing HC weight table once in shared memory per block eliminates sinf/cosf overhead entirely. Instead of computing sinf+cosf for every (i,j) pair per row, weight table is computed once per block and reused by all threads via shared memory lookup.
- MACA (0.097ms) now 1.63x faster than torch CUDA (0.158ms)

### Iteration 11: 128-thread block size
- <precision>: False
- Result: Discarded — precision mismatch

### Iteration 12: Remove __ldg, use direct load
- <time_after_opt>: 0.097 ms
- <precision>: True
- Result: Same performance as iter10. __ldg kept for safety.

### Iteration 13: Remove final __syncthreads
- <time_after_opt>: 0.097 ms
- <precision>: True
- Result: Same performance. Barrier kept for safety.

## Final Result (Updated)
- **Best version**: Iteration 10 (shared memory weight table)
- **Final rerun (500 iters)**:
  - <time_before_opt>: ~1.206 ms
  - <time_after_opt>: ~0.097 ms
  - <runtime_ratio>: ~0.080
  - <precision>: True
- **Speedup**: 12.5x over baseline, 1.63x over torch CUDA
- **Key strategy**: Precompute HC weight table in shared memory once per block, eliminating per-element sinf/cosf transcendental function calls.

## Torch vs MACA Performance Comparison (Updated)
- **Torch GPU (MetaX C500)**: 0.123 ms (PyTorch 2.8.0+metax3.3.0.2)
- **Torch CPU**: 0.081 ms (Intel Xeon)
- **MACA C500 best**: 0.095 ms (Iteration 10)
- **MACA vs Torch GPU**: MACA is 1.29x faster (0.77x the time)
- **MACA vs Baseline**: 12.74x speedup over original MACA kernel
- **Note**: The breakthrough optimization (shared memory weight table) eliminates the main bottleneck of per-element sinf/cosf computation. With precomputed weights, MACA outperforms even PyTorch GPU on this workload.

## Deliverables
1. Best kernel: `inc/tmp_use.cuh` (commit 272e88e, shared memory weight table)
2. Baseline kernel: `inc/tmp_ori.cuh`
3. Test harness: `src/tmp_test.cu`
4. Performance comparison: `PERFORMANCE_COMPARISON.md`
5. Iteration log: `ITERATIONS.md`
6. Run script: `run.sh`
7. Makefile: `Makefile`

## Torch vs MACA Performance Comparison (Updated)
- **Torch GPU (MetaX C500)**: 0.123 ms (PyTorch 2.8.0+metax3.3.0.2)
- **Torch CPU**: 0.081 ms (Intel Xeon)
- **MACA C500 best**: 0.095 ms (Iteration 10)
- **MACA vs Torch GPU**: MACA is 1.29x faster (0.77x the time)
- **MACA vs Baseline**: 12.74x speedup over original MACA kernel
