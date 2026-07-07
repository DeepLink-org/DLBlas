# apply_mix Optimization Log

## Run Info
- Task path: /home/ailab/maca-vendor-workspace/maca_c_opt/workspace/apply_mix_run
- Container: metax_gemm_opt
- Start time: 2026-06-26
- End time: 2026-06-26
- Verification command: export MACA_PATH=/opt/maca/ && bash run.sh 10 100 0

## Target Signature
- Operator: apply_mix (weighted sum over mhc dimension)
- Family: reduction + elementwise (compound)
- dtype: bf16 input (x, mix is float), bf16 output
- Shape: n0=2, n1=1024, mhc=4, h=1280
- Layout: contiguous
- Primary bottleneck: vls_pipeline_stall (93.18%), memory latency
- Key assumptions: C500 warp=64, mhc=4 is small constant

## Reference Files Read
- routing.md, hardware/c500.md, verification.md
- operator_families/reduction.md
- issues/reduction_bottleneck.md
- case_retrieval.md
- Trace-report: profile-artifacts/apply_mix_v0_baseline/REPORT_baseline.md

## Baseline (Round 0)
- time_before_opt: 0.064188 ms
- time_after_opt: 0.043008 ms
- runtime_ratio: 0.670028
- precision: True
- Trace: vls_pipeline_stall=93.18%, MTE duty=41.75%, occupancy=5.00%

## Iteration Summary

| Iter | Strategy | Ratio | Precision | Kept |
|------|----------|-------|-----------|------|
| 1 | block=512 + ldg | 0.669 | True | No |
| 2 | uint32 vectorized loads | 0.665 | True | Yes |
| 3 | uint32 + block=512 | 0.689 | True | No |
| 4 | uint2 loads (4 bf16) | 0.671 | True | No |
| 5 | launch_bounds(256,8) | 0.661 | True | Yes |
| 6 | ldg + launch_bounds | 0.662 | True | No |
| 7 | pragma unroll | ERROR | N/A | No |
| 8 | launch_bounds(256) only | 0.663 | True | No |
| 9 | block=128 | 0.672 | True | No |
| 10 | shared memory tiling | 1.096 | True | No |
| 11 | manual loop unrolling | 0.650 | True | Yes |
| 12 | unrolled + ldg | 0.655 | True | No |
| 13 | unrolled no launch_bounds | 0.659 | True | No |
| 14 | unrolled + separate phases | 0.656 | True | No |
| 15 | unrolled + block=512 | 0.669 | True | No |
| 16 | unrolled simplified | 0.663 | True | No |
| 17 | unrolled + const locals | 0.661 | True | No |
| 18 | unrolled + fmaf() | 0.656 | True | No |

## Final Result
- Best version: Iter 11 - manually unrolled loop + launch_bounds(256,8)
- Best commit: 6e104f8
- Final rerun:
  - time_before_opt: 0.064005 ms
  - time_after_opt: 0.042245 ms
  - runtime_ratio: 0.660027
  - precision: True
- MACA speedup vs baseline: 1.52x
- Rejected: ldg, block=128/512, uint2, shared memory, fmaf, pragma unroll

## Torch Comparison
- Torch time: 0.578394 ms
- MACA opt time: 0.042245 ms
- MACA vs Torch speedup: 13.7x

## Key Optimization Strategies
1. Vectorized uint32_t loads (2 bf16 per load) - reduced load instructions
2. launch_bounds(256,8) - optimized register allocation
3. Manual loop unrolling - eliminated loop overhead for short loop
4. Pre-loaded mix values in registers
5. Fused bf16 output packing (2 bf16 per uint32 write)
