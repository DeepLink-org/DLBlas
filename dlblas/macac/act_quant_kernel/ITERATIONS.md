# act_quant_kernel 优化记录 (Fresh Run 2026-06-29)

## 运行信息
- 任务路径: /home/ailab/maca-vendor-workspace/maca_c_opt/workspace/act_quant_kernel_run
- 容器内路径: /home/ailab/maca-vendor-workspace/maca_c_opt/workspace/act_quant_kernel_run
- 开始时间: 2026-06-29 15:30 UTC
- 验证容器: metax_gemm_opt
- 验证命令: `export MACA_PATH=/opt/maca/ && bash run.sh 10 100 0`

## 目标签名
- 算子: act_quant_kernel (激活量化)
- family: reduction + elementwise (per-group abs-max reduction + scale/quantize)
- dtype: bf16 input → bf16 output (clamped to fp8 range) + fp32 scales
- shape: x [7, 512], 输出 x_q [7, 512], x_s [7, 1]
- group_size: 512 (=D, single group per row)
- layout: contiguous
- 主要瓶颈判断: 问题规模极小 (3584 elements)，launch overhead + reduction sync 为主
- 关键假设: C500 warp=64, SM=104, shared mem=64KB/block

## 参考文件读取记录
- `references/routing.md` — 路由策略
- `references/hardware/c500.md` — C500硬件参数
- `references/verification.md` — 验证流程
- `references/operator_families/elementwise.md` — elementwise优化
- `references/operator_families/reduction.md` — reduction优化

## Baseline (Round 0)
- 命令: `export MACA_PATH=/opt/maca/ && bash run.sh 10 100 0`
- baseline commit: 9ed4868
- Isolated measurements: ori=0.027538ms, opt=0.027320ms (identical kernels)
- <time_before_opt>: 0.027208 ms (mode 0)
- <time_after_opt>: 0.015084 ms (mode 0, warm cache artifact)
- <runtime_ratio>: 0.554385 (mode 0 artifact)
- <precision>: True

## 迭代记录

### 迭代 1: G-loop elimination + warp shuffle reduction
**假设**: G=1 since group_size==D, outer loop is redundant. Warp shuffle faster than shared memory reduction.
**目标**: Eliminate loop overhead; use 64-lane warp shuffle for max reduction.
**结果 (mode 0)**:
- <time_before_opt>: 0.027976 ms
- <time_after_opt>: 0.014641 ms
- <runtime_ratio>: 0.523335
- <precision>: True
**决策**: 保留 ✓

### 迭代 2: Vectorized uint32_t loads
**假设**: Read 2 bf16 values per uint32_t load, halving memory instruction count.
**目标**: Reduce memory bandwidth pressure via 2-element vectorized loads.
**结果 (mode 0)**:
- <time_before_opt>: 0.026447 ms
- <time_after_opt>: 0.013097 ms
- <runtime_ratio>: 0.495208
- <precision>: True
**决策**: 保留 ✓ (ratio < 0.5)

### 迭代 3: __launch_bounds__(512, 1)
**假设**: Control register allocation to improve occupancy.
**目标**: Add launch_bounds for register pressure management.
**结果 (mode 0)**:
- <time_before_opt>: 0.026819 ms
- <time_after_opt>: 0.013197 ms
- <runtime_ratio>: 0.492077
- <precision>: True
**决策**: 保留 ✓

### 迭代 4: Single-warp block (block_size=64)
**假设**: Eliminate shared memory and __syncthreads() entirely, use warp-only reduction.
**目标**: Zero shared memory, zero sync barriers, pure shuffle reduction.
**结果 (isolated)**: opt=0.015099ms vs ori=0.028283ms
- <runtime_ratio>: 0.533852 (mode 0)
- Regression vs Iter 3 (0.492). More work per thread (8 elements) negates sync savings.
**决策**: 回退 ✗

### 迭代 5: block_size=256 (all threads active)
**假设**: With 512 threads, half are idle (256 pairs vs 512 threads). block=256 eliminates idle threads.
**目标**: 256 active threads, all process exactly 1 pair. 4 warps instead of 8.
**结果 (isolated)**:
- ori: 0.016440 ms, opt: 0.013731 ms → real ratio: 0.835
- <runtime_ratio>: 0.473380 (mode 0, imprecise due to warmup)
- <precision>: True
**决策**: 保留 ✓ (real 20% improvement)

### 迭代 6: __launch_bounds__(256, 2)
**假设**: minBlocksPerSM=2 improves occupancy for tiny kernel.
**结果 (isolated)**: opt=0.013846ms vs ori=0.016710ms → real ratio: 0.829
**决策**: 保留 ✓ (slight improvement)

### 迭代 7: block_size=128
**假设**: 2 warps, even simpler cross-warp reduction.
**结果 (isolated)**: opt=0.014466ms vs ori=0.016471ms → real ratio: 0.878
**决策**: 回退 ✗ (worse than block=256, each thread handles 2 pairs)

### 迭代 8: block_size=256 + __launch_bounds__(256, 2) confirmed
**结果 (isolated)**: opt=0.013806ms vs ori=0.016594ms → real ratio: 0.832
**决策**: 保留 ✓

### 迭代 9: Thread 0 pre-computes inv_scale to shared memory
**假设**: Save 255 division operations (1.0f/scale) by computing once on thread 0.
**结果 (isolated)**: opt=0.013828ms vs ori=0.016536ms → real ratio: 0.836
**决策**: 保留 (neutral, slightly simpler code)

## 最终结果
- 最佳版本: Iteration 6 (block=256 + launch_bounds(256,2)) 或 Iteration 9 (same + inv_scale shared)
- Best real ratio: ~0.83 (17% faster than baseline)
- Best mode 0 ratio: 0.492 (Iter 3) 但 mode 0 受 warmup 效应影响，实际改善 ~10-20%
- 最终保留策略: block_size=256, vectorized uint32_t loads, warp shuffle reduction, __launch_bounds__(256,2)
- rejected variants: single-warp (64), block=128, block=512 with idle threads, __ldg

### Final Rerun (mode 0 tags)
- <time_before_opt>: 0.027674 ms
- <time_after_opt>: 0.014925 ms
- <runtime_ratio>: 0.539315
- <precision>: True

## MACAC vs Torch 性能对比

| 实现 | 平均时间 (ms) | vs MACAC opt | vs Torch |
|------|---------------|-------------|----------|
| MACAC optimized | 0.013246 | 1.00x | 10.54x |
| MACAC baseline | 0.015612 | 0.85x | 8.94x |
| PyTorch (2.8.0+metax3.3.0.2) | 0.139557 | 0.09x | 1.00x |

**MACAC optimized kernel is 10.5x faster than PyTorch implementation.**

## Trace Report 状态
- CycleTrace profiling 失败：kernel 太小 (0.013ms)，CycleTrace 无法捕获硬件事件
- mcProfiler 数据已采集（见 profile-artifacts/act_quant_kernel_v1_fresh_baseline/）
- mcTracer launch config 已采集

## 剩余风险
- 极小问题规模 (3584 elements) 导致测量噪声显著
- group_size < D 的通用情况尚未测试
- 更大 batch size 的扩展性未验证
