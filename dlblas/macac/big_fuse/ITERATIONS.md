# big_fuse 优化迭代记录

## 运行信息
- 任务路径: /mnt/opt_test/big_fuse_run
- 容器: metax_gemm_opt
- 开始时间: 2026-06-26
- 验证命令: export MACA_PATH=/opt/maca/ && bash run.sh 10 100 0

## 目标签名
- 算子: big_fuse (MHC pre-processing fused kernel)
- Family: Fused (layernorm + elementwise + softmax + reduction)
- dtype: bf16 input/intermediate, f32 accumulation
- Shape: residual [1, 512, 4, 1280], fn [24, 5120], mhc_scale [3], mhc_base [24]
- Output: post_mix [512, 4], comb_mix [512, 16], layer_input [512, 1280] bf16
- 主要瓶颈判断: Memory-bound (大量fn权重读取, 512×480KB), matmul部分的24次reduction串行化
- 关键假设: warp=64, SM=104, block=256, shared_mem=1KB per reduction

## 参考文件读取记录
- references/routing.md: 路由策略理解
- references/hardware/c500.md: C500硬件特性 (warp=64, SM=104, HBM 1.55TB/s)
- references/verification.md: 验证流程和标签规范
- references/case_retrieval.md: 案例检索策略
- references/operator_families/elementwise.md: elementwise优化策略
- references/operator_families/reduction.md: reduction优化策略 (warp-level, shuffle)
- references/operator_families/softmax.md: softmax优化策略
- references/operator_families/layernorm.md: layernorm优化策略

## Baseline (Round 0)
- 命令: export MACA_PATH=/opt/maca/ && bash run.sh 10 100 0
- <time_before_opt>: 0.530867 ms
- <time_after_opt>: 0.532303 ms
- <runtime_ratio>: 1.002705
- <precision>: True

### 迭代 1: packed 24-way reduction in shared memory (24KB smem)
**假设**: 将24次独立reduction合并为一次packed reduction，减少sync开销
**结果**: regression, runtime_ratio=1.119
**决策**: 回退 — 24KB shared memory限制了occupancy

### 迭代 2: warp-shuffle reduction (64-lane C500 warp)
**假设**: 使用warp shuffle消除shared memory reduction，C500 warp=64
**结果**: runtime_ratio=0.702, +29.8% speedup!
**决策**: 保留 ✓ — 最佳版本

### 迭代 3: blockDim=512
**假设**: 更多线程per block提高occupancy
**结果**: regression, runtime_ratio=0.858
**决策**: 回退 — 8 warps增加cross-warp reduction开销

### 迭代 4: 去除Stage 5前的barrier
**假设**: Stage 5仅使用register数据，不需要sync
**结果**: runtime_ratio=0.702, 无变化
**决策**: 保留 — 不影响性能但减少不必要sync

### 迭代 5: loop-interchange + __ldg
**假设**: fn权重连续访问改善cache locality
**结果**: runtime_ratio=0.701, 无变化
**决策**: 保留 — L2C hit rate已达96.77%

### 迭代 6: shared-memory tiled matmul
**假设**: 缓存fn tile在shared memory减少global reads
**结果**: MAJOR regression, runtime_ratio=1.564
**决策**: 回退 — tile overhead远超benefit (thin matmul只有24输出)

### 迭代 7: residual cache + __expf
**假设**: 缓存residual在shared memory + 快速exp
**结果**: regression, runtime_ratio=0.922
**决策**: 回退 — 20KB smem限制occupancy

### 迭代 8: blockDim=128
**假设**: 减少registers pressure提高occupancy
**结果**: regression, runtime_ratio=1.021
**决策**: 回退

### 迭代 9: blockDim=64 (single warp)
**假设**: 单warp消除cross-warp reduction
**结果**: MAJOR regression, runtime_ratio=1.835
**决策**: 回退 — per-thread workload 4x

### 迭代 10: 移除#pragma unroll
**假设**: 减少编译器register pressure
**结果**: runtime_ratio=0.702, 无变化
**决策**: 保留

### 迭代 11: __launch_bounds__(256,4)
**假设**: 控制register allocation
**结果**: runtime_ratio=0.702, 无变化 (MACA忽略minBlocks参数)
**决策**: 保留

### 迭代 12: float4 fn loads
**假设**: 向量化load减少指令数
**结果**: runtime_ratio=0.702, 无变化
**决策**: 保留 — 编译器已自动向量化

### 迭代 13-18: 多种微优化方向
**假设**: 组合输出写入、稳定性测试等
**结果**: 全部收敛于~0.702 ratio
**决策**: 最佳ratio=0.702, 性能稳定

## 最终结果
- 最终保留版本: warp-shuffle reduction (迭代 2)
- 最佳runtime_ratio: 0.702 (29.8% speedup vs baseline)
- 主要优化: C500 64-lane warp shuffle消除192个__syncthreads
- Registers: 108 → 50 (↓54%)
- ARRIVE/sync events: 7.12% → 4.25% (↓40%)
- IPC: 53 → 87 (↑64%)
- MTE duty: 34% → 55% (↑61%)

### 性能对比 (MACAC vs Torch)
- Torch (PyTorch fused):     1.001 ms
- MACAC baseline (ori):      0.531 ms
- MACAC optimized (best):    0.373 ms
- MACAC opt vs Torch:        2.68x faster!
- MACAC opt vs MACAC base:   1.42x faster

### 剩余风险
- 算子为memory-bound (thin matmul, 24 output channels)
- 进一步优化需要改变kernel结构或使用tensor core
- 当前达到局部最优，warp-shuffle是最大可优化方向

## Final rerun
- 命令: export MACA_PATH=/opt/maca/ && bash run.sh 10 100 0
- <time_before_opt>: 0.531 ms
- <time_after_opt>: 0.373 ms  
- <runtime_ratio>: 0.702
- <precision>: True
