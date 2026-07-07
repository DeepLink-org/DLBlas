## 运行信息
- 任务路径: /mnt/opt_test/head_compute_mix_fwd_run
- 容器: metax_gemm_opt
- 开始时间: 2026-06-29

## 目标签名
- 算子: head_compute_mix_fwd
- family: elementwise
- dtype: float32
- shape: [16, 16384, 4] (total=1048576)
- 语义: output = sigmoid(input_mix * mhc_scale + mhc_base) + mhc_pre_eps
- 主要瓶颈: memory-bound elementwise（待 trace 确认）
- 关键假设: C500 warp=64, SM=~104, block=512 起点

## 参考文件读取记录
- references/routing.md — 路由策略
- references/hardware/c500.md — C500 硬件指导
- references/verification.md — 验证流程
- references/operator_families/elementwise.md — Elementwise 优化策略

## Baseline
- Round 0 命令: bash run.sh 10 500 0
- <time_before_opt>: 0.033345 ms
- <time_after_opt>: 0.031452 ms
- <runtime_ratio>: 0.943233
- <precision>: True

## Trace Analysis (Baseline v0)
- 瓶颈: compute-classified but ISU vls_pipeline_stall=88% → 真正瓶颈是 memory latency
- L2C hit rate: 21.72% (低)
- HBM bandwidth: 327.79 GB/s (17.78% peak)
- MTE share: 56.52%, IPC: 59.60
- 优化方向: grid-stride loop, float4 vectorization, __ldg, block size tuning


### 迭代 1: Grid-stride loop + capped grid
**假设**: Capped grid-stride 减少 block 调度开销，提高 SM 覆盖率
**目标**: 用 capped grid (104*32 blocks) + grid-stride loop
**结果**:
- <time_before_opt>: 0.031904 ms
- <time_after_opt>: 0.031319 ms
- <runtime_ratio>: 0.981641
- <precision>: True
**分析**: 微小提升，grid-stride loop 对此规模帮助有限（total=~1M, grid=2048 已足够覆盖 SM）
**决策**: 回退，准备 float4 向量化

### 迭代 2: float4 向量化加载/存储
**假设**: MHC=4 天然匹配 float4 宽度，向量化可大幅减少访存次数
**目标**: 每次处理 4 个元素（一个 float4），scalar tail 处理余数
**参考依据**: elementwise family + C500 hardware guide (连续访存与向量化)
**结果**:
- <time_before_opt>: 0.032568 ms
- <time_after_opt>: 0.020528 ms
- <runtime_ratio>: 0.630303
- <precision>: True
**分析**: 37% 加速！float4 向量化显著减少 load/store 次数
**决策**: 保留 ✅

### 迭代 3: float4 + capped grid-stride + block=512
**假设**: 组合向量化与 capped grid-stride
**结果**: runtime_ratio=0.652711, precision=True
**决策**: 回退（比 Round 2 差）

### 迭代 4: float4 + block_size=256
**假设**: 更多 blocks 提供更好的延迟隐藏
**目标**: block=256, float4 向量化
**结果**:
- <time_before_opt>: 0.033524 ms (varies)
- <time_after_opt>: 0.019961 ms
- <runtime_ratio>: 0.595430
- <precision>: True
**分析**: 40.5% 加速！block=256 比 512 更好，因为更多 blocks 提供更好的 wave 级并行
**决策**: 保留 ✅ (最佳版本)

### 迭代 5: float4 + block_size=128
**假设**: 更多 blocks = 更好的延迟隐藏
**结果**: runtime_ratio=0.607784, precision=True
**决策**: 回退（不如 block=256）

### 迭代 6: float4 + block=256 + preload scalars
**假设**: 预加载所有标量到寄存器减少冗余 __ldg
**结果**: runtime_ratio=0.617369, precision=True
**决策**: 回退（额外寄存器预加载无益）

### 迭代 7: float4 + block=64 + launch_bounds
**假设**: 更小 block + launch_bounds 提高 occupancy
**结果**: runtime_ratio=0.614267, precision=True
**决策**: 回退

### 迭代 8: float4 + block=256 + direct access (no __ldg on small arrays)
**假设**: 小数组直接访问利用 L1 cache
**结果**: runtime_ratio=0.611140, precision=True
**决策**: 回退（__ldg 对大数组读仍更优）

### 迭代 9: float4 + block=256 + capped grid-stride
**假设**: SM 倍数 capped grid-stride 减少调度开销
**结果**: runtime_ratio=0.634662, precision=True
**决策**: 回退（grid-stride 开销大于收益）

## 最终结果
- 最佳版本: 迭代 4 (float4 + block=256)
- Final rerun (500 iters):
  - <time_before_opt>: 0.032542 ms
  - <time_after_opt>: 0.019766 ms
  - <runtime_ratio>: 0.607387
  - <precision>: True
- 加速比: ~1.65x (约 39% 加速)
- 保留策略: float4 向量化 + block_size=256
- rejected variants: grid-stride loop, block=64/128/512, preload scalars, capped grid
- 剩余风险: 低，核心优化（float4+block tuning）已充分探索
