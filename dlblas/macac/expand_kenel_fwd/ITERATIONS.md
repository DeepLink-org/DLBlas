# expand_kenel_fwd 优化记录

## 运行信息
- 任务路径: /mnt/opt_test/expand_kenel_fwd_run
- 容器: metax_gemm_opt
- 开始时间: 2026-06-26 (初始), 2026-06-29 (续)
- 验证命令: export MACA_PATH=/opt/maca/ && bash run.sh 10 100 0

## 目标签名
- 算子: expand_kenel_fwd (expand/unsqueeze -2 dimension)
- family: data-movement (broadcast/expand)
- dtype: float32
- shape: input [1, 1024, 1280] -> output [1, 1024, 4, 1280]
- layout: contiguous
- 主要瓶颈: memory-bound (纯数据搬移，无计算)。HBM 带宽约 1.55 TB/s。
- 关键假设: 向量化访问、合并内存访问、block/grid 配置调整、减少地址计算开销

## 参考文件读取记录
- references/hardware/c500.md: 硬件参数与优化方向
- references/verification.md: 验证流程与标签含义
- references/routing.md: 路由策略
- references/case_retrieval.md: 案例检索约束
- references/operator_families/elementwise.md: elementwise 族策略
- references/issues/memory_bottleneck.md: 内存瓶颈分析
- references/cases/relu_float4_grid_opt.md: ReLU float4 案例参考

---

## Session 1 (2026-06-26): 20 轮优化 (简要)

### 迭代 1: 2D grid 消除 div/mod
runtime_ratio从0.745降至0.325。保留。

### 迭代 2: float4向量化load+store
ratio: 0.300。保留。

### 迭代 5: 显式展开4次写入
ratio: 0.301。保留。

### 迭代 7: block_size=320, 无循环
ratio: 0.296。保留。

### 迭代 20: 最优模式恢复
ratio: 0.279。保留，作为 Session 1 最终版本。

其他迭代 (3,4,6,8-19): grid-stride, shared memory, scalar __ldg, launch_bounds, block_size 调优等 — 回退。

---

## Session 2 (2026-06-29): Trace Profiling + 9 轮优化

### Trace Profiling 结果 (v0 baseline)
- Kernel: expand_kenel_fwd_kernel_opt
- Grid/block: [1024,1,1] / [320,1,1]
- Bottleneck: memory (HBM bandwidth 92.74%)
- L2C hit rate: 0.72%
- ISU stall: vls_pipeline_stall 54.68%, vls_wdata_stall 45.32%
- Real IPC: 20.58
- Effective occupancy: 2.00%
- Achieved bandwidth: 1709 GB/s (Roofline)
- **结论**: 内存带宽是硬瓶颈，HBM 利用率接近峰值，优化空间极小

### Baseline (Round 0, Session 2)
- 命令: export MACA_PATH=/opt/maca/ && bash run.sh 10 100 0
- <time_before_opt>: 0.101581 ms
- <time_after_opt>: 0.030395 ms
- <runtime_ratio>: 0.299219
- <precision>: True

### 迭代 1 (v21): Multi-row per block (2 rows/block)
**假设**: 减少launch overhead，增加SM工作量
**结果**: 0.030147ms, ratio=0.301493, precision=True
**分析**: 性能基本持平，launch overhead 影响极小
**决策**: 回退

### 迭代 2 (v22): Float2 writes
**假设**: 不同对齐模式可能改善内存事务效率
**结果**: 0.031944ms, ratio=0.321043, precision=True
**分析**: Float2 加倍了store指令数，性能下降
**决策**: 回退

### 迭代 3 (v23): uint4 bitcast
**假设**: 使用整数类型避免浮点流水线
**结果**: 0.031076ms, ratio=0.298029, precision=True
**分析**: 与float4几乎相同，uint4无优势
**决策**: 回退

### 迭代 4 (v24): 1-warp block (64 threads)
**假设**: 更高 occupancy 隐藏内存延迟
**结果**: 0.034324ms, ratio=0.338902, precision=True
**分析**: grid-stride loop 开销 + 非合并访存降低性能
**决策**: 回退

### 迭代 5 (v25): #pragma unroll loop
**假设**: 编译器 unroll 优化可能改善调度
**结果**: 0.031240ms, ratio=0.290582, precision=True
**分析**: 与显式展开等价，ratio略好但abs time略差
**决策**: 回退

### 迭代 6 (v26): 160 threads, 2 float4/thread
**假设**: ILP隐藏内存延迟
**结果**: 0.029740ms, ratio=0.307409, precision=True
**分析**: abs time最好但ratio受baseline波动影响
**决策**: 回退 (ratio劣于v20)

### 迭代 7 (v27): Reverse M write order
**假设**: 改变写顺序影响内存控制器调度
**结果**: 0.030433ms, ratio=0.304446, precision=True
**分析**: 无显著差异
**决策**: 回退

### 迭代 8 (v28): 80 threads, 4 float4/thread
**假设**: 最大化ILP
**结果**: 0.030743ms, ratio=0.301552, precision=True
**分析**: 寄存器压力抵消ILP收益
**决策**: 回退

### 迭代 9 (v29): No bounds check + shift + launch_bounds
**假设**: 减少分支和整数除法开销
**结果**: 0.030431ms, ratio=0.305759, precision=True
**分析**: 微小优化被内存瓶颈掩盖
**决策**: 回退

---

## 最终结果
- **最终保留版本**: v20 (2D grid, 320 threads, float4 load, 4 explicit stores, no loop)
- **核心策略**: 2D grid + float4向量化 + 精确block_size + 显式展开
- **所有9轮Session 2优化均回退** — 内存带宽瓶颈 (HBM 92.74%) 无法通过软件优化突破

### Final rerun
- <time_before_opt>: 0.099453 ms
- <time_after_opt>: 0.030694 ms
- <runtime_ratio>: 0.308631
- <precision>: True

### Torch 性能对比 (2026-06-29, torch 2.8.0+metax3.3.0.2)
| 实现 | 时间 (ms) | vs MACA opt |
|------|-----------|-------------|
| Torch GPU (MACA backend) | 0.061174 | 1.99x slower |
| Torch CPU | 0.056635 | 1.85x slower |
| MACA ori kernel | 0.099453 | 3.24x slower |
| **MACA opt kernel** | **0.030694** | **baseline** |

- **MACA opt vs Torch GPU: 1.99x faster**
- **MACA opt vs MACA ori: 3.24x faster**

### 剩余风险
- HBM 带宽 (92.74%) 是硬极限，理论下限 ~0.017ms
- L2C 缓存命中率 0.72% — 数据流式传输，无缓存复用
- 不同输入 shape (非2的幂hidden_size) 需要tail handling
- VLS pipeline/wdata stall 占 ISU stall 的 100%，确认写带宽瓶颈
