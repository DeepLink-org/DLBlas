# ITERATIONS.md — mhc_post MACA 算子优化

## 运行信息
- 任务路径: /home/ailab/opt_test/mhc_post_run
- 容器: metax_gemm_opt
- 开始时间: 2026-06-26
- 结束时间: 2026-06-26 18:26
- 验证命令: export MACA_PATH=/opt/maca/ && bash run.sh 10 100 0

## 目标签名
- 算子: mhc_post (multi-head convolution post-processing)
- Family: matmul + elementwise fusion (batched small matmul 4x4 @ 4x1280 + broadcast mul + add)
- dtype: bf16 inputs (x, residual), fp32 intermediates (post_layer_mix, comb_res_mix), bf16 output
- shape: n0=2, n1=4096, h=1280, mhc_mult=4
- layout: contiguous in inner dims
- 主要瓶颈判断: memory-bound (large residual tensor 40M elements), non-coalesced access across m dimension
- 关键假设: residual 跨 m 维度的 strided access (stride=h=1280) 导致缓存效率低

## 参考文件读取记录
- references/routing.md — 了解路由和问题文件加载策略
- references/hardware/c500.md — C500 硬件参数 (warp=64, SM=104, HBM 1.55TB/s, shared memory=64KB)
- references/operator_families/matmul.md — matmul family 优化策略
- references/operator_families/elementwise.md — elementwise family 优化策略
- references/verification.md — 验证流程和标签含义

## Baseline

### Round 0
**命令**: `bash run.sh 10 100 0`
**结果**:
- <time_before_opt>: 0.167980 ms
- <time_after_opt>: 0.168563 ms
- <runtime_ratio>: 1.003475
- <precision>: True

**分析**: 初始 kernel 功能正确，opt 和 ori 版本相同 (初始状态)。
**commit**: baf24d8

## Trace Profiling (Baseline)
- Run: profile-artifacts/mhc_post_v0_baseline
- Bottleneck: compute (MTE share=62.6%), but effectively memory-bound
- L2C hit rate: 0.58% (极低 — 缓存几乎无效)
- HBM bandwidth usage: 66.86%
- Effective occupancy: 8.00%
- Key finding: memory-bound, room for bandwidth improvement

## 优化迭代

### 迭代 1: 增大 grid 到 3328 + __ldg() 只读优化
**假设**: 更大的 grid 覆盖更多 SM，__ldg() 改善只读缓存命中
**结果**:
- commit: 69429cf
- <time_before_opt>: 0.168876 ms
- <time_after_opt>: 0.163261 ms
- <runtime_ratio>: 0.966756
- <precision>: True
**分析**: 3.3% 改善，__ldg() + 大 grid 有效
**决策**: 保留

### 迭代 2: block=256 降低寄存器压力
**假设**: 更小的 block 提高 occupancy
**结果**:
- <time_after_opt>: 0.166249 ms
- <runtime_ratio>: 0.991905
- <precision>: True
**分析**: 性能变差，block=512 更优（更多 ILP）
**决策**: 回退

### 迭代 3: h-loop unroll x2 + crm/plm hoisting
**假设**: 减少循环开销，ILP 翻倍
**结果**:
- <time_after_opt>: 0.335921 ms
- <runtime_ratio>: 1.983688
- <precision>: True
**分析**: 严重变差（~2x 慢），寄存器压力过大导致 occupancy 崩溃
**决策**: 回退

### 迭代 4: 紧凑 __ldg() + block=256
**假设**: 紧凑的代码降低寄存器用量 + 小 block 提高 occupancy
**结果**:
- <time_after_opt>: 0.167107 ms
- <runtime_ratio>: 0.981357
- <precision>: True
**分析**: 轻微改善但不如迭代1
**决策**: 保留

### 迭代 5: __launch_bounds__(512,4) + #pragma unroll
**假设**: 提示编译器优化 occupancy 和循环展开
**结果**:
- <time_after_opt>: 0.163502 ms
- <runtime_ratio>: 0.973108
- <precision>: True
**分析**: 2.7% 改善，但 MACA 忽略了 min_blocks 参数
**决策**: 保留

### 迭代 6: bfloat162 向量化存储
**假设**: 半减 store 指令数
**结果**: Memory Violation (对齐错误)
**分析**: bfloat162 需要 4 字节对齐，threadIdx 奇偶导致不对齐
**决策**: 回退

### 迭代 7: 数组 unroll 替换逐变量展开
**假设**: 编译器更容易优化数组形式的循环
**结果**:
- <time_after_opt>: 0.163602 ms
- <runtime_ratio>: 0.962223
- <precision>: True
**分析**: 3.8% 改善，pragma unroll + 数组形式有效
**决策**: 保留

### 迭代 8: crm/plm 提升到 h-loop 外部
**假设**: 消除重复加载 20 个 float
**结果**:
- <time_after_opt>: 0.163466 ms
- <runtime_ratio>: 0.976018
- <precision>: True
**分析**: 2.4% 改善，编译器可能已自动 hoist
**决策**: 保留

### 迭代 9: 2D grid (每 batch_seq 一个 block) + hoist crm/plm
**假设**: 消除 grid-stride 循环开销，固定 batch_seq 简化索引
**结果**:
- commit: 17f869f
- <time_after_opt>: 0.163036 ms
- <runtime_ratio>: 0.962170
- <precision>: True
**分析**: 3.8% 改善，最佳绝对时间，简化控制流有效
**决策**: 保留

## 最终结果
- 最终保留版本: 迭代 9 (2D grid + hoist crm/plm + __ldg())
- 最终 commit: 2b64dc1
- Final rerun (500 iterations):
  - <time_before_opt>: 0.167706 ms
  - <time_after_opt>: 0.161676 ms
  - <runtime_ratio>: 0.964048
  - <precision>: True
- 优化幅度: ~3.6% (memory-bound kernel, HBM bandwidth 为主要限制)
- Rejected variants: 迭代 2, 3, 6 (性能倒退或崩溃)
- 剩余风险: 本 kernel 天然 memory-bound (42M 输出元素, ~180MB 内存流量), 进一步优化空间有限
- 主要瓶颈: HBM bandwidth (L2C hit rate only 0.58%, working set too large for cache)
