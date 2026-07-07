## 运行信息
- 任务路径: /home/ailab/maca-vendor-workspace/maca_c_opt/workspace/pre_split_mixes_run/
- 容器: metax_gemm_opt
- 开始时间: 2026-06-26
- 验证命令: export MACA_PATH=/opt/maca/ && bash run.sh 10 100 0

## 目标签名
- 算子: pre_split_mixes
- Family: elementwise (scale + bias + sigmoid + split/reshape)
- dtype: float32
- shape: B=1, N=1024, M=4, M3=24
- layout: contiguous
- 主要瓶颈判断: compute-bound (MTE=59.56%), low occupancy (4%), tiny workload (1024 rows × 24 ops)
- 关键假设: M=4 small constant, per-row independent, tiny kernel

## 参考文件读取记录
- references/routing.md: 路由策略
- references/hardware/c500.md: C500硬件指导
- references/verification.md: 验证流程
- references/operator_families/elementwise.md: elementwise策略
- trace-report: profile-artifacts/pre_split_mixes_v0_baseline/ (compute-bound, MTE主导, 4% occupancy)

## Baseline (Round 0)
- 命令: MACA_VISIBLE_DEVICES=0 ./test_maca 10 100 0
- <time_before_opt>: 0.040632 ms
- <time_after_opt>: 0.031862 ms  (block=512)
- <runtime_ratio>: 0.784148
- <precision>: True

### 迭代 1: Fused loops (pre+post in one loop)
**假设**: 合并循环减少循环开销
**结果**: opt=0.033434 ms, ratio=0.726604, precision=True
**分析**: 性能变差，合并循环增加寄存器压力，M=4循环开销本身很小
**决策**: 回退

### 迭代 2: __ldg() for read-only data
**假设**: 使用只读缓存减少内存延迟
**结果**: opt=0.032850 ms, ratio=0.795142, precision=True
**分析**: __ldg()增加额外开销，对小数据无益
**决策**: 回退

### 迭代 3: Manual unroll M=4
**假设**: 手动展开消除循环分支
**结果**: opt=0.031869 ms, ratio=0.680869, precision=True
**分析**: 与baseline持平，编译器已自动展开小循环
**决策**: 回退

### 迭代 4: float4 vectorization for comb output
**假设**: 128-bit向量化读写提高带宽
**结果**: opt=0.032778 ms, ratio=0.735608, precision=True
**分析**: float4组装/拆解开销抵消向量化收益
**决策**: 回退

### 迭代 5: block=256
**假设**: 减少block大小增加SM覆盖
**结果**: opt=0.029745 ms, ratio=0.699392, precision=True
**分析**: 4 blocks覆盖4个SM，比2 blocks好
**决策**: 保留，继续探索

### 迭代 6: block=128
**假设**: 更多blocks=更好SM利用率
**结果**: opt=0.028124 ms, ratio=0.617677, precision=True
**分析**: 8 blocks，继续改善
**决策**: 保留方向，继续减小

### 迭代 7: block=64→32 sweep
**假设**: 进一步增加SM覆盖
**结果**: block=64: 0.027341 ms; block=32: 0.026408 ms
**分析**: block=32 (32 blocks) 最佳，充分利用SM
**决策**: 保留block=32

### 迭代 8: block=32 + #pragma unroll + grid-stride
**假设**: unroll减少循环开销，grid-stride提高可扩展性
**结果**: opt=0.026017 ms, ratio=0.598422, precision=True
**分析**: unroll有小幅改善
**决策**: 保留

### 迭代 9: __expf + __fdividef fast math
**假设**: MACA native快速数学函数加速sigmoid
**结果**: opt=0.025802 ms, ratio=0.653377, precision=True
**分析**: __expf比expf更快，__fdividef比除法快
**决策**: 保留为最佳版本

## 最终结果
- 最终保留版本: 迭代9 (block=32 + unroll + __expf/__fdividef)
- rejected variants: fused loops, __ldg, manual unroll, float4, block>=64
- Final rerun:
  - <time_before_opt>: 0.043256 ms
  - <time_after_opt>: 0.025943 ms
  - <runtime_ratio>: 0.599751
  - <precision>: True
- 优化幅度: baseline 0.031862→0.025943 ms (18.6% improvement)
- 剩余风险: 小kernel波动较大；M≠4时需回退到通用循环版本
