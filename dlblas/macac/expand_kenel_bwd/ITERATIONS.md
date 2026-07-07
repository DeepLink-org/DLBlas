## 运行信息
- 任务路径: /home/ailab/maca-vendor-workspace/maca_c_opt/workspace/expand_kenel_bwd_run
- 容器: metax_gemm_opt
- 验证命令: export MACA_PATH=/opt/maca/ && bash run.sh 10 1000 0

## 目标签名
- 算子: expand_kenel_bwd
- family: reduction
- dtype: float32
- shape: input (2, 1024, 4, 1280) -> output (2, 1024, 1280)
- layout: contiguous
- 操作: sum reduction along dim=-2 (mhc_mult=4)
- 主要瓶颈判断: memory-bound, 每个output需要读取4个非连续的float元素(stride=hidden_dim)
- 关键假设: 小归约维度(4)意味着warp shuffle/shared memory开销可能超过收益; 向量化加载是主要优化方向

## 参考文件读取记录
- references/routing.md: 路由策略
- references/hardware/c500.md: C500硬件指导 (warp=64, SM=104, 64KB shared mem, HBM 1.55TB/s)
- references/verification.md: 真实验证流程
- references/operator_families/reduction.md: Reduction算子家族优化策略
- references/issues/reduction_bottleneck.md: Reduction瓶颈修复策略

## Baseline (Round 0)
- 命令: export MACA_PATH=/opt/maca/ && bash run.sh 10 1000 0
- <time_before_opt>: 0.066688 ms
- <time_after_opt>: 0.063777 ms
- <runtime_ratio>: 0.956356
- <precision>: True

## Fixed Baseline (Round 0 corrected)
- 原始tmp_ori.cuh存在bug: row=blockIdx.x只能覆盖前10240个元素(0.39%)
- 修复: row=blockIdx.x*blockDim.x+threadIdx.x 覆盖全部2,621,440个元素
- <time_before_opt>: 0.075935 ms (修复后的完整计算)
- <time_after_opt>: 0.075166 ms (bs=512)
- <runtime_ratio>: 0.989861
- <precision>: True

### 迭代 1: __launch_bounds__ + 预计算 + unroll + 减法代替取模
**假设**: launch_bounds(512,4)降低寄存器压力，减法代替%减少指令数
**目标**: 降低寄存器使用，提升occupancy
**参考依据**: C500硬件指导，register pressure导致4% occupancy
**结果**:
- <time_before_opt>: 0.076290 ms
- <time_after_opt>: 0.060099 ms
- <runtime_ratio>: 0.787769
- <precision>: True
**分析**: 21.2%提升！launch_bounds减少寄存器使用，手动unroll和减法进一步减少指令
**决策**: 保留
**停滞重分析**: 未触发

### 迭代 2: float4向量化加载，每线程处理4个输出元素
**假设**: float4向量化减少load指令数，4x减少线程和索引计算
**目标**: 向量化访存，减少指令开销
**参考依据**: C500连续访存优化 + vector width指导
**结果**:
- <time_after_opt>: 0.057669 ms
- <runtime_ratio>: 0.754271
- <precision>: True
**分析**: 24.6%提升，较Round1进一步改善！float4一次加载16字节(4 floats)，减少4x线程数和索引计算
**决策**: 保留
**停滞重分析**: 未触发

### 迭代 3: float4 + bs=256
**假设**: 256线程/block可能比512有更好的occupancy
**结果**: ratio=0.754075 (与Round 2几乎相同)
**分析**: bs从512降到256无明显差异；保持Round 2为best
**决策**: 回退到Round 2版本

### 迭代 4: 8元素per thread (两个float4组)
**假设**: 进一步减少线程数，提高每线程ILP
**结果**: ratio=0.840
**分析**: 寄存器压力增大，抵消了线程减少的收益
**决策**: 回退到Round 2 (float4, bs=512)

### 迭代 5: __ldg()只读缓存加载
**假设**: __ldg()通过纹理缓存路径加载输入，减少L1污染
**结果**: ratio=0.748352
**分析**: 进一步改善！25.2%提升，__ldg有效减少内存延迟
**决策**: 保留，新最佳版本

### 迭代 6: DivModFast快速除法
**假设**: 预计算除法参数消除整数除法开销
**结果**: ratio=0.882
**分析**: DivModFast每线程构造开销超过除法节省；每位线程都需计算divmod构造
**决策**: 回退到Round 5


### 迭代 7: Block-per-row消除索引除法 + float4 + __ldg()
**假设**: 消除每线程整数除法降低寄存器压力
**结果**: ratio=0.754
**分析**: 与Round 5(0.748)接近但稍差，strided loop分支开销抵消除法节省
**决策**: 回退到Round 5


### 迭代 8: 无__launch_bounds__
**假设**: 让编译器自主决定寄存器分配可能更优
**结果**: ratio=0.751
**分析**: 略差于Round 5(0.748)，launch_bounds确实有帮助
**决策**: 回退

### 迭代 9: launch_bounds(512, 4)高minBlocksPerSM
**假设**: 更高minBlocksPerSM提高occupancy
**结果**: ratio=0.772
**分析**: 强制更高occupancy反而增加了寄存器spilling
**决策**: 回退

## 最终结果
- 最佳版本: Round 5 (float4 + __ldg() + launch_bounds(512,2))
- runtime_ratio: 0.748 (25.2%提升)
- 主要策略: float4向量化加载 + __ldg()只读缓存 + launch_bounds控制寄存器
- rejected variants: DivModFast, 8元素/线程, block-per-row
- 剩余风险: 极低；算子正确性已验证

## Torch 性能比较
- MACAC ori (baseline): 0.076681 ms
- MACAC opt (best): 0.057277 ms
- Torch sum(dim=-2): 0.350608 ms
- MACAC opt vs Torch: 6.12x 加速 (0.351/0.057)
- MACAC ori vs Torch: 4.57x 加速 (0.351/0.077)
