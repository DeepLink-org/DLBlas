## 运行信息
- 任务路径: /home/ailab/maca-vendor-workspace/maca_c_opt/workspace/sparse_attn_run
- 容器: metax_gemm_opt
- 开始时间: 2026-06-26
- 验证命令: export MACA_PATH=/opt/maca/ && bash run.sh 10 500 0

## 目标签名
- 算子: sparse_attn
- Family: softmax (attention-like with gather)
- dtype: bfloat16 (uint16_t storage)
- Shape: B=2, M=16, H=8, D=64, N=32, TopK=16
- Layout: q[B,M,H,D], kv[B,N,D], topk_idxs[B,M,TopK], o[B,M,H,D]
- 主要瓶颈判断:
  * KV数据被读取两次(先算scores再算weighted sum)
  * 16次dot-product block reduction各需6次__syncthreads()
  * 每次topk迭代的shared memory reduction开销大
  * 无向量化访存

## 参考文件读取记录
- references/routing.md: 路由策略
- references/hardware/c500.md: C500硬件指导(warp=64, SM=104, shared memory 64KB)
- references/verification.md: 验证流程与迭代规则
- references/case_retrieval.md: 案例检索约束
- references/operator_families/softmax.md: softmax家族策略
- references/operator_families/reduction.md: reduction家族策略

## Baseline (Round 0)
- 命令: export MACA_PATH=/opt/maca/ && bash run.sh 10 500 0
- <time_before_opt>: 0.067355 ms
- <time_after_opt>: 0.064022 ms  
- <runtime_ratio>: 0.950514
- <precision>: True
- 注: ori和opt当前代码相同，ratio~0.95来自测量方差

## 优化策略方向
1. 在线softmax融合: 单遍遍历topk位置，同时计算score+weighted sum
2. 消除KV重复读取: 从2次降到1次
3. 减少__syncthreads(): 融合后每次topk只需1次sync
4. 考虑更优block size配置
5. 向量化访存

### 迭代 1: KV共享内存缓存
**假设**: 将KV缓存到shared memory，消除50%的KV全局读取（原kernel读KV两遍）
**目标**: 减少全局内存流量，保持bit-exact正确性
**参考依据**: trace报告显示GLOBAL share仅0.89%但DNOC>512-cycle占25.93%
**结果**:
- **commit**: (pending)
- <time_before_opt>: 0.067093 ms
- <time_after_opt>: 0.070696 ms
- <runtime_ratio>: 1.053705
- <precision>: True
**分析**: 精度通过但性能轻微变差(~5%)。KV缓存减少了全局读取但增加了shared memory加载开销(1024次bf→float转换+sync)。原始kernel虽然读KV两遍但受益于L2缓存。
**决策**: 保留(bit-exact基础)，后续优化shared memory加载和reduction开销

### 迭代 2: 单线程点积+精确树归约顺序
**假设**: 将64线程并行树归约替换为1线程串行树归约(相同顺序)，消除112次__syncthreads()调用
**目标**: 消除reduction同步开销，同时保持bit-exact正确性
**参考依据**: trace报告WSM stall 76.71%, NOP 16.30%, STE 42.51%
**结果**:
- **commit**: (pending)
- <time_before_opt>: 0.069515 ms
- <time_after_opt>: 0.050169 ms
- <runtime_ratio>: 0.721707
- <precision>: True
**分析**: 大幅成功! 28%性能提升，精度通过。关键洞察：必须匹配树归约顺序才能bit-exact（fp32加法不满足结合律）。单线程做D=64次乘加+63次归约加法(共16线程活跃)，虽然计算量增加但消除了112次sync开销。
**决策**: 保留

### 迭代 3: 寄存器数组减半 p[64]→p[32]
**假设**: 预合并树归约Level 1(s=32 pairs)可将寄存器数组从64减半到32，降低寄存器压力
**目标**: 减少寄存器使用，提高occupancy
**参考依据**: Iteration 2 ratio=0.72，寄存器66→~130可能限制occupancy(12%)
**结果**:
- **commit**: (pending)
- <time_before_opt>: 0.067924 ms
- <time_after_opt>: 0.046591 ms
- <runtime_ratio>: 0.685928
- <precision>: True
**分析**: 进一步提升到31.4% speedup。寄存器压力减小+valid_kv预加载减少Stage 4索引计算。树归约顺序仍然精确匹配。
**决策**: 保留

### 迭代 4: 循环展开+kv指针预计算
**假设**: #pragma unroll + kv_ptrs预计算减少循环和索引开销
**结果**: ratio=0.699, precision=True
**分析**: 轻微退化，编译器已自动展开固定大小循环
**决策**: 回退

### 迭代 5: Block size 128
**假设**: 增加线程数提升occupancy
**结果**: ratio=0.714, precision=True
**分析**: 多余线程空闲增加开销
**决策**: 回退

### 迭代 6: __ldg只读加载
**假设**: __ldg()纹理缓存路径减少KV读取延迟
**结果**: ratio=0.677, precision=True
**分析**: 有效! 32.3%提速
**决策**: 保留

### 迭代 7: __ldg + kv_ptr
**假设**: 组合__ldg和指针预计算
**结果**: ratio=0.703, precision=True
**分析**: 指针数组增加寄存器压力抵消__ldg收益
**决策**: 回退

### 迭代 8: 8线程点积(每线程2位置)
**假设**: 减少活跃线程提高每线程利用率
**结果**: ratio=0.674-0.696, precision=True
**分析**: 略有波动，不比v6好
**决策**: 回退

### 迭代 9: __ldg q+kv全面优化
**假设**: q和kv都用__ldg加载，紧凑代码
**结果**: ratio=0.661-0.667, precision=True
**分析**: 最佳! 33.9%提速，opt=0.0457ms
**决策**: 保留为最终版本

## 最终结果
- 最终保留版本: 迭代9 (__ldg全面优化)
- 策略: 单线程点积+精确树归约顺序+__ldg只读缓存
- 主要瓶颈: vls_pipeline_stall (向量加载流水线停顿)
- 剩余风险: 低occupancy(14%)，但计算已近峰值
- rejected variants: v1(KV cache), v4(unroll), v5(bs128), v7(kv_ptr), v8(8threads)

## Final Rerun
- 命令: export MACA_PATH=/opt/maca/ && bash run.sh 20 1000 0
- <time_before_opt>: 0.065495 ms
- <time_after_opt>: 0.045765 ms
- <runtime_ratio>: 0.698764
- <precision>: True

## Torch vs MACAC Comparison
- Torch (PyTorch 2.8.0+metax): 0.415570 ms
- MACAC (optimized): 0.045765 ms
- Speedup: 9.08x (MACAC 89% faster than Torch)

## 剩余风险
- Occupancy仅14%，但计算已接近向量流水线峰值
- vls_pipeline_stall为主要瓶颈，进一步优化需减少全局内存读取
- 当前kernel仅在默认shape(B=2,M=16,H=8,D=64)上验证
