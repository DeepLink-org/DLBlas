# ITERATIONS.md — engram_gate_fwd

## 运行信息
- 任务路径: /home/ailab/maca-vendor-workspace/maca_c_opt/workspace/engram_gate_fwd_run
- 容器: metax_gemm_opt
- 开始时间: 2026-06-26 13:46
- 结束时间: 2026-06-26 14:00
- 验证命令: `export MACA_PATH=/opt/maca/ && bash run.sh 10 100 0`

## 目标签名
- 算子: engram_gate_fwd
- Family: layernorm / gate-fusion (RMSNorm + dot reduction + signed_sqrt + sigmoid + gated_add)
- dtype: bfloat16 (uint16_t storage)
- shape: T=4096, M=4, D=4096
- layout: contiguous
- 主要瓶颈判断: 初始为 compute-bound，优化后转为 memory-bound

## 参考文件读取记录
- references/routing.md — 路由策略
- references/hardware/c500.md — C500 硬件指导
- references/verification.md — 验证流程
- references/case_retrieval.md — 案例检索

## Baseline (Round 0)
- Round 0 命令: `bash run.sh 10 100 0`
- <time_before_opt>: 0.757189 ms
- <time_after_opt>: 0.759150 ms
- <runtime_ratio>: 1.002590
- <precision>: True
- Profile: profile-artifacts/engram_gate_fwd_v0_baseline → compute-bound, MTE=54.12%, ISU stall(vls)=79.97%, L2C hit=39.64%, occupancy=4%

### 迭代 1: blockDim=512 提升 occupancy
**假设**: 更大 block 可减少总 block 数、提高 occupancy
**目标**: 尝试 blockDim=512
**结果**:
- commit: (rejected)
- <time_before_opt>: 0.757404 ms
- <time_after_opt>: 0.987443 ms
- <runtime_ratio>: 1.303720
- <precision>: True
**分析**: 更大的 block 增加 shared memory (6KB), 降低 occupancy, 性能变差
**决策**: 回退

### 迭代 2: warp shuffle reduction
**假设**: 使用 warp shuffle 替代 shared memory reduction 可减少 shared memory 使用
**目标**: 消除 shared memory reduction
**结果**:
- <precision>: False (41762 mismatches)
**分析**: __shfl_xor_sync 在 C500 上产生精度差异
**决策**: 回退

### 迭代 3: uint2 vectorized memory access
**假设**: 2-wide vectorized loads 可减少内存指令数
**目标**: 使用 uint2 指针加载 paired bf16
**结果**:
- XNACK error (ATU fault — misaligned access)
**分析**: bf16 数据不是 4 字节对齐, uint2 访问导致非法地址
**决策**: 回退

### 迭代 4: __ldg() for read-only inputs
**假设**: __ldg() 可通过 read-only cache 改善 cache 利用率
**目标**: 对只读输入使用 __ldg()
**结果**:
- <time_after_opt>: 0.759386 ms
- <runtime_ratio>: 1.002335
- <precision>: True
**分析**: 无改善, 编译器已优化只读访问
**决策**: 回退

### 迭代 5: blockDim=128
**假设**: 减少 block size 可减少 shared memory, 提高 occupancy
**目标**: 尝试 blockDim=128
**结果**:
- commit: 3d96409
- <time_before_opt>: 0.760773 ms
- <time_after_opt>: 0.654359 ms
- <runtime_ratio>: 0.860124
- <precision>: True
**分析**: 14% 提升, occupancy 改善有效
**决策**: 保留 ✓

### 迭代 6: #pragma unroll + multiply optimization
**假设**: loop unrolling + rsqrt 中用乘法代替除法可减少指令
**目标**: 添加 pragma unroll + 预计算 1/D
**结果**:
- <runtime_ratio>: 0.863335
- <precision>: True
**分析**: 与 blockDim=128 基础版本相比无明显改善
**决策**: 回退

### 迭代 7: blockDim=64
**假设**: 进一步减少 block size 可更大程度提高 occupancy
**目标**: 尝试 blockDim=64
**结果**:
- commit: 5a8d0f2
- <time_before_opt>: 0.760494 ms
- <time_after_opt>: 0.611782 ms
- <runtime_ratio>: 0.804453
- <precision>: True
**分析**: 20% 提升, 更小的 block 效果更好
**决策**: 保留 ✓

### 迭代 8: blockDim=64 + 4x unroll + fused weight
**假设**: 循环展开 + 预计算 wh*we 可减少每元素乘法和循环开销
**目标**: 4x unroll + fused weight
**结果**:
- commit: 7d99434
- <time_before_opt>: 0.764024 ms
- <time_after_opt>: 0.533332 ms
- <runtime_ratio>: 0.698057
- <precision>: True
**分析**: 30% 提升, unroll + fused weight 非常有效
**决策**: 保留 ✓

### 迭代 9: blockDim=64 + 8x unroll + fused weight ★BEST★
**假设**: 更激进的 8x unroll 可进一步减少循环开销
**目标**: 8x unroll + fused weight
**结果**:
- commit: 3fe496c
- <time_before_opt>: 0.759529 ms
- <time_after_opt>: 0.483203 ms
- <runtime_ratio>: 0.636187
- <precision>: True
**分析**: 36% 提升! 8x unroll 比 4x 效果更好(减少循环开销)
**决策**: 保留 ✓ (BEST)

### 迭代 10: blockDim=32 + 8x unroll
**假设**: 极端小的 block size 可能进一步提高 occupancy
**目标**: 尝试 blockDim=32
**结果**:
- <runtime_ratio>: 0.968044
- <precision>: True
**分析**: 性能变差, warpSize=64 上 blockDim=32 导致半 warp 空闲
**决策**: 回退

### 迭代 11: blockDim=64 + 16x unroll
**假设**: 16x unroll 可进一步减少循环开销
**目标**: 尝试 16x unroll
**结果**:
- <runtime_ratio>: 0.641004
- <precision>: True
**分析**: 与 8x 相似但略差, 寄存器压力增大抵消了 unroll 收益
**决策**: 回退

### 迭代 12: __launch_bounds__ hint
**假设**: launch_bounds 可帮助编译器优化寄存器分配
**目标**: 添加 __launch_bounds__(64, 16)
**结果**:
- <runtime_ratio>: 0.633616
- <precision>: True
**分析**: 与 8x unroll 基本相同, 提高 minBlocksPerSM 无额外收益
**决策**: 回退(边际改进)

### 迭代 13: 移除 __syncthreads (warp-synchronous)
**假设**: blockDim=64=warpSize, 单 warp 内无需 barrier
**目标**: 移除所有 __syncthreads
**结果**:
- <runtime_ratio>: 0.635683
- <precision>: True
**分析**: 编译器可能已优化单 warp 的 barrier, 无额外收益
**决策**: 回退(边际改进)

### 迭代 14: shared memory precompute fused weight
**假设**: 在 shared memory 预计算 wh*we 减少全局内存访问
**目标**: 将 fused_w 存入 shared memory
**结果**:
- <runtime_ratio>: 2.959719
- <precision>: True
**分析**: 16KB shared memory 严重限制 occupancy (仅 3 blocks/SM)
**决策**: 回退

### 迭代 15: blockDim=128 + 8x unroll
**假设**: blockDim=128 可能平衡 occupancy 和 compute
**目标**: blockDim=128 + 8x unroll
**结果**:
- <runtime_ratio>: 0.646076
- <precision>: True
**分析**: 比 blockDim=64 + 8x 稍差
**决策**: 回退

## Profile Comparison: Baseline vs Optimized

| Metric | Baseline (v0) | Optimized (v1) | Change |
|---|---:|---:|---|
| Kernel span (cycles) | 442,893,679 | 420,988,572 | -5% |
| Total instructions | 235,327 | 136,242 | -42% |
| MTE share | 54.12% | 66.18% | +12pp |
| NOP share | 13.25% | 6.25% | -7pp |
| GVM share | 7.84% | 12.94% | +5pp |
| Registers/thread | 24 | 60 | +36 |
| AP MTE duty | 65.44% | 73.66% | +8pp |
| ISU stall % | 79.97% | 84.46% | +4.5pp |
| HBM usage | 42.65% | 68.53% | +26pp |
| Achieved BW (GB/s) | 786 | 1,263 | +61% |
| Achieved FLOPS (TF) | 5.02 | 5.80 | +16% |
| Bound type | compute | memory | shifted |
| L2C hit rate | 39.64% | 38.98% | -0.7pp |

## 最终结果
- 最终保留版本: blockDim=64 + 8x unroll + fused weight (commit: 3fe496c)
- 最佳策略: 减少 block size 提高 occupancy + 8x 循环展开减少指令 + 预计算 wh*we 融合权重
- 次要瓶颈: 寄存器压力 (60 regs/thread 限制 occupancy), L2C hit rate (~39%)
- Final rerun 四个输出标签:
  - <time_before_opt>: 0.758090 ms
  - <time_after_opt>: 0.480973 ms
  - <runtime_ratio>: 0.634453
  - <precision>: True

## MACAC vs Torch 性能比较
- MACAC optimized: 0.481 ms
- MACAC baseline: 0.758 ms
- Torch (PyTorch 2.8.0): 13.728 ms
- MACAC opt vs Torch speedup: 28.6x
- MACAC opt vs MACAC baseline speedup: 1.58x

## 剩余风险
- 寄存器压力 (60/thread) 限制了 occupancy, 进一步优化可能需要减少 unroll 因子
- L2C hit rate 仍然偏低 (~39%), 更高效的访存模式可能进一步改善
- 当前优化仅在 shape T=4096, M=4, D=4096 上验证, 其他 shape 可能需要不同配置
