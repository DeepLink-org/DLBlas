# norm_fn 优化迭代记录

## 运行信息
- 任务路径: /home/ailab/opt_test/norm_fn_run
- 容器内路径: /home/ailab/opt_test/norm_fn_run
- 开始时间: 2026-06-26
- 结束时间: 2026-06-29
- 验证容器: metax_gemm_opt
- 验证命令: export MACA_PATH=/opt/maca/ && bash run.sh 10 100 0

## 目标签名
- 算子: norm_fn
- Family: 自定义融合归一化 (layernorm-like + reduction + matmul-like)
- 语义: output[i,j] = dot(residual[i,:], mhc_fn[j,:]) * rsqrt(sqrsum[i]/rms_group_size + eps)
- dtype: float
- shape: num_rows=13, num_mixes=24, rms_group_size=5120
- layout: contiguous
- 主要瓶颈判断: reduction barrier overhead, 冗余 sqrsum, low occupancy (312 blocks / 104 SM)
- 关键假设: C500 warp=64, block=256, SM≈104

## 参考文件读取记录
- references/hardware/c500.md: 硬件特性指导 (所有轮次)
- references/verification.md: 验证流程
- references/routing.md: 路由策略
- references/case_retrieval.md: 案例检索策略

## Baseline (Round 0)
- 命令: export MACA_PATH=/opt/maca/ && bash run.sh 10 100 0
- <time_before_opt>: 0.051346 ms
- <time_after_opt>: 0.035868 ms
- <runtime_ratio>: 0.698559
- <precision>: True

### 迭代 1: float2 合并 reduction
**假设**: 两次独立 reduction (dot, sqrsum) 的 sync barrier 是主要开销
**目标**: 用 float2 packing 合并为单次 reduction，减半 barrier
**参考依据**: trace-report 显示 NOP=18%, STE=44% (barrier overhead)
**结果**:
- **commit**: 1e90fd7
- <time_before_opt>: 0.048417 ms
- <time_after_opt>: 0.036797 ms
- <runtime_ratio>: 0.760006
- <precision>: True
**分析**: 编译通过，正确。噪声大，小 kernel 测量波动显著
**决策**: 保留作为基线

### 迭代 2: float4 vectorized loads + float2 fused reduction
**假设**: float4 向量化加载可减少 75% load 指令
**目标**: 用 float4 一次加载 4 个 float，与 float2 reduction 组合
**参考依据**: C500 hardware 指南建议合并向量化访问
**结果**:
- **commit**: fd36ef3
- <time_before_opt>: 0.051999 ms
- <time_after_opt>: 0.028828 ms
- <runtime_ratio>: 0.554401
- <precision>: True
**分析**: **BEST VERSION**. float4 显著减少 loop iterations (20→5 次迭代)
**决策**: **保留为最佳版本**

### 迭代 3: warp shuffle reduction + block=128
**假设**: warp shuffle 比 shared memory reduction 更快
**目标**: 用 __shfl_down_sync 替代 shared mem reduction
**参考依据**: C500 warp=64, warp shuffle 避免 shared memory traffic
**结果**:
- **commit**: (reverted)
- <runtime_ratio>: 0.820834
- <precision>: True
**分析**: 性能回退，warp shuffle 指令开销 > shared mem
**决策**: **回退**

### 迭代 4: block=512
**假设**: 更多 threads/block 减少总 block 数，减少 launch overhead
**目标**: 测试 block=512
**结果**:
- <runtime_ratio>: 0.625329
- <precision>: True
**分析**: 比最佳版本差
**决策**: **回退**

### 迭代 5: block=128
**假设**: 更多 blocks 提高 SM 覆盖率
**目标**: 测试 block=128
**结果**:
- <runtime_ratio>: 0.592088
- <precision>: True
**分析**: 比最佳版本差
**决策**: **回退**

### 迭代 6: __ldg 读优化
**假设**: __ldg 缓存 hint 可提升只读数据吞吐
**目标**: 添加 __ldg 到只读 load
**结果**:
- <runtime_ratio>: 0.673121
- <precision>: True
**分析**: __ldg 增加指令开销，抵消缓存收益
**决策**: **回退**

### 迭代 7: precompute inv_group_size
**假设**: 用乘法替代除法可减少指令
**目标**: 预计算 1.0f/rms_group_size
**结果**:
- <runtime_ratio>: 0.582401
- <precision>: True
**分析**: 微小改善但不明显
**决策**: **回退**

### 迭代 8: warp-sync 第一级 reduction
**假设**: 两级 reduction (warp + block) 比纯 shared mem 更优
**目标**: warp shuffle 第一级 + shared mem 跨 warp
**结果**:
- <runtime_ratio>: 0.617640
- <precision>: True
**分析**: 两级 reduction 指令开销 > 单级
**决策**: **回退**

### 迭代 9: 代码优化格式
**假设**: 位移操作 tid<<2 比 tid*4 更高效
**目标**: 用位移和紧凑代码
**结果**:
- <runtime_ratio>: 0.613692
- <precision>: True
**分析**: 编译器已自动优化
**决策**: **回退**

## 最终结果
- 最终保留版本: 迭代 2 (float4 vectorized + float2 fused reduction, block=256)
- 最佳 commit: fd36ef3
- rejected variants: 迭代 1,3,4,5,6,7,8,9
- Final rerun:
  - <time_before_opt>: 0.050353 ms
  - <time_after_opt>: 0.030472 ms
  - <runtime_ratio>: 0.605166
  - <precision>: True
- 剩余风险: 小 kernel 测量噪声大，实际加速比在 1.6-1.8× 范围

---

## 第二轮优化 (2026-06-29)
开始时间: 2026-06-29

### 迭代 10: Row-level sqrsum sharing + per-thread output loop
**假设**: sqrsum 在 baseline 中被计算 24 次/row，通过一行一个 block 共享 sqrsum 避免冗余
**目标**: Grid=13 blocks (one per row)，Phase1 所有 256 线程协作计算 sqrsum，Phase2 串行计算 dot product
**参考依据**: trace-report 显示 sqrsum 冗余计算
**结果**:
- <runtime_ratio>: 6.136681
- <precision>: True
**分析**: Phase2 的串行 dot product (每个线程 5120 次迭代无并行) 导致严重性能回退
**决策**: **回退**

### 迭代 11: 2-way accumulator unrolling
**假设**: 双累加器交错可隐藏 FMA 流水线延迟
**目标**: 在 float4+float2 fused reduction 基础上增加 2-way accumulator
**结果**:
- <runtime_ratio>: 0.609105
- <precision>: True
**分析**: 比当前最佳版本差，额外寄存器压力抵消收益
**决策**: **回退**

### 迭代 12: float4 + block=128
**假设**: 更多 blocks (624 vs 312) 提供更好的 wave 级并行
**目标**: block=128 + float4 + float2 fused reduction
**结果**:
- <runtime_ratio>: 0.564284
- <precision>: True
**分析**: 接近最佳版本但稍差，block=256 的延迟隐藏更好
**决策**: **回退**

### 迭代 13: No scalar tail + inv_K
**假设**: K=5120 可被 4 整除，去除标量 tail 循环 + 用乘法替代除法
**目标**: 移除 scalar tail loop，预计算 inv_K
**结果**:
- <runtime_ratio>: 0.611287
- <precision>: True
**分析**: 编译器已优化 tail 分支，inv_K 单独使用收益有限
**决策**: **回退**

### 迭代 14: Warp-shuffle reduction (C500 warp=64)
**假设**: `__shfl_down_sync` 消除 shared memory reduction traffic，减少 barrier 开销
**目标**: Warp-level shuffle (64 lanes) + cross-warp shared memory (4 warps)
**参考依据**: C500 warp=64, hardware guide warp shuffle 建议
**结果**:
- **commit**: 690e1e6
- <runtime_ratio>: 0.551462
- <precision>: True
**分析**: **NEW BEST!** Warp shuffle 避免 shared memory 流量，仅 cross-warp 需要 1 个 barrier
**决策**: **保留为最佳版本**

### 迭代 15: Warp-shuffle + block=128
**假设**: block=128 (2 warps) + warp shuffle，更多 blocks
**目标**: 减少 cross-warp barrier 开销
**结果**:
- <runtime_ratio>: 0.646835
- <precision>: True
**分析**: block=128 的延迟隐藏不足，256 线程/4 warps 更优
**决策**: **回退**

### 迭代 16: Warp-shuffle + block=64 (single warp, no barrier)
**假设**: 单 warp (64 threads)，纯 warp shuffle 无需 __syncthreads
**目标**: 消除所有 barriers
**结果**:
- <runtime_ratio>: 0.691328
- <precision>: True
**分析**: 单 warp 延迟隐藏差，每线程 K 迭代数多 (5120/64=80 vs 5120/256=20)
**决策**: **回退**

### 迭代 17: 4-way accumulator unrolling
**假设**: 4 路独立累加器可更好隐藏 FMA 延迟
**目标**: 每次迭代处理 16 个元素 (4×float4)，4-way interleave
**结果**:
- <runtime_ratio>: 0.753462
- <precision>: True
**分析**: 寄存器压力过大，编译器调度更差
**决策**: **回退**

### 迭代 18: Warp-shuffle + inv_K precompute
**假设**: 在 warp-shuffle 最佳版本上加入 inv_K 预计算
**目标**: 用乘法替代除法 (sqrsum * inv_K)
**结果**:
- **commit**: 4ffc6d3
- <runtime_ratio>: 0.545845
- <precision>: True
**分析**: **NEW BEST!** inv_K 组合 warp-shuffle 进一步减少指令
**决策**: **保留为最佳版本**

### 迭代 19: Two-kernel approach (sqrsum precompute)
**假设**: 将 sqrsum 计算分离为独立 kernel，dot product kernel 更精简
**目标**: Kernel 1: sqrsum (13 blocks), Kernel 2: dot product (312 blocks)
**结果**:
- <runtime_ratio>: 0.657178
- <precision>: True
**分析**: 双 kernel launch overhead + global memory sqrsum 读取代价 > 收益
**决策**: **回退**

### 迭代 20: Hardcoded inv_5120
**假设**: 编译器可能更好优化编译时常量
**目标**: 硬编码 `1.0f/5120.0f` 替代运行时除法
**结果**:
- <runtime_ratio>: 0.547896
- <precision>: True
**分析**: 与迭代 18 等效，硬编码 shape 不可取
**决策**: **回退**，保持迭代 18 为最终版本

## 最终结果
- 最终保留版本: **迭代 18** (warp-shuffle reduction + float4 vectorization + inv_K precompute, block=256)
- 最佳 commit: 4ffc6d3
- 总优化轮次: 20 (迭代 1-20)
- rejected variants: 1,3-13,15-17,19-20
- Final rerun (500 iters):
  - <time_before_opt>: 0.040582 ms
  - <time_after_opt>: 0.027792 ms
  - <runtime_ratio>: 0.684852
  - <precision>: True
- 最佳单次: runtime_ratio=0.546 (迭代 18, 100 iters)

## 最终保留策略
1. **float4 向量化加载**: 每次加载 4 个 float，减少 75% load 指令
2. **warp-shuffle reduction**: `__shfl_down_sync` 替代 shared memory tree（64-lane warp）
3. **cross-warp 仅需 1 个 barrier**: 4 个 warp 通过 4 元素 shared memory 合并
4. **inv_K 预计算**: 用乘法替代除法
5. **block=256 (4 warps)**: 最优延迟隐藏与 barrier 开销平衡

## Torch vs MACA 性能对比
- Torch (einsum+rsqrt):    0.129956 ms
- MACAC ori (baseline):    0.040582 ms
- MACAC opt (best):        0.027792 ms
- **MACA speedup vs torch: 4.68×**
- MACAC improvement:       1.46× over baseline
- 剩余风险: 小 kernel 测量噪声 (±0.005ms)，实际加速比在 3-5× 范围
