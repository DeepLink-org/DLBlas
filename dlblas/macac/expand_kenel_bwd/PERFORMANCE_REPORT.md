# expand_kenel_bwd MACAC C500 性能报告

## 算子信息
| 项目 | 值 |
|------|-----|
| 算子名称 | expand_kenel_bwd |
| 算子族 | reduction |
| 操作 | sum reduction along dim=-2 |
| 输入形状 | (2, 1024, 4, 1280) |
| 输出形状 | (2, 1024, 1280) |
| 数据类型 | float32 |
| 布局 | contiguous |
| 容器 | metax_gemm_opt (MACA 3.3.0.15) |

## 性能结果

| 指标 | 时间 | 说明 |
|------|------|------|
| MACAC baseline (ori) | 0.076681 ms | 修复后的完整baseline |
| MACAC optimized (opt) | 0.057277 ms | Round 5 最佳版本 |
| **runtime_ratio** | **0.746950** | **25.3% 提升** |
| Torch sum(dim=-2) | 0.350608 ms | PyTorch 2.8.0+metax |
| MACAC vs Torch | **6.12x 加速** | 0.351 / 0.057 |

## 最佳实现策略
1. **float4 向量化加载**: 每线程处理4个输出元素，使用float4一次加载16字节
2. **`__ldg()` 只读缓存**: 通过纹理缓存路径加载输入数据
3. **`__launch_bounds__(512, 2)`**: 控制寄存器使用，平衡occupancy
4. **减法代替取模**: 减少整数除法指令
5. **手动循环展开**: 消除M=4的内层循环

## 优化历程

| Round | 策略 | ratio | 决策 |
|-------|------|-------|------|
| 1 | launch_bounds + unroll + sub-mod | 0.788 | 保留 |
| 2 | float4 向量化 | 0.754 | 保留 |
| 3 | float4 + bs=256 | 0.754 | 回退(无改善) |
| 4 | 8元素/线程 | 0.840 | 回退(恶化) |
| **5** | **+ __ldg() 只读缓存** | **0.748** | **✅ 最佳** |
| 6 | + DivModFast | 0.882 | 回退(恶化) |
| 7 | Block-per-row | 0.754 | 回退(稍差) |
| 8 | 无launch_bounds | 0.751 | 回退(稍差) |
| 9 | launch_bounds(512,4) | 0.772 | 回退(恶化) |

## 发现的关键Bug
原始baseline (`tmp_ori.cuh`) 存在严重bug：
- `row=blockIdx.x` 仅使用每个block的一个线程
- grid = ceil(total/256) = 10240 blocks，仅覆盖前10240个输出元素(0.39%)
- 修复: `row=blockIdx.x*blockDim.x+threadIdx.x`

## Profiling 发现
- mcTracer: 原始kernel mtreg_occupancy仅4%（寄存器压力严重）
- CycleTrace: 无法捕获硬件事件（kernel执行时间~0.06ms太短）
- mcProfiler: Roofline因kernel太小触发ZeroDivisionError

## 交付物
- `inc/tmp_use.cuh`: 最佳优化版内核
- `inc/tmp_ori.cuh`: 修复后的baseline
- `ITERATIONS.md`: 完整迭代记录
- `bench_torch.py`: Torch对比脚本
