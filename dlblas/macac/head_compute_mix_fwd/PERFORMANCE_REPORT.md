# head_compute_mix_fwd 性能报告

## 算子信息
- **算子名称**: head_compute_mix_fwd
- **语义**: output = sigmoid(input_mix * mhc_scale + mhc_base) + mhc_pre_eps
- **数据类型**: float32
- **输入 shape**: input_mix=[16, 16384, 4], mhc_scale=[1], mhc_base=[4], mhc_pre_eps=scalar(0.01)
- **输出 shape**: [16, 16384, 4] (共 1,048,576 元素)
- **算子 family**: elementwise

## 性能对比

### MACA (MetaX C500)
| 版本 | 耗时 (ms) | 相对基线 |
|------|-----------|---------|
| Baseline (naive scalar) | 0.032542 | 1.00x |
| Optimized (float4 + block=256) | 0.019766 | 0.607x (1.65x 加速) |

### Torch (MetaX C500)
| 版本 | 耗时 (ms) | vs MACA optimized |
|------|-----------|-------------------|
| Torch sigmoid | 0.038323 | 1.94x slower |

### 总结
- MACA 优化版本比 MACA 基线快 **1.65x**
- MACA 优化版本比 Torch 快 **1.94x**
- 主要优化: **float4 向量化** (MHC=4 天然对齐) + **block_size=256**

## Trace Profile 对比

| 指标 | Baseline | Optimized | 变化 |
|------|----------|-----------|------|
| Kernel span (cycles) | 27,789 | 13,066 | -53.0% |
| MTE share | 56.52% | 68.10% | +11.6pp |
| AP MTE duty | 53.52% | 59.28% | +5.8pp |
| HBM bandwidth usage | 17.78% | 30.89% | +13.1pp |
| L2C hit rate | 21.72% | 3.23% | -18.5pp |
| Real IPC | 59.60 | ~65 | +9% |
| Grid/Block | [2048,512] | [1024,256] | 更小block更多并行 |
| Instruction count | 4,857 | ~2,000 | -58.8% (vectorization) |

### 分析
- float4 向量化将指令数减少约 59%，MHC=4 天然匹配 float4 宽度
- block_size=256 比 512 提供更好的 wave 级并行度
- HBM 带宽利用率提升 74%，更充分利用内存带宽
- vls_pipeline_stall 仍是主要 ISU stall (88%→81%)，说明 memory latency 是最终瓶颈

## 最终优化策略
- ✅ float4 向量化加载/存储
- ✅ block_size=256
- ✅ __ldg() 用于只读大数组
- ❌ grid-stride loop (无益)
- ❌ capped grid (无益)
- ❌ scalar preloading (无益)

## 工作区文件
- 最佳 kernel: inc/tmp_use.cuh
- 迭代记录: ITERATIONS.md
- Baseline trace: profile-artifacts/head_compute_mix_fwd_v0_baseline/
- Optimized trace: profile-artifacts/head_compute_mix_fwd_v1_opt_float4_b256/
