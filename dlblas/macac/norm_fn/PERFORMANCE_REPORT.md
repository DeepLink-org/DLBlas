# norm_fn 性能报告

## 算子信息
- **算子**: norm_fn (RMS-normalized batched dot product)
- **语义**: `output[m,n] = sum_k(residual[m,k] * fn[n,k]) * rsqrt(sum_k(residual[m,k]^2)/K + eps)`
- **Family**: 自定义融合 (matmul-like + layernorm-like)
- **dtype**: float32
- **Shape**: M=13, N=24, K=5120 (total output: 312 elements)

## 优化策略
1. **float4 向量化加载**: 每次加载 4 个 float，减少 75% load 指令
2. **warp-shuffle reduction**: `__shfl_down_sync` 替代 shared memory tree reduction
3. **inv_K 预计算**: 用乘法 `sqrsum * (1.0f/K)` 替代除法
4. **block=256 (4 warps × 64 lanes)**: 最优延迟隐藏与 barrier 开销平衡

## 性能结果

### Final Rerun (500 iterations)
| Metric | Value |
|--------|-------|
| time_before_opt (baseline) | 0.040582 ms |
| time_after_opt (best) | 0.027792 ms |
| runtime_ratio | 0.684852 |
| precision | True |
| MACA improvement | 1.46× |

### Torch vs MACA Comparison
| Implementation | Time (ms) | Speedup vs torch |
|----------------|-----------|------------------|
| Torch (einsum+rsqrt) | 0.129956 | 1.00× |
| MACAC ori (baseline) | 0.040582 | 3.20× |
| **MACAC opt (best)** | **0.027792** | **4.68×** |

## 最佳 Commit
- **Commit**: 4ffc6d3
- **文件**: inc/tmp_use.cuh
- **策略**: warp-shuffle + float4 + inv_K

## Profile Analysis
- Bottleneck: compute (MTE=39.8%, vls_pipeline_stall=88.32%)
- Occupancy: 3.00% (low due to small grid 312 blocks)
- L2C hit: 83.87%
- NOP share: 17.35% (barrier overhead)
- Key optimization: warp shuffle reduced barrier overhead by eliminating shared memory tree reduction traffic

## 迭代历史
- Total: 20 iterations
- Best: Iteration 18 (warp-shuffle + float4 + inv_K)
- Rejected: 16 variants explored (block sizes 64/128/512, serial dot product, 2/4-way unrolling, 2-kernel, __ldg, etc.)
- Full log: ITERATIONS.md
