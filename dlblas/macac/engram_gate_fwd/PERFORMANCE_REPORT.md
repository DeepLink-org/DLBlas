# engram_gate_fwd — MACAC vs Torch Performance Report

## 算子信息
- **算子名称**: engram_gate_fwd
- **算子语义**: `output = x + sigmoid(signed_sqrt(dot(RMSNorm(x, wh), RMSNorm(k, we)) * scalar)) * v`
- **数据类型**: bfloat16
- **Shape**: T=4096, M=4, D=4096 (num_tokens=4096, hc_mult=4, hidden_size=4096)
- **容器**: metax_gemm_opt
- **PyTorch 版本**: 2.8.0+metax3.3.0.2

## 性能结果

| 实现 | 耗时 (ms) | vs Torch | vs MACAC Baseline |
|---|---:|---:|---:|
| Torch (PyTorch 2.8.0) | 13.728 | 1.00x | - |
| MACAC Baseline (tmp_ori) | 0.758 | 18.11x | 1.00x |
| MACAC Optimized (tmp_use) | **0.481** | **28.55x** | **1.58x** |

## 优化策略

最终最佳版本的优化策略：
1. **blockDim=64**: 减少 block size 以降低 shared memory/block (768 bytes)，提高 SM 利用率
2. **8x 循环展开**: 将 64 次迭代减少到 8 次，大幅减少循环开销
3. **融合权重预计算**: 将 `x * wh * k * we` 优化为 `x * k * (wh * we)`，每元素减少 1 次乘法
4. **warp-synchronous 执行**: blockDim=64=warpSize，避免不必要的 barrier

## 性能分析 (Trace Report)

### Baseline (v0) 瓶颈
- Bound type: **compute**
- MTE instruction share: 54.12%
- ISU stall (vls_pipeline): 79.97%
- L2C hit rate: 39.64%
- HBM usage: 42.65%
- 关键问题: 指令开销大 + 内存延迟导致 pipeline stall

### Optimized (v1) 瓶颈
- Bound type: **memory** (shifted from compute)
- MTE instruction share: 66.18% (更高，因其他指令减少)
- Total instructions: -42% (235K → 136K)
- HBM usage: 68.53% (+61% bandwidth)
- Achieved bandwidth: 1,263 GB/s
- 关键问题: 内存带宽接近饱和(68.5%), L2C hit rate 偏低(~39%)

### 优化的关键效果
- 指令数减少 42%: 235,327 → 136,242
- HBM 带宽提升 61%: 786 → 1,263 GB/s
- 瓶颈从 compute 转移到 memory
- 寄存器压力增加: 24 → 60 regs/thread (unroll 的代价)

## 最终验证
```
Final rerun: bash run.sh 10 100 0
<time_before_opt>: 0.758090 ms
<time_after_opt>: 0.480973 ms
<runtime_ratio>: 0.634453
<precision>: True
```

## 性能分析证据

Trace profile artifacts:
- Baseline: `profile-artifacts/engram_gate_fwd_v0_baseline/`
- Optimized: `profile-artifacts/engram_gate_fwd_v1_opt8x/`

Key metrics comparison:
| Metric | Baseline | Optimized | Δ |
|---|---:|---:|---|
| Kernel span | 442.9M cycles | 421.0M cycles | -5% |
| Instructions | 235,327 | 136,242 | -42% |
| Registers/thread | 24 | 60 | +150% |
| HBM bandwidth | 786 GB/s | 1,263 GB/s | +61% |
| Achieved FLOPS | 5.02 TF | 5.80 TF | +16% |
| L2C hit rate | 39.64% | 38.98% | -1.7% |
| AP MTE duty | 65.44% | 73.66% | +12.6% |
| DPC imbalance | 7.69% | 7.68% | 0% |
