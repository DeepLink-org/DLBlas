# head_compute_mix_bwd 性能报告

## 算子信息
- **名称**: head_compute_mix_bwd
- **语义**: sigmoid backward pass (`mhc_head_compute_mix` 的反向)
  - grad_z = grad_out * sigmoid(z) * (1 - sigmoid(z))
  - grad_input_mix = grad_z * mhc_scale
  - grad_mhc_base = sum(grad_z, dim=(0,1))
  - grad_mhc_scale = sum(grad_z * input_mix)
- **Family**: elementwise + reduction (mixed)
- **dtype**: float32
- **Shape**: batch0=2, batch1=1024, mhc_mult=4 (total 8192 elements)
- **Target**: MetaX C500 (MACA)

## 优化历程

| 轮次 | 策略 | opt时间(ms) | vs baseline | 决策 |
|------|------|------------|-------------|------|
| R0 | baseline (block=512, shared mem 5ch) | 0.054717 | — | baseline |
| 1 | block_size=256 | 0.058742 | -7.4% | 回退 |
| 2 | 寄存器累加base | 0.055299 | -1.1% | 回退 |
| 3 | warp shuffle替代shared mem | 0.052936 | +3.3% | 保留 |
| 4 | float4向量化 | crash | — | 回退 |
| 5 | block-level base reduce | crash | — | 回退 |
| 6 | fast sigmoid | 0.054766 | -0.1% | 回退 |
| 7 | 4elem/loop branchless | 0.054223 | +0.9% | 回退 |
| 8 | **block=128 + warp shuffle** | **0.051551** | **+5.8%** | ✅ BEST |
| 9 | block=64 single warp | 0.053655 | +1.9% | 回退 |

## 最终结果

### 正确性
- `<precision>True</precision>` ✅

### 性能标签
```
<time_before_opt>0.064535 ms</time_before_opt>
<time_after_opt>0.051551 ms</time_after_opt>
<runtime_ratio>0.798802</runtime_ratio>
<precision>True</precision>
```

### MACA vs PyTorch 对比
| 后端 | 时间 (ms) | 加速比 |
|------|-----------|--------|
| MACA (macac optimized) | 0.051551 | **1.92x** |
| PyTorch (torch cuda) | 0.098781 | 1.00x |

### 最终策略
- **block_size**: 128 (64 blocks, good SM coverage)
- **Scale reduction**: warp shuffle (__shfl_xor_sync)
- **Cross-warp reduction**: 2 slots shared memory
- **Base reduction**: per-thread register accum + atomicAdd
- **Shared memory**: 8 bytes/block (minimal)

### 关键瓶颈 (from trace profiling)
- effective_occupancy: 2.00% (small problem size limits parallelism)
- WSM stall: 52% (shared memory access latency)
- NOP: 21.75% (pipeline bubbles)
- L2C hit rate: 13.88%

### 剩余风险
- 小shape (8192 elems) 测试结果，大shape表现待验证
- warp shuffle 依赖 64-lane warp (C500特性)
- atomicAdd 竞争在大shape下可能成为瓶颈
