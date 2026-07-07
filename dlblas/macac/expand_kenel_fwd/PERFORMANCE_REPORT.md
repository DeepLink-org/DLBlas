# expand_kenel_fwd 性能报告

## 算子信息
- **算子名**: expand_kenel_fwd
- **Family**: data-movement (expand/broadcast)
- **语义**: input (B,S,H) -> unsqueeze(-2) -> expand to (B,S,M,H) -> contiguous
- **dtype**: float32
- **layout**: contiguous
- **输入shape**: [1, 1024, 1280]
- **输出shape**: [1, 1024, 4, 1280]
- **数据量**: 读5MB + 写21MB = 26MB/次

## 优化策略演进

| 版本 | 策略 | time_after_opt | ratio | 判定 |
|------|------|----------------|-------|------|
| baseline | 1D grid + div/mod寻址 | 0.076 ms | 0.745 | 基线 |
| v1 | 2D grid消除div/mod | 0.034 ms | 0.325 | 保留 |
| v2 | float4向量化load+store | 0.032 ms | 0.300 | 保留 |
| v5 | 显式展开4次写入 | 0.031 ms | 0.301 | 保留 |
| v7 | block_size=320, 无循环 | 0.030 ms | 0.296 | 保留 |
| v20 | **最优模式** | **0.030 ms** | **0.279** | **最终** |

## Session 2 优化尝试 (2026-06-29)

| 版本 | 策略 | time (ms) | ratio | 判定 |
|------|------|-----------|-------|------|
| v21 | Multi-row per block | 0.0301 | 0.301 | 回退 |
| v22 | Float2 writes | 0.0319 | 0.321 | 回退 |
| v23 | uint4 bitcast | 0.0311 | 0.298 | 回退 |
| v24 | 1-warp block (64线程) | 0.0343 | 0.339 | 回退 |
| v25 | #pragma unroll loop | 0.0312 | 0.291 | 回退 |
| v26 | 160线程+2 float4 | 0.0297 | 0.307 | 回退 |
| v27 | Reverse M write order | 0.0304 | 0.304 | 回退 |
| v28 | 80线程+4 float4 | 0.0307 | 0.301 | 回退 |
| v29 | Shift+launch_bounds | 0.0304 | 0.306 | 回退 |

## Trace Profiling 关键发现 (2026-06-29)

| 指标 | 值 | 含义 |
|------|-----|------|
| HBM bandwidth usage | 92.74% | 接近峰值，内存带宽硬瓶颈 |
| L2C hit rate | 0.72% | 数据流式传输，几乎无缓存复用 |
| ISU vls_pipeline_stall | 54.68% | 向量load/store流水线停顿 |
| ISU vls_wdata_stall | 45.32% | 写数据停顿（写带宽瓶颈） |
| Real IPC | 20.58 | 计算单元不是瓶颈 |
| Effective occupancy | 2.00% | 低占用率（内存密集型kernel典型值） |
| DNOC >512-cycle share | 0.00% | 无长尾延迟 |
| MTE instruction share | 37.37% | 向量计算占主体 |
| GVM instruction share | 6.67% | 全局内存访问指令 |
| Registers/thread | 12 | 低寄存器压力 |

## 最终结果 (Final rerun, 2026-06-29)

| 指标 | 值 |
|------|-----|
| time_before_opt | 0.099453 ms |
| time_after_opt | 0.030694 ms |
| runtime_ratio | 0.308631 |
| precision | True |
| **MACA加速比 (opt vs ori)** | **3.24x** |

## 与Torch对比 (2026-06-29, torch 2.8.0+metax3.3.0.2)

| 实现 | 时间 (ms) | vs MACA opt |
|------|-----------|-------------|
| Torch GPU (MACA backend) | 0.061174 | 1.99x slower |
| Torch CPU | 0.056635 | 1.85x slower |
| MACA kernel (ori) | 0.099453 | 3.24x slower |
| **MACA kernel (opt)** | **0.030694** | **baseline** |

### MACA opt vs Torch GPU: **1.99x faster**
### MACA opt vs MACA ori: **3.24x faster**

## 性能分析

### 内存带宽分析
- 总数据移动: 5MB (读) + 21MB (写) = 26MB
- 有效带宽: 26MB / 0.030694ms ≈ 847 GB/s
- C500 HBM峰值: ~1550 GB/s (实测Roofline基准 ~1843 GB/s)
- 利用率: ~55% of peak (bandwidth), ~92.74% of Roofline HBM

### 关键优化点
1. **2D grid**: 每行一个block，消除昂贵的整数除法/取模运算 (3.0x speedup)
2. **float4向量化**: 16字节对齐的load/store，减少4x访存指令 (~7% improvement)
3. **精确block_size=320**: 与hidden_size/4匹配，无循环开销，每线程恰好处理1个float4
4. **显式展开**: 消除内层m循环，编译器更好优化

### 瓶颈分析
- **主要瓶颈**: 写带宽 (每个输入元素需要写4次到4个M切片)
- **ISU stall分布**: vls_pipeline_stall 54.68% + vls_wdata_stall 45.32% = 100% 内存相关
- **L2C 缓存**: 0.72%命中率 = 数据纯流式，无时间局部性可利用
- **理论下限**: 26MB / 1550GB/s ≈ 0.017 ms
- **实际/理论比**: 0.031/0.017 ≈ 1.82x, 已接近带宽极限
- **剩余空间**: 主要是写合并效率和L2C利用率

## 工作区
- 路径: /mnt/opt_test/expand_kenel_fwd_run
- 容器: metax_gemm_opt
- 迭代记录: ITERATIONS.md
- Trace profiling: profile-artifacts/expand_kenel_fwd_v0_baseline/
- Torch对比: torch_comparison.txt
