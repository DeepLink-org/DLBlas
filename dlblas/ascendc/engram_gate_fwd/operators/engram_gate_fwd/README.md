# engram_gate_fwd

Ascend C 实现 engram gate 前向融合算子。

## 功能

```
output = x + sigmoid(signed_sqrt(dot(RMSNorm(x, wh), RMSNorm(k, we)) * scalar)) * v
```

融合计算: RMSNorm(仅 rstd) + Weighted Dot Product + Signed Sqrt Gate + Sigmoid + Broadcast Output

## 输入输出

| 名称 | Shape | dtype | 说明 |
|------|-------|-------|------|
| hidden_states | [num_tokens, hc_mult, hidden_size] | bf16 | 输入特征 |
| k | [num_tokens, hc_mult, hidden_size] | bf16 | Key 嵌入 |
| v | [num_tokens, hidden_size] | bf16 | Value 嵌入 |
| weight_hidden | [hc_mult, hidden_size] | bf16 | RMSNorm 权重 |
| weight_embed | [hc_mult, hidden_size] | bf16 | k RMSNorm 权重 |
| clamp_value | scalar | float | Signed sqrt 下界 |
| eps | scalar | float | 数值稳定常数 |

| 名称 | Shape | dtype | 说明 |
|------|-------|-------|------|
| output | [num_tokens, hc_mult, hidden_size] | bf16 | Gate 加权输出 |
| raw_dot | [num_tokens, hc_mult] | fp32 | 未归一化点积 |
| gate_score | [num_tokens, hc_mult] | fp32 | Gate 值 |
| rstd_x | [num_tokens, hc_mult] | fp32 | x RMSNorm rstd |
| rstd_k | [num_tokens, hc_mult] | fp32 | k RMSNorm rstd |

## 编译运行

```bash
# 完整流程 (编译 + 数据生成 + 运行 + 验证)
bash run.sh

# 跳过编译 (复用已有产物)
bash run.sh --skip-build

# PyTorch 通路测试
bash run.sh --torch
```

## 测试

### 测试覆盖 (Level 0-2)

| Level | 用例 | 结果 |
|-------|------|------|
| Level 0 (基础) | hidden_size=16, 256 | 全部 PASS |
| Level 1 (典型) | hidden_size=1024, 4096 | 全部 PASS |
| Level 2 (边界) | single_token, single_hc, 512, 4097 (非对齐) | 全部 PASS |
| Level 2 (溢出保护) | hidden_size=8192 | 正确拒绝 (UB > 192KB) |

### 精度

所有 fp32 中间输出 (raw_dot, gate_score, rstd_x, rstd_k) 精度优于 1e-5 (绝对值), bf16 输出在量化精度范围内 (max abs err < 8e-3, max rel < 8e-3)。

### 精度标准

| 输出类型 | MERE 阈值 | MARE 阈值 | 实际值 | 达标 |
|---------|----------|----------|--------|------|
| bf16 output | < 0.00781 | < 0.0781 | 2.4e-04 / 1.5e-02 | PASS |
| fp32 中间量 | < 0.000122 | < 0.00122 | < 1e-04 / < 5e-5 | PASS |

### 非对齐 hidden_size 支持

本实现通过显式 padding 清零策略支持非 32B 对齐的 hidden_size（如 4097）：
- DataCopyPad 仅拷贝有效数据字节数，UB buffer 大小仍按 32B 对齐
- 每次 Cast 后将 fp32 buffer 的 padding 区域显式清零，确保 Mul/ReduceSum 不受垃圾数据影响

## 架构

- 芯片: Ascend 910B2
- NpuArch: DAV_2201
- CANN: 9.0.0
- 编程路线: SIMD/MemBase
- 归约模式: AR-FullLoad
- 权重策略: 懒加载 (逐 head 加载单行)

## 文件结构

```
├── CMakeLists.txt          # 双 target: 可执行文件 + libengram_gate_fwd_ops.so
├── op_kernel/
│   ├── engram_gate_fwd_tiling.h    # TilingData 结构 + ComputeTiling + ComputeUBUsage
│   └── engram_gate_fwd_kernel.asc  # Kernel 实现
├── op_host/
│   ├── engram_gate_fwd.asc         # Host + main 入口
│   └── data_utils.h                # 文件读写工具
├── op_extension/                   # PyTorch 接入 (TORCH_LIBRARY)
│   ├── engram_gate_fwd_torch.cpp
│   ├── register.cpp
│   └── ops.h
├── scripts/
│   ├── gen_data.py                 # 测试数据生成 (多 Level 支持)
│   ├── golden.py                   # 参考实现 (fp32 golden)
│   ├── verify_result.py            # 直调精度验证
│   └── test_torch.py               # PyTorch 通路测试
├── docs/
│   ├── DESIGN.md                   # 设计文档
│   ├── PLAN.md                     # 开发计划与结果
│   ├── REVIEW.md                   # 审查报告 (Round 0)
│   └── perf/round_003/             # msprof op 性能数据 (2026-07-02 重建)
└── run.sh
```

## 性能 (Round 3 - 2026-07-02 重建验证)

`msprof op` 数据归档在 `docs/perf/round_003/`:

| 指标 | 值 |
|------|-----|
| Task Duration | 16.42 us (32x4x4096, 48 cores, 32 active) |
| aiv_vec (vector compute) | 6.06 us (45.3%) |
| aiv_scalar (scalar pipe) | 4.72 us (35.3%) |
| aiv_mte2 (memory read) | 2.45 us (18.3%) |
| aiv_mte3 (memory write) | 2.71 us (20.3%) |

**msprof Performance Summary**:
- MTE2/MTE3 bandwidth utilization < 80% when active (single buffer, no DMA/compute overlap)
- aivector compute usage < 20% (across all 48 cores; active cores achieve 45% vec ratio)
- Roofline: latency bound (compute caused)

**瓶颈**: Scalar pipe (35.3%, Gate 标量计算) + 高等待开销 (单缓冲 Pipeline stall)

## 已知限制

1. **hidden_size 上限**: 当前最大 ~6800 (UB 192KB 约束)。hidden_size=8192 需约 200KB > 192KB，Host 侧正确拒绝执行。
2. **单缓冲**: 无双缓冲流水线，DMA 和计算无法重叠，导致 Pipeline stall 占比高。
3. **标量 Gate**: SignedSqrt 和 Sigmoid 使用标量路径 (1-element Sqrt/Exp API)，scalar pipe 占比 38.7%。
4. **v 重复加载**: 同 token 内每个 head 都重新加载 v[t, :]，应移出 hc 内层循环。
5. **非对齐 hidden_size 性能**: 非对齐尺寸需要 padding 清零开销 (每行最多 7 次 SetValue)。
