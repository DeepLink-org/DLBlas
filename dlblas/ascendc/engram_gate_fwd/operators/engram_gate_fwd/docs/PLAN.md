# engram_gate_fwd AscendC 算子开发计划 (PLAN.md)

> 设计文档: [DESIGN.md](./DESIGN.md)
> 最后更新: 2026-07-02

---

## 1. 需求概述

### 1.1 算子功能

全融合前向算子：RMSNorm(rstd) + Dot Product + SignedSqrt Gate + Sigmoid + Broadcast Gated Addition。

### 1.2 输入输出

```
输入 (7):
  hidden_states: [num_tokens, hc_mult, hidden_size], bf16
  k:             [num_tokens, hc_mult, hidden_size], bf16
  v:             [num_tokens, hidden_size], bf16
  weight_hidden: [hc_mult, hidden_size], bf16
  weight_embed:  [hc_mult, hidden_size], bf16
  clamp_value:   scalar, float
  eps:           scalar, float

输出 (5):
  output:     [num_tokens, hc_mult, hidden_size], bf16
  raw_dot:    [num_tokens, hc_mult], fp32
  gate_score: [num_tokens, hc_mult], fp32
  rstd_x:     [num_tokens, hc_mult], fp32
  rstd_k:     [num_tokens, hc_mult], fp32
```

### 1.3 硬件目标

| 参数 | 值 |
|------|-----|
| 芯片 | Ascend 910B2 |
| NpuArch | DAV_2201 |
| CANN | 9.0.0 |
| UB | 192KB |
| AI Core | 24 |
| 编程路线 | SIMD/MemBase |

### 1.4 约束

| # | 约束 | 说明 |
|---|------|------|
| C1 | 禁止 Host 侧预处理输入 tensor | 如转置、reshape 等 |
| C2 | 全融合单 kernel | 不拆分为多个子算子 |
| C3 | bf16 输入, fp32 中间计算, bf16 输出 | 混合精度管线 |

---

## 2. 算子拆分方式

### 2.1 决策

**全融合单 Kernel，不拆分**。

计算流水线（RMSNorm + Dot Product + Gate + Broadcast）的所有阶段在单一 kernel 中完成。各阶段之间在 UB 内部通过 fp32 buffer 复用传递中间结果。

### 2.2 不拆分的理由

1. **数据耦合度高**：x 和 k 数据被 RMSNorm 和 Dot Product 两个阶段共用，拆分会导致重复加载
2. **中间结果数据量小**：rstd_x/rstd_k/raw_dot/gate_score 仅为每行 4 个 float，但拆分后的 kernel launch 开销大
3. **UB 容量充足**：单行 fp32 数据 16KB << 192KB，全行载入无压力

---

## 3. 向量化策略

### 3.1 总体路线

所有逐元素运算和归约运算均使用 **AscendC Level 2 Vector API**，硬件自动向量化。无手工 SIMD intrinsic。

### 3.2 各阶段向量化详情

| 计算阶段 | 向量化操作 | API |
|---------|-----------|-----|
| 类型提升 (bf16->fp32) | 256 元素/指令，全行并行 | `Cast<float, bf16_t>` |
| RMSNorm 平方 | 逐元素并行 | `Mul` (in-place) |
| RMSNorm 归约 | 树形归约，硬件加速 | `ReduceSum` |
| Dot Product 逐元素乘 | 全行并行 | `Mul` (x*wh, k*we) |
| Dot Product 归约 | 树形归约 | `ReduceSum` |
| Gate 标量计算 | count=1 向量操作 | `Sqrt` + `Exp` |
| Gated Addition | 全行并行 | `Muls` (v*gate) + `Add` |
| 类型转换 (fp32->bf16) | 全行并行 | `Cast<bf16_t, float>` |

### 3.3 32B 对齐保证

所有 UB buffer 的 InitBuffer 大小按 32B 对齐分配，确保 Vector API 在非对齐 hidden_size 下仍可安全执行。padding 区域在 Cast 后显式清零，排除对 ReduceSum/Mul 精度的影响。

---

## 4. Tiling 方案

### 4.1 多核切分

- 切分维度：`total_rows = num_tokens * hc_mult`
- 切分粒度：`tile_rows_per_core = ceil(total_rows / core_num)`，对齐到 `hc_mult` 边界
- 边界处理：尾 core `row_end = min(row_start + tile_rows_per_core, total_rows)`；空核提前退出

### 4.2 UB 切分

- 单行全载入（AR-FullLoad）
- 单缓冲模式（无 Double Buffer）
- Host 侧 `ComputeUBUsage` 函数验证容量：超过 UB_CAPACITY_DAV_2201 (192KB) 拒绝执行

### 4.3 支持的 hidden_size 范围

| hidden_size | 策略 |
|-------------|------|
| <= 6800 | AR-FullLoad |
| > 6800 | 当前拒绝（未来 AR-ColSplit） |

---

## 5. 内存管理

### 5.1 UB Buffer 布局

| Queue | 类型 | 大小(字节) | 用途 |
|-------|------|----------|------|
| weight_hidden_q_ | VECIN | hbb_align | weight_hidden 当前行 |
| weight_embed_q_ | VECIN | hbb_align | weight_embed 当前行 |
| v_q_ | VECIN | hbb_align | v 当前 token 行 |
| x_q_ | VECIN | hbb_align | hidden_states 当前行 |
| k_q_ | VECIN | hbb_align | k 当前行 |
| out_q_ | VECOUT | hbb_align | 输出行 |
| buf_a_q_ | VECIN | hfb_align | 主计算工作区 |
| buf_b_q_ | VECIN | hfb_align | 辅助计算工作区 |
| buf_c_q_ | VECIN | hfb_align | v 广播工作区 |
| tmp_q_ | VECIN | 8192 | ReduceSum 临时 buffer |
| scalar_q_ | VECIN | 32 | 标量写入中转 |

### 5.2 Weight 加载策略

逐 head 懒加载（单行），不持久化。避免占用 `hc_mult * hidden_size * 2 * 2` 的 UB 空间。

### 5.3 v 加载策略

当前：per-head 加载（内层循环）。优化方向：外提至 token 循环外层，同 token 复用。

### 5.4 GM 管理

- 所有输入/输出均位于 GM
- DataCopyPad 处理 GM-UB 数据传输
- 无手动 GM 分配/释放（由框架管理）

---

## 6. 计算精度

### 6.1 精度标准

| 输出 dtype | MERE 阈值 | MARE 阈值 |
|-----------|----------|----------|
| bf16 | < 2^-7 ≈ 0.00781 | < 10 * 2^-7 ≈ 0.0781 |
| fp32 | < 2^-13 ≈ 0.000122 | < 10 * 2^-13 ≈ 0.00122 |

### 6.2 精度管线

```
bf16 GM_READ → fp32 UB (CAST_NONE)
  → fp32 Compute (Mul/Add/ReduceSum/Sqrt/Exp)
  → bf16 UB (CAST_ROUND)
  → bf16 GM_WRITE
```

### 6.3 数值稳定性

| 风险 | 缓解 |
|------|------|
| ReduceSum 累加误差 | fp32 精度，4096 元素相对误差 < 6.4e-6 |
| Rsqrt 零值 | eps = 1e-20 |
| Sqrt(0) | clamp_value = 1e-6 |
| Padding 垃圾数据 | Cast 后显式清零 |

---

## 7. 测试计划

### 7.1 测试用例

| 编号 | 用例名称 | num_tokens | hc_mult | hidden_size | 测试重点 | 结果 |
|------|---------|-----------|---------|-------------|---------|------|
| TC0 | L0_small_basic | 2 | 2 | 16 | Level 0 基础功能 | PASS |
| TC1 | L0_small_hs256 | 2 | 2 | 256 | Level 0 基础功能 | PASS |
| TC2 | L1_typical | 32 | 4 | 1024 | Level 1 典型场景 | PASS |
| TC3 | L1_large | 32 | 4 | 4096 | 核心场景基准 | PASS |
| TC4 | L2_single_token | 1 | 4 | 4096 | num_tokens=1 边界 | PASS |
| TC5 | L2_single_hc | 8 | 1 | 4096 | hc_mult=1 边界 | PASS |
| TC6 | L2_small_hs | 8 | 4 | 512 | 小列数 | PASS |
| TC7 | L2_unaligned | 4 | 4 | 4097 | 非 32B 对齐 | PASS |
| TC8 | L2_large_hs | 4 | 4 | 8192 | UB 溢出保护 | 正确拒绝 |

### 7.2 测试覆盖矩阵

| 维度 | 覆盖值 |
|------|--------|
| num_tokens | 1, 2, 4, 8, 32 |
| hc_mult | 1, 2, 4 |
| hidden_size | 16, 256, 512, 1024, 4096, 4097, 8192 |
| 对齐 | 32B 对齐 (4096) + 非对齐 (4097) |
| 精度验证 | MERE/MARE vs fp32 golden |
| 溢出保护 | hidden_size=8192 应触发 UB 检查拒绝 |

### 7.3 精度测试详情

| 输出 | hidden_size=4096 (对齐) | hidden_size=4097 (非对齐) |
|------|------------------------|--------------------------|
| output (bf16) | MAE=2.44e-04, MRE=1.51e-02 | MAE=7.81e-03, MRE=7.69e-03 |
| raw_dot (fp32) | MAE=3.81e-05, MRE=4.52e-05 | MAE=2.48e-05, MRE=6.28e-06 |
| gate_score (fp32) | MAE=4.77e-07, MRE=9.16e-07 | MAE=1.19e-07, MRE=2.70e-07 |
| rstd_x (fp32) | MAE=2.38e-07, MRE=2.41e-07 | MAE=1.19e-07, MRE=1.21e-07 |
| rstd_k (fp32) | MAE=1.19e-07, MRE=1.22e-07 | MAE=1.19e-07, MRE=1.21e-07 |

全部精度指标在 fp32/bf16 社区标准内。

---

## 8. 开发阶段与检查项

### 8.1 阶段划分

| 阶段 | 内容 | 状态 | 检查项 |
|------|------|------|--------|
| 1 | 工程搭建 | 完成 | CMake 配置、AscendC 环境、direct-invoke 模板 |
| 2 | Tiling 实现 | 完成 | TilingData 结构、ComputeTiling、ComputeUBUsage、UB 容量检查 |
| 3 | Kernel 实现 | 完成 | 全融合计算管线、bf16/fp32 混合精度、非对齐处理、5 路输出 |
| 4 | 编译验证 | 完成 | DAV_2201 编译无错误/无警告 |
| 5 | Level 0 验证 | 完成 | 小 shape (16, 256) 功能正确 |
| 6 | Level 1 验证 | 完成 | 典型 shape (1024, 4096) 功能+精度 |
| 7 | Level 2 验证 | 完成 | 边界条件 (num_tokens=1, hc_mult=1, 非对齐, 大 hidden_size) |
| 8 | 性能采集 | 完成 | msprof 数据采集 |
| 9 | Double Buffer | 待开发 | x/k/out 双缓冲流水线 |
| 10 | v-load hoisting | 待开发 | v 加载外提至 token 外循环 |
| 11 | AR-ColSplit | 待开发 | hidden_size > 6800 分载路径 |

### 8.2 检查清单

- [x] Kernel 入口函数使用 `__global__ __vector__` 属性
- [x] 使用 `AscendC::TPipe` + `TQue` 模式
- [x] AllocTensor / FreeTensor 正确配对
- [x] ReduceSum 使用正确的 tmpBuf 大小 (8192 B)
- [x] DataCopyPad blockLen 使用有效数据字节数
- [x] 非对齐 hidden_size 的 padding 区域清零
- [x] 多核切分对齐到 token 边界
- [x] 空核提前退出检查
- [x] UB 容量运行时检查
- [x] 无硬编码 blockDim/blockIdx
- [ ] Double Buffer 流水线优化
- [ ] v-load hoisting 优化
- [ ] AR-ColSplit 大 hidden_size 支持

---

## 9. 已知限制

| # | 限制 | 说明 | 计划 |
|---|------|------|------|
| 1 | hidden_size 上限 ~6800 | 受 UB 192KB 约束，hidden_size=8192 会溢出 | AR-ColSplit (阶段 11) |
| 2 | 单缓冲 | CopyIn / Compute / CopyOut 串行，流水线 stall ~59% | Double Buffer (阶段 9) |
| 3 | v 重复加载 | 同 token 内每个 head 重新加载 v[t,:] | v-load hoisting (阶段 10) |

---

## 10. 性能基线

### 10.1 测试条件

- Shape: num_tokens=32, hc_mult=4, hidden_size=4096
- Block Dim: 48
- 频率: 1800 MHz
- 工具: msprof op

### 10.2 核心指标 (Round 3 - 重建验证)

| 指标 | 值 | 占比 |
|------|-----|------|
| Task Duration | 16.42 us | — |
| aiv_vec | 6.06 us | 45.3% |
| aiv_scalar | ~4.72 us | 35.3% |
| aiv_mte2 | ~2.45 us | 18.3% |
| aiv_mte3 | ~2.71 us | 20.3% |
| Block Dim | 48 | — |
| Active Cores | 32 | 16 idle early-exit |
| Freq | 1800 MHz | rated

### 10.3 优化目标

| 优化 | 当前 | 目标 |
|------|------|------|
| Pipeline stall | ~59% | < 30% (Double Buffer 后) |
| Task Duration | 16.52 us | < 12 us (综合优化后) |

---

## 11. 文件清单

| 文件 | 说明 |
|------|------|
| `op_kernel/engram_gate_fwd_kernel.asc` | Device 侧 Kernel 实现（全融合，含 padding 清零） |
| `op_kernel/engram_gate_fwd_tiling.h` | TilingData + ComputeTiling + ComputeUBUsage |
| `op_host/engram_gate_fwd.asc` | Host 侧入口 + main + UB 容量验证 |
| `op_host/data_utils.h` | 文件读写工具 |
| `op_extension/engram_gate_fwd_torch.cpp` | PyTorch 接入层 |
| `op_extension/register.cpp` | TORCH_LIBRARY 注册 |
| `op_extension/ops.h` | 函数声明 |
| `scripts/gen_data.py` | 多级测试数据生成 (Level 0/1/2) |
| `scripts/golden.py` | fp32 参考实现 + bf16 转换 |
| `scripts/verify_result.py` | 精度验证脚本 |
| `scripts/test_torch.py` | PyTorch 通路测试 |
| `CMakeLists.txt` | 双 target: 可执行文件 + .so |
| `run.sh` | 一键编译运行 |
| `docs/DESIGN.md` | 技术设计文档 |
| `docs/PLAN.md` | 开发计划（本文档） |
| `docs/perf/round_001/` | 首轮性能采集数据 |
| `docs/perf/round_002/` | Round 2 性能采集数据 (msprof) |
| `docs/precision/summary.txt` | 精度验证汇总 |

---

## 12. 参考资源

| 资源 | 路径/来源 |
|------|----------|
| Reduce API 头文件 | `kernel_operator_vec_reduce_intf.h` |
| Unary API 头文件 | `kernel_operator_vec_unary_intf.h` |
| Binary API 头文件 | `kernel_operator_vec_binary_intf.h` |
| Vconv API 头文件 | `kernel_operator_vec_vconv_intf.h` |
| DataCopy API 头文件 | `kernel_operator_data_copy_intf.h` |
| AR-FullLoad 设计模式 | `ascendc-tiling-design/references/reduction/` |
| 精度转换最佳实践 | `ascendc-api-best-practices/references/api-precision.md` |
| 精度验证标准 | `ops-precision-standard/reference/float_compute_community.md` |
| 硬件参数参考 | `npu-arch/references/npu-hardware-params.md` |
