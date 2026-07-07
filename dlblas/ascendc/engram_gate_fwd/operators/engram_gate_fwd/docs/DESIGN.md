# engram_gate_fwd AscendC 算子设计文档 (DESIGN.md)

> 版本: 2.0
> 生成时间: 2026-07-02
> CANN 版本: 9.0.0
> 目标芯片: Ascend 910B2 (DAV_2201)

---

## 1. 环境信息

| 参数 | 值 | 来源 |
|------|-----|------|
| 芯片型号 | Ascend 910B2 | 用户指定 |
| NpuArch | `DAV_2201` | `/npu-arch` skill 查表 |
| `__NPU_ARCH__` | 2201 | 编译宏 |
| UB 容量 | 192 KB (196,608 B) | DAV_2201 硬件规范 |
| L0C 容量 | 128 KB | DAV_2201 硬件规范 |
| L1 容量 | 128 KB | DAV_2201 硬件规范 |
| AI Core 数 | 24 | `/npu-arch` skill 查表 |
| CANN 版本 | 9.0.0 | 用户指定 |
| 编译器 | bisheng | CANN 9.0.0 配套 |

---

## 2. 算子概述

### 2.1 功能描述

`engram_gate_fwd` 是一个**全融合前向算子**，对输入的 `hidden_states`、`k`、`v` 三个张量执行带门控的残差连接。计算流水线为：

```
RMSNorm(rstd) -> Dot Product -> SignedSqrt Gate -> Sigmoid -> Gated Addition -> Cast
```

**核心计算步骤（逐行，fp32 精度）**：

```
给定行索引 (token t, head h)，令 x = hidden_states[t,h,:]，k = k[t,h,:]，v = v[t,:]：

1. RMSNorm rstd:
   rstd_x = 1 / sqrt(mean(x^2) + eps)
   rstd_k = 1 / sqrt(mean(k^2) + eps)

2. Dot Product:
   raw_dot = sum((x * weight_hidden[h,:]) * (k * weight_embed[h,:]))

3. Scale:
   dot = raw_dot * rstd_x * rstd_k * hidden_size^(-0.5)

4. Signed Sqrt Gate:
   signed_sqrt = sign(dot) * sqrt(max(|dot|, clamp_value))

5. Sigmoid:
   gate_score = 1 / (1 + exp(-signed_sqrt))

6. Gated Addition + Cast:
   output[t,h,:] = bf16(x + gate_score * v)
```

### 2.2 算子拆分方式

**全融合 Kernel（单算子，不拆分）**：RMSNorm、Dot Product、Gate、Gated Addition 全部在一个 kernel 内完成。

| 融合阶段 | 操作 | 融合理由 |
|---------|------|---------|
| RMSNorm (x) | square + ReduceSum + rsqrt | 与 Dot Product 共享已加载的 x 数据 |
| RMSNorm (k) | square + ReduceSum + rsqrt | 与 Dot Product 共享已加载的 k 数据 |
| Dot Product | elementwise mul + ReduceSum | 复用 RMSNorm 后的中间 fp32 buffer |
| Gate | scalar abs/clamp/sqrt/sign/sigmoid | 标量操作，不涉及向量数据 |
| Gated Addition | broadcast mul + add + cast | 最终输出，使用已有的 fp32 buffer |

**不拆分为子算子的理由**：
- 各阶段的数据高度耦合（x 和 k 被 RMSNorm 和 Dot Product 共用）
- 拆分会导致中间结果（rstd_x, rstd_k, raw_dot, gate_score）需在 GM 中来回读写
- 中间结果数据量小（每行 4 个 float = 16 字节），但拆分后的 kernel launch 开销和 GM 带宽浪费不可忽略
- DAV_2201 的 UB 容量（192KB）足以容纳单行（4096 元素）的全量计算

### 2.3 输入输出规格

| 张量 | Shape | dtype | 说明 |
|------|-------|-------|------|
| hidden_states | [num_tokens, hc_mult, hidden_size] | bf16 | 输入特征 |
| k | [num_tokens, hc_mult, hidden_size] | bf16 | Key 嵌入 |
| v | [num_tokens, hidden_size] | bf16 | Value 嵌入 |
| weight_hidden | [hc_mult, hidden_size] | bf16 | hidden_states 的 RMSNorm 权重 |
| weight_embed | [hc_mult, hidden_size] | bf16 | k 的 RMSNorm 权重 |
| clamp_value | scalar | float | Signed sqrt 下界 (典型值 1e-6) |
| eps | scalar | float | RMSNorm 数值稳定常数 (典型值 1e-20) |

| 输出 | Shape | dtype | 说明 |
|------|-------|-------|------|
| output | [num_tokens, hc_mult, hidden_size] | bf16 | Gate 加权输出 |
| raw_dot | [num_tokens, hc_mult] | fp32 | 未归一化点积 (backward 用) |
| gate_score | [num_tokens, hc_mult] | fp32 | Gate 值 (backward 用) |
| rstd_x | [num_tokens, hc_mult] | fp32 | hidden_states 的 rstd (backward 用) |
| rstd_k | [num_tokens, hc_mult] | fp32 | k 的 rstd (backward 用) |

### 2.4 典型 Shape

| 参数 | 基准值 | 说明 |
|------|--------|------|
| num_tokens | 4096 | Token 数量 |
| hc_mult | 4 | Head count multiplier |
| hidden_size | 4096 | 隐藏维度，天然 32B 对齐 |

---

## 3. 技术路线决策

### 3.1 决策树

```
目标芯片 Ascend 910B2 → NpuArch = DAV_2201
  → 非 DAV_3510 → 统一走通用 SIMD/MemBase 路线
  → RegBase (需 DAV_3510) & Blaze/tensor_api (需 DAV_3510) 均不适用
  → 算子类型 = 融合算子 (Reduction + Elementwise + Broadcast)
  → 归约轴 = 尾轴 (hidden_size 维度)
  → 单行数据 (fp32: ~16KB for hidden_size=4096) << UB 192KB
  → 采用 AR-FullLoad 模式（全行载入 UB，一次性完成归约）
```

### 3.2 决策结果

| 决策项 | 结论 | 理由 |
|--------|------|------|
| 编程路线 | **SIMD/MemBase** | DAV_2201 唯一可用路线 |
| 设计方法论 | `/ascendc-tiling-design` Reduction AR-FullLoad + Elementwise + Broadcast | 复用成熟设计模式 |
| 融合策略 | **全融合单 Kernel** | 各阶段数据高度耦合，UB 容量充足 |
| 归约模式 | **AR-FullLoad** | 单行 fp32 数据量 16KB，远小于 UB 192KB |
| 多核切分 | **按 row 均分** | 行间计算独立，天然并行 |

---

## 4. 多核切分策略 (Tiling)

### 4.1 切分方式

按 `total_rows = num_tokens * hc_mult` 均匀分配到 AI Core。切分单元对齐到 token 边界（以 hc_mult 为单位），保证同一 token 的不同 head 分配给同一 core，从而 v 可被同 token 内共享。

```
tile_rows_per_core = ceil(total_rows / core_num)
tile_rows_per_core = ceil(tile_rows_per_core / hc_mult) * hc_mult   // 对齐到 token 边界
```

### 4.2 每个 Core 的处理范围

```
row_start = block_idx * tile_rows_per_core
row_end   = min(row_start + tile_rows_per_core, total_rows)
if (row_start >= total_rows) → 空核提前退出
```

Core 内部采用双层循环：外层遍历 token，内层遍历 head。同 token 的 v[t,:] 数据在内层 head 循环中可复用（当前版本加载于 head 循环内，未来优化为外提至 token 循环外）。

### 4.3 TilingData 结构

```cpp
struct EngramGateFwdTilingData {
    // 维度
    uint64_t num_tokens;
    uint64_t hc_mult;
    uint64_t hidden_size;
    uint64_t hidden_size_align;        // 32B 对齐后的 fp32 元素数
    uint64_t hidden_size_align_bf16;   // 32B 对齐后的 bf16 元素数

    // 多核切分
    uint64_t tile_rows_per_core;
    uint64_t total_rows;
    uint32_t core_num;

    // 标量参数（Host 侧预计算）
    float clamp_value;
    float eps;
    float scalar;                      // hidden_size^(-0.5)
    float hidden_size_float;

    // GM 基地址偏移
    uint64_t hidden_states_offset;
    uint64_t k_offset;
    uint64_t v_offset;
    uint64_t weight_hidden_offset;
    uint64_t weight_embed_offset;
    uint64_t output_offset;
    uint64_t raw_dot_offset;
    uint64_t gate_score_offset;
    uint64_t rstd_x_offset;
    uint64_t rstd_k_offset;
};
```

Tiling 计算在 Host 侧执行（`ComputeTiling` 函数），计算结果打包为 TilingData 下发到 Device。Host 侧还负责 UB 容量检查：若 `ComputeUBUsage(hidden_size, hc_mult) > UB_CAPACITY`，拒绝执行并报错。

---

## 5. UB 切分与内存管理

### 5.1 32B 对齐规范

DAV_2201 要求 UB 数据按 32 字节对齐才能使用 Vector 指令高效处理：

```
hidden_size_align_bf16 = ceil(hidden_size * sizeof(uint16_t) / 32) * 32 / sizeof(uint16_t)
hidden_size_align      = ceil(hidden_size * sizeof(float)    / 32) * 32 / sizeof(float)
```

对于 `hidden_size=4096`（天然 32B 对齐）：`hidden_size_align_bf16 = 4096`，`hidden_size_align = 4096`。

### 5.2 Buffer 规划

采用**单缓冲模式**（`TQue<..., 1>`），每条 Queue 分配 1 个 slot。所有 Queue 在 `Init()` 中一次性分配，kernel 运行期间固定不变。

| Queue 名称 | 方向 | 元素类型 | 大小 (字节) | 用途 |
|-----------|------|---------|------------|------|
| `weight_hidden_q_` | VECIN | bf16 | `hbb_align` | weight_hidden 当前 head 行 |
| `weight_embed_q_` | VECIN | bf16 | `hbb_align` | weight_embed 当前 head 行 |
| `v_q_` | VECIN | bf16 | `hbb_align` | v 当前 token 行 |
| `x_q_` | VECIN | bf16 | `hbb_align` | hidden_states 当前行 |
| `k_q_` | VECIN | bf16 | `hbb_align` | k 当前行 |
| `out_q_` | VECOUT | bf16 | `hbb_align` | 输出行 buffer |
| `buf_a_q_` | VECIN | fp32 | `hfb_align` | 工作区 A (主计算区) |
| `buf_b_q_` | VECIN | fp32 | `hfb_align` | 工作区 B (辅助计算区) |
| `buf_c_q_` | VECIN | fp32 | `hfb_align` | 工作区 C (v 广播区) |
| `tmp_q_` | VECIN | fp32 | 8192 | ReduceSum 临时 buffer |
| `scalar_q_` | VECIN | fp32 | 32 | 标量写入中转 buffer |

其中：
- `hbb_align = AlignTo32B(hidden_size * sizeof(uint16_t))`  —— bf16 行对齐字节数
- `hfb_align = AlignTo32B(hidden_size * sizeof(float))`     —— fp32 行对齐字节数

### 5.3 UB 容量分析

```
UB_total = hbb_align * 6 + hfb_align * 3 + 8192 + 32
```

| hidden_size | hbb_align | hfb_align | UB 总使用 | UB 192KB | 处理策略 |
|-------------|-----------|-----------|----------|----------|---------|
| 1024 | 2,048 | 4,096 | 12,288+12,288+8,224 = 32,800 B | OK | AR-FullLoad |
| 4096 | 8,192 | 16,384 | 49,152+49,152+8,224 = 106,528 B (~104KB) | OK (54%) | AR-FullLoad |
| 6800 | 13,600 | 27,200 | 81,600+81,600+8,224 = 171,424 B (~167KB) | OK (87%) | AR-FullLoad |
| 8192 | 16,384 | 32,768 | 98,304+98,304+8,224 = 204,832 B (~200KB) | **溢出** | 需 AR-ColSplit 或拒绝 |

**策略**：
- `hidden_size <= 6800`：AR-FullLoad，全行载入 UB 一次性归约
- `hidden_size > 6800`：当前 Host 侧拒绝（`ComputeUBUsage` 检查），未来可实现 AR-ColSplit 分载模式（将列拆分为多个 chunk 分别处理）

### 5.4 Weight 逐行加载策略（懒加载）

`weight_hidden` 和 `weight_embed` 的 shape 为 `[hc_mult, hidden_size]`，采用**逐 head 懒加载**策略：每处理一个 head，用 `DataCopyPad` 加载对应的 `wh[h,:]` 和 `we[h,:]` 单行到 UB。不使用持久化预加载。

**选择理由**：
- `hc_mult` 通常很小（默认 4），懒加载的额外 DMA 开销（每次约 `hidden_size*2` 字节）可忽略
- 避免在 UB 中分配 `hc_mult * hidden_size * 2 * 2` 的持久化权重空间
- 释放的 UB 空间用于保证大 `hidden_size` 场景的容量裕量

### 5.5 v 数据加载策略

当前版本中 v 在 head 内层循环中加载（per-head 加载），同 token 的不同 head 会重复读取相同的 v[t,:] 行。`hc_mult=4` 时重复 3 次。

**优化方向**：将 v 加载外提至 token 循环外层，同 token 只加载一次，通过 UB buffer 复用避免重复 DMA。预期节省约 `(hc_mult-1) * hidden_size * 2` 字节的 DMA 读取量。

### 5.6 Buffer 复用时序

Buffer A (`buf_a_q_`, fp32) 生命周期（单行处理内复用 5 次）：
```
x_fp32 (Cast) -> x^2 (Mul) -> ReduceSum ->
x_fp32 (Re-Cast) * wh_fp32 (Cast) -> Prod (Mul) -> ...
k_fp32 (Re-Cast) * we_fp32 (Cast) -> Prod (Mul) -> ...
(x*wh) * (k*we) -> ReduceSum ->
x_fp32 (Re-Cast) + gate*v_fp32 (Mul+Add) -> output_fp32
```

Buffer B (`buf_b_q_`, fp32) 生命周期（复用 4 次）：
```
k_fp32 (Cast) -> k^2 (Mul) -> ReduceSum ->
wh_fp32 (Cast) ->
k_fp32 (Re-Cast) * we_fp32 (after cast in buf_C) -> ...
```

Buffer C (`buf_c_q_`, fp32) 生命周期（复用 3 次）：
```
we_fp32 (Cast) ->
v_fp32 (Cast) * gate_score (Muls) ->
gate_scaled_v (复用)
```

---

## 6. 向量化策略

### 6.1 Vector API 使用原则

所有逐元素运算和归约运算均使用 AscendC Level 2 Vector API，硬件自动向量化处理。

| 阶段 | 操作 | Vector API | 操作长度 | 说明 |
|------|------|-----------|---------|------|
| RMSNorm | bf16->fp32 | `Cast` | `hs_align` | 类型提升 |
| RMSNorm | 平方 | `Mul` (in-place) | `hs_align` | x^2, k^2 |
| RMSNorm | 归约求和 | `ReduceSum` | `hs_align` | 沿 hidden_size 归约 |
| Dot Product | bf16->fp32 逐行 | `Cast` | `hs_align` | wh, we, x, k 各一次 |
| Dot Product | 逐元素乘法 | `Mul` | `hs_align` | x*wh, k*we, (x*wh)*(k*we) |
| Dot Product | 归约求和 | `ReduceSum` | `hs_align` | 最终点积 |
| Gated Addition | fp32 广播乘法 | `Muls` | `hs_align` | v * gate_score |
| Gated Addition | fp32 加法 | `Add` | `hs_align` | x + gate*v |
| Output Cast | fp32->bf16 | `Cast` | `hs_align` | CAST_ROUND |

### 6.2 非对齐 hidden_size 处理

当 `hidden_size` 非 32B 对齐时（如 `hidden_size=4097`），需要特殊处理：

1. **DataCopyPad blockLen**：使用有效数据字节数 `hidden_size * sizeof(uint16_t)`，防止读取 GM padding 区域的垃圾数据
2. **Vector API count**：使用对齐后的 `hs_align` 以满足 Vector 指令的 32B 对齐需求
3. **Padding 清零**：每个 Cast 操作后，显式将 fp32 buffer 的 padding 区域 (`[hidden_size, hidden_size_align)`) 置零，确保 Mul 和 ReduceSum 不受垃圾数据影响

### 6.3 Gate 计算的标量路径

Signed Sqrt 和 Sigmoid 作用于 ReduceSum 输出后的**标量**（每行 1 个 float），采用标量路径实现：

- **Sqrt/Exp**：将标量值写入 buffer 首元素，调用 `count=1` 的 Level 2 Vector API
- **符号处理**：直接调用 C 语言标量运算

不使用 `AscendC::Sigmoid` 高级 API 的理由：该 API 在 DAV_2201 上需要 `sharedTmpBuffer`（较大临时 buffer），而标量场景下 `1/(1+exp(-x))` 仅需 3 个标量操作，更高效且不占用额外 UB。

---

## 7. 数据流设计

### 7.1 整体架构

```
┌──────────────────────────────────────────────────────────────────┐
│ Host Side (Tiling)                                                │
│  1. ComputeTiling: 计算 tile_rows_per_core、对齐参数、标量      │
│  2. ComputeUBUsage: 验证 UB 容量                                 │
│  3. 下发 EngramGateFwdTilingData 到 Device                       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Device Side (Kernel - per core)                                  │
│                                                                  │
│  for each token t in core's token range:                         │
│    for each head h in token's head range:                        │
│      ┌─ Load Phase ───────────────────────────────────────┐     │
│      │ DataCopyPad: wh[h,:], we[h,:], x[t,h,:], k[t,h,:] │     │
│      │ DataCopyPad: v[t,:] (per-head, 优化方向: per-token) │    │
│      └────────────────────────────────────────────────────┘     │
│      ┌─ Compute Phase (fp32) ────────────────────────────┐     │
│      │ Phase 1: RMSNorm rstd_x                            │     │
│      │   Cast(x) → Mul(x^2) → ReduceSum → /N+eps → Sqrt  │     │
│      │   → rstd_x = 1/Sqrt_result                         │     │
│      │ Phase 2: RMSNorm rstd_k                            │     │
│      │   Cast(k) → Mul(k^2) → ReduceSum → /N+eps → Sqrt  │     │
│      │   → rstd_k = 1/Sqrt_result                         │     │
│      │ Phase 3: Dot Product                               │     │
│      │   Cast(x)*Cast(wh) → Cast(k)*Cast(we) → Mul →     │     │
│      │   ReduceSum → raw_dot                              │     │
│      │ Phase 4: Gate (scalar fp32)                        │     │
│      │   SignedSqrt + Sigmoid → gate_score                │     │
│      │ Phase 5: Broadcast Output                          │     │
│      │   Cast(v)*gate → Cast(x)+result → Cast→bf16       │     │
│      └────────────────────────────────────────────────────┘     │
│      ┌─ Store Phase ─────────────────────────────────────┐     │
│      │ DataCopyPad: output_bf16 → GM                      │     │
│      │ DataCopyPad: raw_dot, gate_score, rstd_x, rstd_k  │     │
│      └────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

### 7.2 数据依赖与流水线

```
GM_READ: wh, we, x, k, v  (5 条 DataCopyPad, 可流水)
    │
    ▼
UB COMPUTE: Cast -> Mul -> ReduceSum -> (Gate scalar) -> Mul/Add -> Cast
    │
    ▼
GM_WRITE: output, raw_dot, gate_score, rstd_x, rstd_k  (5 条 DataCopyPad, 可流水)
```

当前为单缓冲模式，CopyIn / Compute / CopyOut 串行执行。优化方向为引入 Double Buffer 流水线：在处理当前行计算的同时预取下一行数据。

---

## 8. API 映射与验证

### 8.1 数据搬运

| 操作 | API | 签名 | 头文件 | 验证状态 |
|------|-----|------|--------|---------|
| GM→UB (bf16) | `DataCopyPad` (MTE2) | `DataCopyPad(LocalTensor<T>, GlobalTensor<T>, DataCopyParams, DataCopyPadParams)` | `kernel_operator_data_copy_intf.h` | 已验证 |
| UB→GM (bf16) | `DataCopyPad` (MTE3) | `DataCopyPad(GlobalTensor<T>, LocalTensor<T>, DataCopyParams)` | `kernel_operator_data_copy_intf.h` | 已验证 |
| UB→GM (fp32 标量) | `DataCopyPad` (MTE3) | 同上, `blockLen=4` | `kernel_operator_data_copy_intf.h` | 已验证 |

**关键约束**：
- `DataCopyPad` 自动处理非对齐场景，**blockLen 使用有效数据字节数**，不传入对齐后的大小
- `DataCopyPad` 的 `DataCopyParams` 结构中：`{maxBlockNum, blockLen, srcStride, dstStride}`

### 8.2 精度转换

| 操作 | API | 签名 | 头文件 | 验证状态 |
|------|-----|------|--------|---------|
| bf16->fp32 | `Cast` (Level 2) | `Cast<float, bf16_t>(dst, src, CAST_NONE, count)` | `kernel_operator_vec_vconv_intf.h` | 已验证 |
| fp32->bf16 | `Cast` (Level 2) | `Cast<bf16_t, float,>(dst, src, CAST_ROUND, count)` | `kernel_operator_vec_vconv_intf.h` | 已验证 |

**关键约束**：
- 输入 Cast 使用 `CAST_NONE`（截断模式，bf16->fp32 无损）
- 输出 Cast 使用 `CAST_ROUND`（最近舍入模式）
- `bfloat16_t` 是 AscendC 原生类型 (`DT_BF16=27`)

### 8.3 归约操作

| 操作 | API | 签名 | 头文件 | 验证状态 |
|------|-----|------|--------|---------|
| 整行求和 | `ReduceSum` (Level 2) | `ReduceSum<T>(dst, src, sharedTmpBuffer, count)` | `kernel_operator_vec_reduce_intf.h` | 已验证 |

**关键约束**：
- `count` 参数使用 `hs_align`（对齐后的元素个数，padding 区域已清零）
- `dst` 起始地址需 8 字节对齐
- `sharedTmpBuffer` 类型必须与计算类型相同 (float)
- `sharedTmpBuffer` 最小大小：取 `ComputeReduceBufSize(rLengthAlign, 4)` 和 `4096` 的较大值；当前分配 8192 字节

### 8.4 逐元素运算

| 操作 | API | 签名 | 头文件 | 验证状态 |
|------|-----|------|--------|---------|
| 逐元素乘 | `Mul` (Level 2) | `Mul<T>(dst, src0, src1, count)` | `kernel_operator_vec_binary_intf.h` | 已验证 |
| 逐元素加 | `Add` (Level 2) | `Add<T>(dst, src0, src1, count)` | `kernel_operator_vec_binary_intf.h` | 已验证 |
| 标量乘 | `Muls` (Level 2) | `Muls<T>(dst, src, scalar, count)` | `kernel_operator_vec_binary_scalar_intf.h` | 已验证 |

### 8.5 超越函数

| 操作 | API | 签名 | 头文件 | 验证状态 |
|------|-----|------|--------|---------|
| 平方根 | `Sqrt` (Level 2) | `Sqrt<T>(dst, src, count)` | `kernel_operator_vec_unary_intf.h` | 已验证 |
| 指数 | `Exp` (Level 2) | `Exp<T>(dst, src, count)` | `kernel_operator_vec_unary_intf.h` | 已验证 |

### 8.6 Sigmoid 实现方式说明

DAV_2201 头文件中未发现独立的 `AscendC::Sigmoid` API。即使存在高级 Sigmoid API，也需要 `sharedTmpBuffer`（较大临时 buffer）。在当前逐行处理标量 gate 的场景下，使用 `Exp` + 标量算术组合 (`1/(1+exp(-x))`) 更高效且不占用额外 UB 空间。

---

## 9. 计算精度策略

### 9.1 精度标准

按 `/ops-precision-standard` 浮点计算类社区标准：

| 输出类型 | MERE 阈值 | MARE 阈值 |
|---------|----------|----------|
| bf16 output | < 2^-7 ≈ 0.00781 | < 10 * 2^-7 ≈ 0.0781 |
| fp32 (raw_dot, gate_score, rstd) | < 2^-13 ≈ 0.000122 | < 10 * 2^-13 ≈ 0.00122 |

### 9.2 精度路径

| 阶段 | 输入 dtype | 计算 dtype | 输出 dtype | 理由 |
|------|-----------|-----------|-----------|------|
| RMSNorm (x^2, k^2) | bf16 | fp32 | fp32 | 平方需 fp32 防溢出 |
| ReduceSum | fp32 | fp32 | fp32 | 归约累加需 fp32 精度 |
| Dot Product (Mul chain) | fp32 | fp32 | fp32 | 乘积累加需 fp32 |
| Gate (sqrt/exp) | fp32 scalar | fp32 | fp32 | 标量计算，fp32 充足 |
| Gated Addition | fp32 | fp32 | fp32 | 保持累计精度 |
| 最终输出 Cast | fp32 | — | bf16 | 最近舍入 |

**总体路径**：`bf16 input → fp32 compute → bf16 output`

### 9.3 数值稳定性保障

| 风险点 | 缓解措施 | 详情 |
|--------|---------|------|
| ReduceSum 大向量累加 (D=4096) | fp32 累加 | fp32 7 位有效数字，4096 个元素累加相对误差 < 6.4e-6 |
| Rsqrt 零值 | eps = 1e-20 | 极小 epsilon 保证分母非零 |
| Sigmoid 饱和 | fp32 计算 | signed_sqrt 在合理范围内 (|dot| clamped) |
| Clamp 边界 | clamp_value = 1e-6 | 避免 sqrt(0) 产生 subnormal |
| 非对齐 padding 干扰 | Cast 后显式清零 | 确保 ReduceSum 不受垃圾数据影响 |

### 9.4 实测精度

| 输出 | hidden_size | Max Abs Err | Max Rel Err | 判定 |
|------|------------|------------|------------|------|
| output (bf16) | 4096 | 2.44e-04 | 1.51e-02 | PASS |
| output (bf16) | 4097 | 7.81e-03 | 7.69e-03 | PASS |
| raw_dot (fp32) | 4096 | 3.81e-05 | 4.52e-05 | PASS |
| gate_score (fp32) | 4096 | 4.77e-07 | 9.16e-07 | PASS |
| rstd_x (fp32) | 4096 | 2.38e-07 | 2.41e-07 | PASS |
| rstd_k (fp32) | 4096 | 1.19e-07 | 1.22e-07 | PASS |

全部 8 个测试用例（含非对齐 hidden_size=4097）均通过精度验证。

---

## 10. 分支场景覆盖

### 10.1 Shape 分支

| 场景 | hidden_size | UB 使用 | 处理策略 |
|------|-------------|---------|---------|
| 小 hidden_size | <= 4096 | <= 104 KB | AR-FullLoad |
| 中 hidden_size | 4097~6800 | 104~167 KB | AR-FullLoad（需 padding 清零） |
| 大 hidden_size | 6801~8192 | 167~200 KB | 当前：Host 侧 UB 检查拒绝；未来：AR-ColSplit |

### 10.2 对齐分支

| 场景 | 示例 | 处理 |
|------|------|------|
| 32B 对齐 | hidden_size=4096 | `hs_align == hidden_size`，向量 API 直接处理，无需 padding 清零 |
| 非 32B 对齐 | hidden_size=4097 | `hs_align = 4098`（bf16）/ `hs_align = 4100`（fp32）；DataCopyPad blockLen 使用有效数据大小；每个 Cast 后显式清零 padding 区域 |

### 10.3 边界分支

| 场景 | 处理 |
|------|------|
| Core 数 > 所需 block 数 | `if (row_start >= row_end) return;` 空核提前退出 |
| 尾 token 的 head 不完整 | 内层 head 循环按 `head_end = min(hc_mult, row_end % hc_mult)` 限制，确保不越界 |
| num_tokens=1 | 正常处理，tile_rows_per_core 对齐逻辑不变 |

---

## 11. 性能分析

### 11.1 实测指标 (num_tokens=32, hc_mult=4, hidden_size=4096)

| 指标 | 值 |
|------|-----|
| Task Duration | 16.52 us |
| aiv_vec (vector compute) | 6.06 us (41.7%) |
| aiv_scalar (scalar pipe) | ~5.6 us (38.7%) |
| aiv_mte2 (memory read) | ~2.1 us (14.3%) |
| aiv_mte3 (memory write) | ~2.7 us (18.5%) |
| Pipeline stall | ~59% (单缓冲导致) |

### 11.2 瓶颈分析

1. **流水线 stall (~59%)**：单缓冲模式下 CopyIn/Compute/CopyOut 串行，大量时间消耗在等待
2. **标量操作开销 (38.7%)**：Gate 计算的 Sqrt/Exp 使用标量路径，每行 2 次标量 API 调用
3. **v 重复加载**：同 token 内每个 head 都重新加载 v[t,:]，额外 DMA

### 11.3 优化路线图

| 优先级 | 优化项 | 预期收益 | 说明 |
|--------|--------|---------|------|
| P0 | Double Buffer (x/k/out 双缓冲) | ~20-30% | 隐藏 DMA 延迟 |
| P1 | v-load hoisting (外提至 token 外循环) | ~5-10% | 减少 DMA 读取次数 |
| P2 | AR-ColSplit 大 hidden_size 路径 | 扩展 shape 支持 | hidden_size > 6800 |

---

## 12. 实现状态

| 阶段 | 状态 | 备注 |
|------|------|------|
| 工程搭建 | 完成 | 基于 AscendC direct-invoke 模板 |
| Tiling 实现 | 完成 | `EngramGateFwdTilingData` + `ComputeTiling` + `ComputeUBUsage` |
| Kernel 实现 | 完成 | 全融合: RMSNorm + Dot + Gate + Broadcast |
| 编译 | 完成 | DAV_2201, CANN 9.0.0, bisheng |
| Level 0~2 验证 | 全部通过 (8/8) | 含非对齐 hidden_size=4097 |
| 性能采集 | 完成 | msprof |
| PyTorch 接入 | 完成 | `torch.ops` 注册 |
| Double Buffer | 待优化 | 当前单缓冲 |

---

## 13. 附录

### A. 参考资源

| 资源 | 路径 |
|------|------|
| Reduce API | `$ASCENDC_DIR/include/basic_api/interface/kernel_operator_vec_reduce_intf.h` |
| Unary API (Sqrt, Exp) | `$ASCENDC_DIR/include/basic_api/interface/kernel_operator_vec_unary_intf.h` |
| Binary API (Mul, Add) | `$ASCENDC_DIR/include/basic_api/interface/kernel_operator_vec_binary_intf.h` |
| Binary Scalar API (Muls) | `$ASCENDC_DIR/include/basic_api/interface/kernel_operator_vec_binary_scalar_intf.h` |
| Vconv API (Cast) | `$ASCENDC_DIR/include/basic_api/interface/kernel_operator_vec_vconv_intf.h` |
| DataCopy API | `$ASCENDC_DIR/include/basic_api/interface/kernel_operator_data_copy_intf.h` |
| AR-FullLoad 设计 | `ascendc-tiling-design/references/reduction/patterns.md` |
| 精度标准 | `ops-precision-standard/reference/float_compute_community.md` |

### B. 术语表

| 术语 | 说明 |
|------|------|
| AR-FullLoad | All-Reduce FullLoad：归约轴数据全量一次性载入 UB |
| AR-ColSplit | 归约轴数据分多个 column chunk 分批载入处理 |
| hc_mult | Head count multiplier |
| hs_align | 32B 对齐后的元素个数（用于 Vector API count） |
