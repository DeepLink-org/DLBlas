# engram_gate_w_reduce 算子技术设计文档 (DESIGN)

## 1. 基本信息

| 项目 | 内容 |
|------|------|
| **算子名称** | engram_gate_w_reduce |
| **算子类型** | Fused Reduction + Broadcast Elementwise |
| **目标架构** | DAV_2201 (Ascend 910B2), `__NPU_ARCH__=2201` |
| **CANN 版本** | 9.0.0 |
| **编程模型** | SIMD / MemBase（通用路线） |

## 2. 路线决策

| 决策维度 | 结论 | 理由 |
|----------|------|------|
| **目标架构** | DAV_2201 | 芯片型号 Ascend910B2 |
| **算子类型** | Reduction + Elementwise 融合 | 核心操作为沿 dim=0 的 ReduceSum，后接 Broadcast Multiply-Accumulate |
| **技术路线** | **SIMD / MemBase（通用路线）** | DAV_2201 不走 RegBase 也不走 Blaze；该架构仅支持通用 SIMD/MemBase 路线 |
| **融合策略** | **单 Kernel 融合** | Reduction 与 Elementwise 在同一个 kernel 内完成，避免中间结果 GM 读写、最大化数据复用 |

## 3. 数学定义

### 3.1 输入张量

| 张量 | Shape | dtype | 布局 |
|------|-------|-------|------|
| `grad_w_partial` | [108, 4, hidden_size] | float32 | 连续 |
| `weight_hidden` | [4, hidden_size] | bfloat16 | 连续 |
| `weight_embed` | [4, hidden_size] | bfloat16 | 连续 |
| `grad_weight_hidden` | [4, hidden_size] | float32 | 连续（in-place 累加目标） |
| `grad_weight_embed` | [4, hidden_size] | float32 | 连续（in-place 累加目标） |

### 3.2 计算

```
Step 1 (Reduce):    grad_w_sum = sum(grad_w_partial, dim=0)    // [108,4,H] → [4,H]
Step 2 (MulAccum):  grad_weight_hidden += grad_w_sum ⊙ weight_embed_float
Step 3 (MulAccum):  grad_weight_embed += grad_w_sum ⊙ weight_hidden_float
```

### 3.3 输出张量

| 张量 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `grad_weight_hidden` | [4, hidden_size] | float32 | 累加后的结果（与输入同一 buffer） |
| `grad_weight_embed` | [4, hidden_size] | float32 | 累加后的结果（与输入同一 buffer） |

## 4. 合轴分析（Reduction）

原始 shape: `[108, 4, hidden_size]`, axes: `[0]`

- 轴标记: `[R=108, A=4, A=hidden_size]`
- 合并相邻同类型轴: `[R=108, A=4*hidden_size]`
- **模式判定**: ARA 模式（R 在前，A0 = 4*hidden_size > 1），轴0归约

### 4.1 分载判定

- R = 108
- A0 = 4 * hidden_size（以 hidden_size=4096 为例，A0=16384）
- UB = 192KB = 196608 bytes

全载条件检查：在 UB 中至少容纳所有 R 行 × tileA0Len 的数据。当 tileA0Len > 0，108 * tileA0Len * 4 字节将远超 UB 容量，因此需要 **ARA-RowSplit（分载）** 模式。

**分载策略**: 沿 R 维分 chunk 处理。但由于 R=108 本身不大，且 CANN 9.0.0 DAV_2201 不支持 `Pattern::Reduce::RA`，采用**逐行加载 + 累加**的简化策略。

## 5. 算子架构总览

```
┌──────────────────────────────────────────────────┐
│                  Host 侧 Tiling                    │
│  - 多核切分：沿 hidden_size 维度均分              │
│  - A0 = 4 * hidden_size, tileA0Len = A0 / coreNum│
│  - R=108 固定                                     │
│  - 计算 tmpBufSize（给 ReduceSum/Add API）         │
└──────────────────────┬───────────────────────────┘
                       │ TilingData
┌──────────────────────▼───────────────────────────┐
│                  Device 侧 Kernel                  │
│                                                    │
│  Phase 1: Reduction (逐行累加)                     │
│  ┌──────────────────────────────────────────────┐ │
│  │ for row 0..107:                              │ │
│  │   Load grad_w_partial[row, :tileA0Len] → UB  │ │
│  │   if row == 0: accumBuf = inputBuf           │ │
│  │   else:        accumBuf += inputBuf (Add)    │ │
│  │  (Double Buffer: Load row i+1 while Add i)   │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  Phase 2: Multiply-Accumulate (广播逐元素)         │
│  ┌──────────────────────────────────────────────┐ │
│  │ Load weight_hidden[0:4, tile_hid] (BF16)     │ │
│  │ Cast → FP32                                   │ │
│  │ Load weight_embed[0:4, tile_hid] (BF16)      │ │
│  │ Cast → FP32                                   │ │
│  │ Load grad_weight_hidden[0:4, tile_hid] (FP32) │ │
│  │ Load grad_weight_embed[0:4, tile_hid] (FP32) │ │
│  │                                              │ │
│  │ grad_weight_hidden += accumBuf * weight_embed │ │
│  │ grad_weight_embed += accumBuf * weight_hidden │ │
│  │                                              │ │
│  │ Store grad_weight_hidden, grad_weight_embed  │ │
│  └──────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
```

## 6. 多核切分策略

### 6.1 切分方案

沿 **hidden_size 维度**（即 A0 在 [4, hidden_size] 上的投影）均分给多个 AI Core：

```
core_i 处理:
  - grad_w_partial[:, :, start_hid : start_hid + tile_hidden_len]
  - weight_hidden[:, start_hid : start_hid + tile_hidden_len]
  - weight_embed[:, start_hid : start_hid + tile_hidden_len]
  - grad_weight_hidden[:, start_hid : start_hid + tile_hidden_len]
  - grad_weight_embed[:, start_hid : start_hid + tile_hidden_len]
```

### 6.2 Tiling 参数

```cpp
// Host 侧
uint32_t blockDim = min(coreNum, hidden_size);      // 最大并行度受限于 hidden_size
uint32_t tileHiddenLen = (hidden_size + blockDim - 1) / blockDim;  // 每个核处理的 hidden_size 段
uint32_t tileA0Len = 4 * tileHiddenLen;              // 最后核可能略小
uint32_t tailHiddenLen = hidden_size - (blockDim - 1) * tileHiddenLen;
```

## 7. UB Buffer 规划

### 7.1 UB 容量基线

- **UB 总容量**: 192 KB = 196,608 bytes（DAV_2201）
- **Cache Line**: 32 bytes
- **Vector Register Width**: 256 bits = 32 bytes = 8 FP32 elements

### 7.2 Phase 1 (Reduction) Buffer 列表

| Buffer | 类型 | 大小 | 用途 |
|--------|------|------|------|
| `pingBuf` | FP32 | tileA0Len * 4 B | 双缓冲-乒，存放当前读入的 grad_w_partial 行 |
| `pongBuf` | FP32 | tileA0Len * 4 B | 双缓冲-乓，预读下一行 |
| `accumBuf` | FP32 | tileA0Len * 4 B | 累加器，跨 108 行累积（输出 grad_w_sum） |

Phase 1 总 UB 用量: `3 * tileA0Len * 4 B = 12 * tileA0Len B`

以 hidden_size=4096, 24 cores 为例: tileA0Len=684, UB=8,208 B << 192KB

### 7.3 Phase 2 (Elementwise) Buffer 列表

Phase 2 复用 Phase 1 的 pingBuf/pongBuf 空间：

| Buffer | 类型 | 大小 | 用途 |
|--------|------|------|------|
| `accumBuf` | FP32 | tileA0Len * 4 B | grad_w_sum（保留自 Phase 1） |
| `whBf16Buf` | bfloat16_t | tileA0Len * 2 B | weight_hidden 加载 (BF16) |
| `weBf16Buf` | bfloat16_t | tileA0Len * 2 B | weight_embed 加载 (BF16) |
| `whFp32Buf` | FP32 | tileA0Len * 4 B | weight_hidden 转 FP32 后 |
| `weFp32Buf` | FP32 | tileA0Len * 4 B | weight_embed 转 FP32 后 |
| `ghBuf` | FP32 | tileA0Len * 4 B | grad_weight_hidden (in-place) |
| `geBuf` | FP32 | tileA0Len * 4 B | grad_weight_embed (in-place) |

Phase 2 总 UB 用量: `(4*4 + 2*2 + 4 + 4 + 4) * tileA0Len = 24 * tileA0Len B`

以 hidden_size=4096, 24 cores: tileA0Len=684, UB=16,416 B << 192KB

### 7.4 总 UB 使用

取 Phase 1 和 Phase 2 的较大者（Phase 2 可以直接复用 Phase 1 的 ping/pong 空间）:

```
UB_max = 24 * tileA0Len B
```

极端情况（hidden_size 很大且核心数很少时 tileA0Len 较大），需要校验 `UB_max ≤ 192KB`。

## 8. 数据流

### 8.1 Phase 1: Reduction (逐行累加)

```
GM[grad_w_partial]                              UB
     │                                            │
     ├─[DataCopy] row 0 ─────────────────────→ pingBuf     ──[Duplicate 0]──→ accumBuf
     ├─[DataCopy] row 1 ─────────────────────→ pongBuf     ──[Add]─────────→ accumBuf (+= pongBuf)
     │    ... (Double Buffer 流水)                         │
     ├─[DataCopy] row i ─────────────────────→ pingBuf     ──[Add]─────────→ accumBuf (+= pingBuf)
     ├─[DataCopy] row i+1 ───────────────────→ pongBuf     ──[Add]─────────→ accumBuf (+= pongBuf)
     │    ...                                              │
     └─[DataCopy] row 107 ───────────────────→ ___Buf      ──[Add]─────────→ accumBuf (= grad_w_sum)
```

**Double Buffer 流水**:
```
Iter 0: Load row 0 → pingBuf (同步)
Iter 1: Load row 1 → pongBuf (异步) | Add pingBuf → accumBuf
Iter 2: Load row 2 → pingBuf (异步) | Add pongBuf → accumBuf
...
```

### 8.2 Phase 2: Multiply-Accumulate

```
GM[weight_hidden]  ──[DataCopy]──→ whBf16Buf  ──[Cast]──→ whFp32Buf
GM[weight_embed]   ──[DataCopy]──→ weBf16Buf  ──[Cast]──→ weFp32Buf
GM[grad_w_hidden]  ──[DataCopy]──→ ghBuf
GM[grad_w_embed]   ──[DataCopy]──→ geBuf
                                      │
    accumBuf ─────────────────────────┤
    weFp32Buf ────────────────────────┤──[MulAddDst]──→ ghBuf  (+= accumBuf * weFp32)
    accumBuf ─────────────────────────┤
    whFp32Buf ────────────────────────┤──[MulAddDst]──→ geBuf  (+= accumBuf * whFp32)
                                      │
GM[grad_w_hidden]  ←──[DataCopy]── ghBuf
GM[grad_w_embed]   ←──[DataCopy]── geBuf
```

## 9. API 映射

### 9.1 已验证 API 清单

| 序号 | API | 签名 | 用途 | 验证状态 |
|------|-----|------|------|---------|
| 1 | `DataCopy` | `DataCopy<T>(LocalTensor<T>, GlobalTensor<T>, DataCopyParams)` | GM→UB 搬运 | ✅ CANN 9.0.0 已确认 |
| 2 | `DataCopy` | `DataCopy<T>(GlobalTensor<T>, LocalTensor<T>, DataCopyParams)` | UB→GM 搬运 | ✅ CANN 9.0.0 已确认 |
| 3 | `DataCopy` | `DataCopy<float, bfloat16_t>(LocalTensor<float>, LocalTensor<bfloat16_t>, DataCopyParams, DataCopyEnhancedParams)` | UB 内 BF16→FP32 转换 | ⚠️ 仅限 DAV_2201 (NPU_ARCH=2201) |
| 4 | `Cast` | `Cast<T,U>(LocalTensor<T>, LocalTensor<U>, RoundMode, uint32_t count)` | BF16→FP32 类型转换（替代方案） | ✅ CANN 9.0.0 已确认 |
| 5 | `Add` | `Add<T>(LocalTensor<T>, LocalTensor<T>, LocalTensor<T>, int32_t count)` | 逐元素加法（Phase 1 累加） | ✅ CANN 9.0.0 已确认 |
| 6 | `MulAddDst` | `MulAddDst<T,U>(LocalTensor<T>, LocalTensor<U>, LocalTensor<U>, int32_t count)` | dst += src0 * src1（Phase 2 融合乘加） | ✅ CANN 9.0.0 已确认 |
| 7 | `Duplicate` | `Duplicate<T>(LocalTensor<T>, T scalarValue, int32_t count)` | Phase 1 累加器初始化为 0 | ✅ CANN 9.0.0 已确认 |

### 9.2 API 类型约束

| 约束 | 说明 |
|------|------|
| GM→UB DataCopy 要求 dst 与 src 类型一致 | BF16 数据以 `bfloat16_t` 加载到 UB |
| MulAddDst 支持 `<T, U>` 双类型模板 | dst(T) 与 src0/src1(U) 可不同类型 |
| Cast 支持任意 `T→U` 类型对 | `RoundMode::CAST_NONE` 适用于扩展精度（无舍入损失） |
| DataCopy UB内BF16→FP32 仅限 `__NPU_ARCH__ == 2201` | 需 `#if` 条件编译保护 |

### 9.3 BF16 处理路径选择

两种可行路径：

- **路径 A**（优先）: DataCopy 内转换 — `DataCopy<float, bfloat16_t>(fp32Buf, bf16Buf, params, enhancedParams)` — 仅 DAV_2201 支持，性能更优
- **路径 B**（兜底）: Cast 显式转换 — `Cast<float, bfloat16_t>(fp32Buf, bf16Buf, RoundMode::CAST_NONE, count)` — 通用支持

## 10. Tiling 计算

### 10.1 Host 侧 TilingData

```cpp
struct EngramGateWReduceTiling {
    uint32_t blockDim;        // 使用的核数
    uint32_t totalHiddenSize; // hidden_size（用户传入）
    uint32_t tileHiddenLen;   // 每核处理的 hidden 段长度
    uint32_t tileA0Len;       // 每核处理的 A0 长度 = tileHiddenLen * 4
    uint32_t R;               // 归约维度大小 = 108（固定）
};
```

### 10.2 tileA0Len 计算

```
tileHiddenLen = ceil(hidden_size / blockDim)
tileA0Len = tileHiddenLen * 4
```

约束:
- `tileA0Len * 24 B ≤ 192 KB` → `tileA0Len ≤ 8192` → 始终满足（tileA0Len ≤ 4 * hidden_size，hidden_size 实际场景远小于 2048 * blockDim）

## 11. 精度策略

### 11.1 精度分析

| 计算步骤 | 类型 | 精度风险 | 策略 |
|----------|------|---------|------|
| ReduceSum (108 rows) | FP32 sum | 累加 108 次，相对误差 ~1e-7 量级 | 顺序累加，风险极低 |
| BF16→FP32 Cast | 类型扩展 | 无精度损失 | 使用硬件转换 |
| Mul (FP32 * FP32) | FP32 乘法 | 标准 FP32 误差 | 使用原生 MulAddDst |
| Accumulate (FP32 += FP32) | FP32 加法 | 标准 FP32 误差 | 使用原生 MulAddDst |

### 11.2 混合精度策略

- 输入 weight_hidden/weight_embed 为 BF16，在计算前转换为 FP32
- 中间计算全在 FP32 下进行
- 输出 (grad_weight_hidden, grad_weight_embed) 保持 FP32

### 11.3 精度标准

本算子属于**浮点计算类社区标准**（用户未明确要求商用标准）。验收策略：
- 与 PyTorch 标杆（fp32 全精度）对比，相对误差容限 `1e-4`（考虑 BF16→FP32 转换的固有误差 + 累加传播误差）

## 12. 分支场景覆盖

| 分支维度 | 策略 |
|----------|------|
| **hidden_size 变化** | 自适应 tiling：tileHiddenLen = ceil(hidden_size / blockDim)，UB 容量自适应 |
| **hidden_size 极小** (< 4) | blockDim 缩减到 min(coreNum, hidden_size)，避免空核 |
| **hidden_size 极大** (O(10K+)) | tileA0Len 增长但 UB 仍充裕（tileA0Len ≤ 4*16K/24 ≈ 2730, UB=65KB） |
| **R = 108 (固定)** | 硬编码 R 值，不做分 chunk 处理 |
| **dtype (固定)** | 输入类型按需求固定：FP32 grads + BF16 weights，输出 FP32 |
| **数据对齐** | 所有 GM tensor 连续存储；UB buffer 通过 DataCopy 自动处理对齐 |

## 13. 非功能性设计

### 13.1 In-Place 语义

- `grad_weight_hidden` 和 `grad_weight_embed` 既是输入也是输出
- Phase 2 先 Load 到 UB，原地修改后 Store 回同一 GM 地址
- 保证无数据竞争（每个 core 处理不相交的 hidden_size 段）

### 13.2 内存复用

- Phase 1 的 pingBuf/pongBuf 在 Phase 2 被复用为 BF16 加载 buffer 和部分 FP32 buffer
- accumBuf 跨两阶段复用（Phase 1 输出 = Phase 2 输入 grad_w_sum）

### 13.3 流水线

- Phase 1 使用 Double Buffer (`SetQueue` / `EnQue` / `DeQue`) 实现 GM 读取与 Vector Add 计算重叠
- Phase 2 使用顺序执行（数据量小，流水线收益有限）
