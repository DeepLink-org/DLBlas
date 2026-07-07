# engram_gate_bwd 反向算子架构设计

> **版本**: v1.0 | **日期**: 2026-07-06 | **设计者**: Architect

---

## 1. 算子需求分析

### 1.1 数学定义

`engram_gate_bwd` 是 Engram Gate 机制的反向传播算子，接收上游梯度 `grad_out` 和前向输入 `x, k, v, wh, we`，输出 5 个梯度：`grad_x, grad_k, grad_v, grad_wh, grad_we`。

算子包含两个阶段：

**Phase A -- 前向中间量重算（从 x, k, v, wh, we 推导）**

```
rstd_x  = rsqrt(mean(x^2,  dim=-1) + eps)        # (T, H)
rstd_k  = rsqrt(mean(k^2,  dim=-1) + eps)        # (T, H)
raw_dot = sum(x * wh * (k * we), dim=-1)          # (T, H)
dot     = raw_dot * rstd_x * rstd_k * scalar       # (T, H)  , scalar = D^{-0.5}
s_sqrt  = |dot|.clamp_min(cv).sqrt() * sign(dot)   # (T, H)
gate    = sigmoid(s_sqrt)                           # (T, H)
```

**Phase B -- 反向梯度计算（使用 go 和 Phase A 中间量）**

```
grad_v      = sum_{h}(go[t,h,d] * gate[t,h])                           # (T, D), reduce H
grad_gate   = sum_{d}(go[t,h,d] * v[t,d])                              # (T, H), reduce D
grad_s_sqrt = grad_gate * gate * (1 - gate)                            # (T, H), elementwise
mask        = (|dot| >= cv).to(f32)
grad_dot    = grad_s_sqrt * mask * 0.5 / sqrt(|dot|.clamp_min(cv))      # (T, H), elementwise
grad_raw_dot= grad_dot * rstd_x * rstd_k * scalar                       # (T, H), elementwise
grad_rstd_x = grad_dot * raw_dot * rstd_k * scalar                      # (T, H), elementwise
grad_rstd_k = grad_dot * raw_dot * rstd_x * scalar                      # (T, H), elementwise
grad_x      = go + grad_raw_dot * wh * (k*we) + grad_rstd_x * (-x/D) * rstd_x^3  # (T,H,D), broadcast+elementwise
grad_k      = grad_raw_dot * we * (x*wh) + grad_rstd_k * (-k/D) * rstd_k^3        # (T,H,D), broadcast+elementwise
grad_wh     = sum_{t}(grad_raw_dot[t,h] * k[t,h,d] * we[h,d] * x[t,h,d])           # (H,D), reduce T
grad_we     = sum_{t}(grad_raw_dot[t,h] * x[t,h,d] * wh[h,d] * k[t,h,d])           # (H,D), reduce T
```

### 1.2 输入输出规格

| 张量 | Shape | Dtype | 角色 |
|------|-------|-------|------|
| `grad_out` (go) | (T, H, D) | bf16 | 上游梯度输入 |
| `x_data` | (T, H, D) | bf16 | 前向 x 缓存 |
| `k_data` | (T, H, D) | bf16 | 前向 k 缓存 |
| `v_data` | (T, D) | bf16 | 前向 v 缓存 |
| `wh_data` | (H, D) | bf16 | 前向 wh 权重 |
| `we_data` | (H, D) | bf16 | 前向 we 权重 |
| `clamp_value` | scalar | f32 | clamp 下限 (典型 1e-6) |
| `eps` | scalar | f32 | 数值稳定常数 (典型 1e-20) |

| 输出张量 | Shape | Dtype | 说明 |
|----------|-------|-------|------|
| `grad_x` | (T, H, D) | bf16 | x 的梯度 |
| `grad_k` | (T, H, D) | bf16 | k 的梯度 |
| `grad_v` | (T, D) | bf16 | v 的梯度 |
| `grad_wh` | (H, D) | bf16 | wh 的梯度 |
| `grad_we` | (H, D) | bf16 | we 的梯度 |

**典型 shape**: T=14, H=4, D=128（可泛化到任意 T/H/D）

### 1.3 计算精度要求

- 输入/输出: `bfloat16`
- 内部计算: `float32`（bf16 输入加载时 Cast 到 f32，所有中间计算在 f32 下进行，输出时 Cast 回 bf16）
- 数值稳定: eps=1e-20 防止除零，clamp_value=1e-6 保证 sqrt 安全域

---

## 2. 架构选择

### 2.1 目标平台确认

| 参数 | 值 | 来源 |
|------|-----|------|
| 芯片型号 | Ascend 910B2 | 需求文档 |
| NpuArch | `DAV_2201` | `/npu-arch` skill 查表 |
| `__NPU_ARCH__` | `2201` | 编译宏 |
| `--npu-arch` | `dav_2201_vec` | Vector 算子编译参数 |
| UB 容量 | 192 KB | 硬件参数（npu-arch skill） |
| L0C 容量 | 128 KB | 硬件参数 |
| AI Core 数 | 24 | Ascend 910B2 规格 |
| CANN 版本 | 9.0.0 | 需求文档 |

### 2.2 技术路线决策

**决策流程（按设计规范 Step 0.5）**：

1. 芯片型号 Ascend 910B2 → NpuArch = `DAV_2201`
2. 算子类型判断：
   - 核心操作：多次归约（reduce over D, reduce over H, reduce over T）+ 大量逐元素运算 + 少量广播
   - 不含 MatMul/Cube 运算
   - **分类：Reduction + Elementwise + Broadcast 混合融合算子**
3. 路线判定：
   - `DAV_2201` **不是** `DAV_3510` → **走通用 SIMD/MemBase 路线**
   - 不加载 RegBase/Blaze best-practice（仅 DAV_3510 适用）

**选择理由**：
- 目标架构 DAV_2201 不支持 RegBase 编程模型和 Blaze tensor_api 路线
- 使用标准 Ascend C SIMD/MemBase API（ReduceSum、Mul、Add、Div、Cast 等）
- 充分利用 Vector 引擎的归约和逐元素计算能力

### 2.3 单核 vs 多核决策

**决策：多核并行，沿 T 维度切分。**

理由：
- T 是可伸缩的序列长度维度（典型 14，可泛化到上千），天然适合并行
- H 通常较小（4），不适合切分；D（128）是归约轴，切分会引入复杂的分段归约
- T 维度切分后，每个核的计算独立性强，仅 `grad_wh` 和 `grad_we`（T 轴归约）需要跨核合并

### 2.4 多核切分策略

```
总任务切分方式：按 T 维度均分

tileT  = ceil(totalT / coreNum)
核 i 处理 T 范围：[i * tileT, min((i+1) * tileT, totalT))

每个核独立的子任务：
  - 加载 wh_data, we_data（全量，H*D 很小）
  - 加载本核负责的 x[T_slice], k[T_slice], v[T_slice], go[T_slice]
  - 计算本核的 grad_x[T_slice], grad_k[T_slice], grad_v[T_slice]
  - 累加本核的 partial_grad_wh, partial_grad_we

跨核归约（grad_wh, grad_we）：
  - 每个核将 partial 写入 Workspace
  - SyncAll() 同步
  - 核 0 读取所有 partial 求和，写出最终 grad_wh, grad_we
```

---

## 3. Tiling 设计

### 3.1 维度分析（合轴）

计算中的维度角色：

| 原始维度 | 角色 | 合轴后 |
|---------|------|--------|
| T | 外层循环 / 多核切分轴 | A1 (保留轴) |
| H | 部分场景的归约轴 / 广播轴 | 视具体操作而定 |
| D | 最内归约轴 (大多数 reduction) | R (归约轴) |

**关键归约分析**：

| 归约操作 | 输入 shape | 归约轴 | 合轴后 | 模式 |
|---------|-----------|--------|--------|------|
| rstd_x: mean(x^2, -1) + rsqrt | (T, H, D) | D (axis=-1) | (T*H, D) → AR | AR-FullLoad (D≤128 时整行放入 UB) |
| rstd_k: mean(k^2, -1) + rsqrt | (T, H, D) | D (axis=-1) | 同上 | AR-FullLoad |
| raw_dot: sum(x*wh*k*we, -1) | (T, H, D) | D (axis=-1) | 同上 | AR-FullLoad |
| grad_gate: sum(go*v, -1) | (T, H, D) | D (axis=-1) | 同上 | AR-FullLoad |
| grad_v: sum(go*gate, 1) | (T, H, D) | H (axis=1) | (T*D, H) → AR | AR-FullLoad (H≤4 时极短) |
| grad_wh: sum(..., 0) | (T, H, D) | T | 跨核归约 | Group Reduce (跨核) |
| grad_we: sum(..., 0) | (T, H, D) | T | 跨核归约 | Group Reduce (跨核) |

对于 D 轴归约（rLength = D）：
- 若 `D * sizeof(f32) ≤ UB 容量` 且 `D 足够小使整行放入 UB`：使用 **AR-FullLoad**
- 若 D 过大无法整行放入 UB：使用 **AR-ColSplit**（分段归约 + 跨段合并）

D 对齐值:
```
D_align = ((D * sizeof(f32) + 31) / 32) * 32 / sizeof(f32)
```
对于 D=128：`D_align = 128`（已对齐）。

### 3.2 UB 切分策略

每个核在 UB 内按批次处理 T 元素：

```
tileTPerLoop = min(tileT, maxBatchT)

maxBatchT = floor((UB_SIZE - constBufSize - accumBufSize - tmpBufSize) / perTDataSize)
```

**UB 常量区（所有 T batch 共享）**：
| Buffer | 大小 | 说明 |
|--------|------|------|
| wh_f32 | H * D * 4B = 2KB | 权重 wh，bf16→f32 后常驻 |
| we_f32 | H * D * 4B = 2KB | 权重 we，bf16→f32 后常驻 |
| wh_we_f32 | H * D * 4B = 2KB | 预计算 wh⊙we，复用 |

**UB 每 T 元素数据（f32）**：
| Buffer | 大小 | 用途 |
|--------|------|------|
| x_f32 / k_f32 / go_f32 | H*D*4B = 2KB each | 3D Tensor 的一层 (H, D) |
| v_f32 | D*4B = 512B | v 向量的一层 (D,) |
| work1_f32 | H*D*4B = 2KB | 临时工作区 (x*wh, k*we 等) |
| work2_f32 | H*D*4B = 2KB | 临时工作区 |
| grad_x_f32 / grad_k_f32 | H*D*4B = 2KB each | 输出梯度 |
| grad_v_f32 | D*4B = 512B | 输出梯度 |

**UB 标量/小数组（每 batch 所有 T 共享，合并为一个 buffer）**：
| Buffer | 大小 | 说明 |
|--------|------|------|
| scalar_buf | tileTPerLoop * H * N_SCALARS * 4B | rstd_x, rstd_k, raw_dot, dot, s_sqrt, gate, grad_gate, grad_s_sqrt, grad_dot, grad_raw_dot, grad_rstd_x, grad_rstd_k, mask 共 12 个标量组 → tileTPerLoop * H * 12 * 4B |
| reduce_tmp | ~4KB | ReduceSum tmpBuf (取 maxTmpSize) |
| accum_grad_wh | H*D*4B = 2KB | 跨 T 累加器 |
| accum_grad_we | H*D*4B = 2KB | 跨 T 累加器 |

**容量估算（典型 shape: H=4, D=128, tileTPerLoop=4）**：

| 区域 | 计算 | 大小 |
|------|------|------|
| 常量 (wh, we, wh_we) | 3 × 2KB | 6 KB |
| 输入 (x, k, go, v) × 4 T | 4 × (3×2KB + 0.5KB) | 26 KB |
| 工作区 (work1, work2) × 4 T | 4 × 4KB | 16 KB |
| 输出 (grad_x, grad_k, grad_v) × 4 T | 4 × (2×2KB + 0.5KB) | 18 KB |
| 标量数组 | 4 × 4 × 12 × 4B | ~0.8 KB |
| 归约 tmp | 4KB | 4 KB |
| 累加器 (grad_wh, grad_we) | 2 × 2KB | 4 KB |
| **合计** | | **~75 KB** |

192KB UB 容量下，留有余量用于对齐 padding 和 double buffer。

### 3.3 Double Buffer 策略

采用输入侧双缓冲：为 (x, k, go, v) 的当前 batch 和下一 batch 各准备一份。

```
双缓冲 per T 增加: 6.5KB → 双缓冲合计增加约 26KB
总 UB 使用: 75KB + 26KB ≈ 101KB → 充裕 (192KB)
```

当 D 增大时，tileTPerLoop 自动缩减以保证 UB 不溢出。

### 3.4 分支场景覆盖

| 分支维度 | 条件 | 策略 |
|---------|------|------|
| D 大小 | D ≤ D_threshold (UB可容纳完整归约行) | AR-FullLoad，整行归约 |
| D 大小 | D > D_threshold | AR-ColSplit，分段归约+合并 |
| H 大小 | H ≤ 8 | 直接逐行处理或小批量 AR |
| H 大小 | H > 8 | 按 ARA 模式处理 H 维归约 |
| T 大小 | T ≤ coreNum | 核数 > T 时，减少使用核数，至少每核处理 1 个 T |
| T 大小 | T 很大 | tileTPerLoop 由 UB 预算公式自动确定 |
| 边界处理 | 尾 batch T 数 < tileTPerLoop | 尾 batch 特殊处理，循环次数递减 |

---

## 4. 数据流设计

### 4.1 整体数据流

```
┌────────────────────────────────────────────────────────────┐
│                        Host (Tiling)                        │
│  1. 计算 tileT, tileTPerLoop, workspace 大小                │
│  2. 分配 GM workspace (coreNum × 2 × H × D × 4B)           │
│  3. 下发 tiling 参数到 device                               │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                Device (每核 AI Core)                         │
│                                                              │
│  ┌───────────────────────────────────────────────┐          │
│  │ Step 0: 初始化                                  │          │
│  │  - wh, we 从 GM (bf16) → Cast → UB (f32, 常驻) │          │
│  │  - 重置 accum_grad_wh, accum_grad_we = 0       │          │
│  └───────────────────────────────────────────────┘          │
│                       │                                      │
│                       ▼                                      │
│  ┌───────────────────────────────────────────────┐          │
│  │ for t_batch in [0, tileT) step tileTPerLoop:    │          │
│  │                                                  │          │
│  │  ┌─────────────────────────────────────────┐    │          │
│  │  │ DMA: 预取下一 batch 输入 (double buffer)  │    │          │
│  │  └─────────────────────────────────────────┘    │          │
│  │                                                  │          │
│  │  ┌─────────────────────────────────────────┐    │          │
│  │  │ Phase A: 前向重算 (x, k, wh, we, scalar) │    │          │
│  │  │  A1. Cast x,k bf16→f32                   │    │          │
│  │  │  A2. ReduceSum(x^2, dim=-1) → rstd_x      │    │          │
│  │  │  A3. ReduceSum(k^2, dim=-1) → rstd_k      │    │          │
│  │  │  A4. Mul(wh, we) → wh_we (常量预计算)     │    │          │
│  │  │  A5. ReduceSum(x*wh * k*we, dim=-1)       │    │          │
│  │  │      → raw_dot (逐T)                       │    │          │
│  │  │  A6. dot = raw_dot * rstd_x * rstd_k       │    │          │
│  │  │      * scalar                              │    │          │
│  │  │  A7. s_sqrt = sign(dot)*sqrt(|dot|.clamp)  │    │          │
│  │  │  A8. gate = sigmoid(s_sqrt)                │    │          │
│  │  └─────────────────────────────────────────┘    │          │
│  │                       │                          │          │
│  │                       ▼                          │          │
│  │  ┌─────────────────────────────────────────┐    │          │
│  │  │ Phase B: 反向梯度 (go, v, 中间量)        │    │          │
│  │  │  B1. Cast go,v bf16→f32                   │    │          │
│  │  │  B2. ReduceSum(go·v, dim=-1) → grad_gate  │    │          │
│  │  │  B3. grad_s_sqrt = grad_gate·gate·(1-gate)│    │          │
│  │  │  B4. mask→grad_dot 链 (elementwise)       │    │          │
│  │  │  B5. grad_raw_dot, grad_rstd_x, grad_rstd_k│   │          │
│  │  │  B6. Mul+Broadcast: grad_x (3路径求和)     │    │          │
│  │  │  B7. Mul+Broadcast: grad_k (2路径求和)     │    │          │
│  │  │  B8. ReduceSum(go·gate, dim=1) → grad_v    │    │          │
│  │  │  B9. 累加 partial_grad_wh, partial_grad_we │    │          │
│  │  └─────────────────────────────────────────┘    │          │
│  │                                                  │          │
│  │  ┌─────────────────────────────────────────┐    │          │
│  │  │ Store: grad_x, grad_k, grad_v f32→bf16   │    │          │
│  │  │        → GM                               │    │          │
│  │  └─────────────────────────────────────────┘    │          │
│  └──────────────────────────────────────────────────┘          │
│                       │                                      │
│                       ▼                                      │
│  ┌───────────────────────────────────────────────┐          │
│  │ Cross-Core Reduction (grad_wh, grad_we)        │          │
│  │  1. 各核将 accum_grad_wh/we → workspace[i]     │          │
│  │  2. SyncAll()                                   │          │
│  │  3. 核 0: 读取所有核的 partial，求和             │          │
│  │  4. 核 0: Cast f32→bf16 写出 grad_wh, grad_we  │          │
│  └───────────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────────┘
```

### 4.2 GM 内存规划

| 区域 | 大小 | 说明 |
|------|------|------|
| 输入 go | T * H * D * 2B | bf16 |
| 输入 x | T * H * D * 2B | bf16 |
| 输入 k | T * H * D * 2B | bf16 |
| 输入 v | T * D * 2B | bf16 |
| 输入 wh | H * D * 2B | bf16 |
| 输入 we | H * D * 2B | bf16 |
| 输出 grad_x | T * H * D * 2B | bf16 |
| 输出 grad_k | T * H * D * 2B | bf16 |
| 输出 grad_v | T * D * 2B | bf16 |
| 输出 grad_wh | H * D * 2B | bf16 |
| 输出 grad_we | H * D * 2B | bf16 |
| Workspace | coreNum * 2 * H * D * 4B | f32, 跨核归约中间量 |

### 4.3 跨核归约详细流程

```
Phase 1 (各核并行):
  for t in t_slice:
      partial_grad_wh += grad_raw_dot[t,:,None] * k[t,:,:] * we[:,:] * x[t,:,:]
      partial_grad_we += grad_raw_dot[t,:,None] * x[t,:,:] * wh[:,:] * k[t,:,:]
  DataCopy(partial_grad_wh_f32 → workspace[coreIdx * 2 * H * D * 4])
  DataCopy(partial_grad_we_f32 → workspace[(coreIdx * 2 + 1) * H * D * 4])

Phase 2 (SyncAll 后):
  if coreIdx == 0:
      grad_wh_f32 = zeros(H, D)
      grad_we_f32 = zeros(H, D)
      for i in [0, coreNum):
          grad_wh_f32 += workspace[i * 2 * H * D * 4]
          grad_we_f32 += workspace[(i * 2 + 1) * H * D * 4]
      Cast(bf16): grad_wh_f32 → output, grad_we_f32 → output
```

对于 Workspace 空间：核 0 读取时可能需分批，避免超出 UB 容量。由于 grad_wh 仅 H*D 个元素（典型 512），可以一次性全部加载到 UB。

---

## 5. API 映射表

| 计算操作 | Ascend C API | 模式 | 说明 |
|---------|-------------|------|------|
| bf16→f32 Cast | `Cast<float, bfloat16_t>(dst, src, CAST_NONE, count)` | MemBase | 输入 Cast |
| f32→bf16 Cast | `Cast<bfloat16_t, float>(dst, src, CAST_ROUND, count)` | MemBase | 输出 Cast |
| 逐元素乘法 | `Mul(dst, src0, src1, count)` | MemBase | H*D 元素级乘法 |
| 标量乘法 (广播) | `Muls(dst, src, scalar, count)` | MemBase | 单值广播到数组 |
| 逐元素加法 | `Add(dst, src0, src1, count)` | MemBase | 多路径求和 |
| 逐元素除法 | `Div(dst, src0, src1, count)` | MemBase | 元素级除法 |
| 标量除法 | `Div(dst, src, scalar_broadcast, count)` | MemBase | 或 Muls(dst, src, 1/scalar) |
| 平方 | `Mul(dst, src, src, count)` | MemBase | 自乘 |
| abs | `Abs(dst, src, count)` | MemBase | 绝对值 |
| sqrt | `Sqrt(dst, src, count)` | MemBase | 平方根 |
| rsqrt | `Rsqrt(dst, src, count)` | MemBase | 倒数平方根 |
| sigmoid | `Sigmoid(dst, src, count)` | MemBase | 1/(1+exp(-x)) |
| sign | 用 Compare + Select | MemBase | 取符号 |
| clamp_min | `Maximum(dst, src, clamp_val, count)` | MemBase | 逐元素取 max |
| ReduceSum (沿 D) | `ReduceSum<float>(dst, src, tmp, rLength)` | Level 2 AR | 逐行归约 D 轴 |
| ReduceSum (沿 H) | `ReduceSum<float>(dst, src, tmp, rLength)` | Level 2 AR | 逐行归约 H 轴（重塑为 (T*D, H)） |
| Broadcast (T,H)→(T,H,D) | `Duplicate` + `Mul` 或 `BinaryRepeatParams` | MemBase | 标量广播到 (H, D) |
| Broadcast v (T,D)→(T,H,D) | `Duplicate` + `Mul` | MemBase | 沿 H 复制 |
| Cross-core reduce | `DataCopy` + `SyncAll` + 手动合并 | GM | Workspace 通信 |
| 双缓冲输入搬运 | `DataCopy` (对齐) / `DataCopyPad` (非对齐) | MemBase | GM→UB DMA |

### 5.1 关键 API 参数验证要点

| API | 验证项 |
|-----|--------|
| `Cast<DstT, SrcT>()` | RoundMode: CAST_NONE (低→高), CAST_ROUND (高→低) |
| `ReduceSum<T>()` | tmpBuffer 类型必须与 T 一致；Level 2 接口无对齐要求 |
| `Mul/Add/Div` | 参与计算的 Tensor 地址需 32B 对齐 |
| `DataCopy` | 仅 32B 对齐时使用，非对齐用 DataCopyPad |
| `Maximum` | 对标量 clamp 使用 scalar 参数版本 |
| `Sigmoid` | 确认在 f32 类型下可用（DAV_2201 支持） |

---

## 6. 精度策略

### 6.1 整体精度路径

```
bf16 Input → Cast(CAST_NONE) → f32 compute → Cast(CAST_ROUND) → bf16 Output
```

### 6.2 数值稳定性保护

| 风险点 | 保护措施 |
|--------|---------|
| `rsqrt(mean + eps)` 中 mean 为 0 | eps=1e-20 保证非零分母 |
| `sqrt(|dot|.clamp_min(cv))` 中 dot 为 0 | clamp_value=1e-6 保证 sqrt 安全域 |
| `0.5 / sqrt(|dot|.clamp_min(cv))` 中分母极值 | clamp 保证 sqrt ≥ 1e-3，分母不会爆炸 |
| `sigmoid` 输入极大正值 | sigmoid 在 f32 下稳定，1/(1+exp(-x)) 不会溢出 |
| 多路径求和累积误差 (grad_x 有三条路径) | 在 f32 下累加，误差远小于 bf16 直算 |
| ReduceSum 大向量精度 | D 通常 ≤ 4096，f32 下有足够精度；若 D 极大可启用二分累加 |

### 6.3 精度标准

按 `ops-precision-standard` 判定：浮点计算类、bf16 混合精度场景。采用社区标准（与 PyTorch bf16 参考实现对比）。

**验收标准**：
- 以 PyTorch 全 f32 计算结果为 Golden
- Ascend C 算子实现（bf16 输入→f32 内部→bf16 输出）与 Golden 对比
- bf16 精度：相对误差 ≤ 1e-2（bf16 精度极限约 3 位有效数字）
- f32 精度路径确保中间累积误差最小化

---

## 7. 关键技术点分析

### 7.1 D 轴归约（RMS Norm 类计算）

`rstd_x = rsqrt(mean(x^2, dim=-1) + eps)` 和 `rstd_k` 是典型的 RMS Norm 模式：

```
每行 (H, D) → ReduceSum(x^2, axis=-1) → /D → +eps → rsqrt → (H,)
```

- 使用 Level 2 `ReduceSum` API（AR 模式），逐行处理
- 对于 D=128，配合 T*H=56 行（T=14, H=4），所有行完成约需 56 次 ReduceSum 调用
- 若 D 很大需分载（AR-ColSplit），使用分段 ReduceSum + 跨段 Add 合并

### 7.2 H 轴归约

`grad_v = sum(go * gate, dim=1)`：沿 H 轴归约。

- 将 (T, H, D) 重塑为 (T*D, H)，每行 H 个元素
- 典型 H=4，归约极短，Level 2 API 直接完成
- 若 H 较大（如 64），可考虑 ARA 模式批处理

### 7.3 T 轴跨核归约

`grad_wh` 和 `grad_we` 沿 T 轴求和，且 T 维度已做多核切分。

- 使用 Workspace + SyncAll 的两阶段 Group Reduce
- Workspace 大小: `coreNum * 2 * H * D * sizeof(f32)`
- 核 0 合并阶段需确保 Workspace 分批加载不超出 UB

### 7.4 Broadcast 操作

多个操作涉及从 (T, H) 广播到 (T, H, D)：

- `gate[T, H]` 广播参与 `grad_v = sum(go * gate[:,:,None], dim=1)`
- `grad_raw_dot[T, H]` 广播参与 `grad_x` 和 `grad_k` 计算

在 UB 内实现方式：
- 对于小 H（4），使用 `Duplicate` 将第 3 维复制 H 次，或直接在循环中处理每个 H
- 使用 `BinaryRepeatParams` 配合 `src1RepStride=0` 实现高效广播

### 7.5 常量预计算（wh_we）

`wh * we` 在 T 循环中多次使用（raw_dot、grad_x、grad_k、grad_wh、grad_we 各式中均出现）。

优化：在核初始化时预计算 `wh_we = wh ⊙ we`（逐元素乘），作为常量驻留 UB，避免每次 T batch 重复计算。

### 7.6 依赖关系与计算顺序保证

Phase A（前向重算）必须在 Phase B（反向梯度）之前完成：

```
依赖链:
  x,k → rstd_x,rstd_k ─┐
  x,k,wh,we → raw_dot ─┤
                        ├→ dot → s_sqrt → gate ─→ grad_v, grad_s_sqrt
                                           go,v ─→ grad_gate ─┘
  gate,grad_gate → grad_s_sqrt → grad_dot → grad_raw_dot,grad_rstd_x,grad_rstd_k
  grad_raw_dot,grad_rstd_x → grad_x, grad_k
  grad_raw_dot → partial_grad_wh, partial_grad_we
```

kernel 内自然按照 Phase A → Phase B 的顺序保证。

---

## 8. Tiling 参数定义

```cpp
struct EngramGateBwdTiling {
    // 维度参数
    uint64_t totalT;          // 总序列长度
    uint64_t totalH;          // 头数
    uint64_t totalD;          // 隐层维度
    uint64_t D_align;        // D 的 32B 对齐值 (f32 下)
    
    // 多核切分
    uint64_t tileT;           // 每核 T 元素数 = ceil(totalT / coreNum)
    uint64_t coreNum;         // 实际使用核数
    uint64_t coreIdx;         // 当前核索引
    
    // UB 切分
    uint64_t tileTPerLoop;    // 每轮 UB 处理的 T 元素数
    uint64_t tailTPerLoop;    // 尾轮 T 元素数 (可能 < tileTPerLoop)
    uint64_t loopCount;       // T 循环次数 = ceil(tileT / tileTPerLoop)
    
    // 常量
    float clampValue;         // clamp 下限
    float eps;                // 数值稳定常数
    float scalar;             // D^{-0.5}
    float invD;               // 1.0 / D
    float half;               // 0.5 (grad_dot 公式)
    float one;                // 1.0
    
    // Workspace
    uint64_t workspaceSize;   // coreNum * 2 * H * D * sizeof(f32)
    uint64_t workspaceOffset; // 当前核 partial 在 workspace 中的偏移
};
```

---

## 9. 约束与限制

| 约束 | 说明 |
|------|------|
| D 必须 > 0 | 隐层维度不能为 0 |
| H 必须 > 0 | 头数不能为 0 |
| T 必须 > 0 | 序列长度不能为 0 |
| clamp_value > 0 | 保证 sqrt 域有效 |
| eps ≥ 0 | rsqrt 输入非负即可 |
| bf16 输入 | 仅支持 bf16 输入数据类型（需求指定） |
| D ≥ 32 | 建议 D 至少为 32 以保证 Vector 指令效率 |
| 不使用 Host 侧预处理 | 所有计算在 Device 侧完成 |
