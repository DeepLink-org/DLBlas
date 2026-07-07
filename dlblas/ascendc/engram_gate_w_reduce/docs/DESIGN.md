# engram_gate_w_reduce 算子架构设计

## 1. 算子概述

### 1.1 数学定义

```
grad_w_sum        = sum(grad_w_partial, dim=0)                    // [108, 4, 4096] -> [4, 4096]
grad_weight_hidden = grad_weight_hidden + grad_w_sum * weight_embed.float()   // [4, 4096]
grad_weight_embed  = grad_weight_embed  + grad_w_sum * weight_hidden.float()   // [4, 4096]
```

### 1.2 输入输出规格

| 张量 | Shape | dtype | 角色 |
|------|-------|-------|------|
| `grad_w_partial` | [108, 4, 4096] | float32 | 输入，归约源 |
| `weight_hidden` | [4, 4096] | bfloat16 | 输入，逐元素乘法 |
| `weight_embed` | [4, 4096] | bfloat16 | 输入，逐元素乘法 |
| `grad_weight_hidden` | [4, 4096] | float32 | 输入(累加器)/输出 |
| `grad_weight_embed` | [4, 4096] | float32 | 输入(累加器)/输出 |

### 1.3 算子类型分类

**Hybrid: Reduction (axis=0) + Broadcast Element-wise (Mul+Add)**

- 主计算: 沿 axis=0 归约求和 (Reduction)
- 后处理: 逐元素乘法 + 原位加法 (Broadcast Element-wise)

---

## 2. 架构与环境

| 参数 | 值 | 来源 |
|------|-----|------|
| **芯片型号** | Ascend 910B2 | environment.md |
| **NpuArch** | `DAV_2201` | `/npu-arch` skill (SocVersion -> Arch lookup) |
| **`__NPU_ARCH__`** | `2201` | 条件编译宏 |
| **`--npu-arch`** | `daVinci2201_vec` | 编译参数 (vector 算子) |
| **CANN** | 9.0.0 | environment.md |
| **UB 容量** | 192 KB (196608 B) | DAV_2201 参数 |
| **L0C 容量** | 128 KB | DAV_2201 参数 |
| **Cube 核数** | 24 | Ascend 910B2 |

---

## 3. 技术路线决策

### 3.1 决策流程

```
算子类型: Vector (Reduction + Element-wise)
    |
    NpuArch == DAV_3510?
    ├─ NO → 通用 SIMD/MemBase 路线  ← 选定
    └─ YES → RegBase 路线
```

### 3.2 决策理由

1. **NpuArch = DAV_2201**: 不支持 DAV_3510 的 RegBase / Blaze 新架构能力。
2. **算子类型为 Vector 类**: Reduction + Element-wise 组合，使用 Vector (SIMD) 指令集。
3. **通用 SIMD/MemBase 路线**: 使用标准 `AscendC::` API (DataCopyPad, Add, Mul, Cast, Duplicate)，无 RegBase 依赖。

### 3.3 使用的 Skill 资源

- `/ascendc-tiling-design` -- Reduction ARA 模式设计方法论（合轴、Buffer 规划、多核切分）
- `/ascendc-api-best-practices` -- Cast 精度转换、算术运算 API 使用规范
- `/ops-precision-standard` -- 浮点计算社区标准（MERE/MARE 阈值）

---

## 4. 合轴与 Shape 分析

### 4.1 合轴过程

```
原始 shape = [108, 4, 4096], axes = [0]

标记 A/R:
  - dim 0 (108): R (归约轴)
  - dim 1 (4):   A (保留轴)
  - dim 2 (4096): A (保留轴)

合并相邻同类型轴:
  - 相邻 A 合并: 4 * 4096 = 16384
  - 归约后 shape: [108, 16384] = (R, A0)
  
模式判定: A0 = 16384 > 1 → ARA 模式
```

### 4.2 ARA 模式评估

经典 ARA-FullLoad 需要将所有 R=108 行数据同时放入 UB。以 `a0TileBase=64` 计算:
- 双缓冲输入: 2 * 108 * 64 * 4 = 55296 B
- 双缓冲输出: 2 * 64 * 4 = 512 B
- tmpBuf: ~4096 B
- 每 64 元素消耗约 60 KB UB

这将导致 tileA0Len 最大约 192 个元素（16384/192 ≈ 86 tiles），tile 数量过多。

### 4.3 方案选择: 手动迭代归约（Custom Iterative Reduction）

不使用 `Pattern::Reduce::RA`，改为手动逐行加载+累加:

```
accum[tileElems] = 0
for i in 0..107:
    load row_i[tile] -> row_buf[tileElems]
    accum += row_buf
```

**优势**:
- 只需 `accum + row_buf` 两个 buffer，无须同时驻留 108 行
- tileElems 可以更大，减少 tile 数量
- 后处理可直接复用 row_buf 空间

---

## 5. Tiling 策略

### 5.1 切分维度

仅沿 hidden_dim (4096) 切分。每个 tile 覆盖全部 4 个 channel，处理 tileHidLen 个 hidden 元素。

```
tileElems = 4 * tileHidLen  (4 channels × hidden tile)
```

每个 tile 在 row 内的映射:
```
row[i, :, tileHidStart:tileHidStart+tileHidLen]
  = [ch0[tileHidStart:...], ch1[tileHidStart:...], ch2[...], ch3[...]]
```

### 5.2 Tiling 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `tileHidLen` | 512 (默认) | 每个 channel 内的 hidden tile 长度 |
| `tileElems` | 2048 | 每个 tile 的总元素数 (4 * tileHidLen) |
| `numTilesPerDim` | 8 | 4096 / tileHidLen |
| `totalTiles` | 8 | numTilesPerDim (channel 全部包含) |
| `numCores` | 8 (默认) | 每个 core 处理 1 个 tile |

### 5.3 多核切分

```
总工作量: [4, 4096] = 16384 输出元素
每核处理: tileElems = 2048 输出元素
核数: totalTiles = 8

负载均衡: 所有 tile 大小相等 → 完美均衡
核间通信: 无需通信 (每个 tile 完全独立)
```

可调参数: 若需要更多核，可将 `tileHidLen` 减小到 256 (16 核)。

---

## 6. UB Buffer 规划

### 6.1 Buffer 列表 (per core)

| Buffer | 类型 | 大小 | 说明 |
|--------|------|------|------|
| `accum` | TBuf\<float\> | `tileElems * 4 = 8 KB` | 归约累加器，持有 grad_w_sum 结果 |
| `rowBuf0` | TBuf\<float\> | `tileElems * 4 = 8 KB` | 双缓冲 slot 0，加载每行 tile |
| `rowBuf1` | TBuf\<float\> | `tileElems * 4 = 8 KB` | 双缓冲 slot 1，加载每行 tile |
| `weightBf16` | TBuf\<bfloat16_t\> | `tileElems * 2 = 4 KB` | 权重 bf16 输入 |
| `weightFp32` | TBuf\<float\> | `tileElems * 4 = 8 KB` | 权重 fp32 (Cast 输出 + Mul 输出) |
| `gradBuf` | TBuf\<float\> | `tileElems * 4 = 8 KB` | 梯度累加器 (原位更新) |

**总 UB 使用量**: 8 + 8 + 8 + 4 + 8 + 8 = **44 KB** << 192 KB (安全余量充足)

### 6.2 Buffer 复用策略

归约阶段与逐元素阶段时间上分离，rowBuf0/rowBuf1 在归约完成后释放，但本设计中选择显式独立分配以保证代码清晰性。UB 余量 > 75%，无需复用优化。

### 6.3 对齐要求

| 数据类型 | 32B 对齐元素数 | tileHidLen=512 是否对齐? |
|---------|--------------|------------------------|
| float32 (4B) | 8 | 512 / 8 = 64, 对齐 |
| bfloat16_t (2B) | 16 | 512 / 16 = 32, 对齐 |

所有 buffer 均满足 32 字节对齐要求，可以使用 `DataCopy` (也可统一用 `DataCopyPad`)。

---

## 7. 数据流

### 7.1 单 Core 处理流程

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1: 归约 (Reduction)                                │
│                                                         │
│  Dup(accum, 0.0f, tileElems)                             │
│                                                         │
│  for row i = 0 to 107:                                   │
│    ┌─ DMA: GM → rowBuf (DataCopyPad)                     │
│    │    src = grad_w_partial[i * 16384 + tileHidStart]   │
│    │    blockCount=4, blockLen=tileHidLen*4               │
│    │    srcStride=(4096-tileHidLen)*4                     │
│    └─ Compute: accum += rowBuf (Add<float>)              │
│                                                         │
│  // accum = grad_w_sum[tile]                              │
├─────────────────────────────────────────────────────────┤
│ Phase 2: 逐元素 (Element-wise) — 输出1                    │
│                                                         │
│  ┌─ DMA: weight_embed[tile] → weightBf16 (DataCopyPad)    │
│  ├─ Cast: weightBf16 → weightFp32 (CAST_NONE)             │
│  ├─ DMA: grad_weight_hidden[tile] → gradBuf (DataCopyPad)  │
│  ├─ Mul: weightFp32 = accum * weightFp32                  │
│  ├─ Add: gradBuf += weightFp32                            │
│  └─ DMA: gradBuf → grad_weight_hidden[tile] (DataCopyPad) │
│                                                         │
│ Phase 3: 逐元素 (Element-wise) — 输出2                    │
│                                                         │
│  ┌─ DMA: weight_hidden[tile] → weightBf16 (DataCopyPad)   │
│  ├─ Cast: weightBf16 → weightFp32 (CAST_NONE)             │
│  ├─ DMA: grad_weight_embed[tile] → gradBuf (DataCopyPad)  │
│  ├─ Mul: weightFp32 = accum * weightFp32                  │
│  ├─ Add: gradBuf += weightFp32                            │
│  └─ DMA: gradBuf → grad_weight_embed[tile] (DataCopyPad)  │
└─────────────────────────────────────────────────────────┘
```

### 7.2 GM 地址计算

```
// 计算当前 core 处理的 tile
tileIdx = blockIdx;  // 每个 core 一个 tile
tileHidStart = tileIdx * tileHidLen;

// grad_w_partial row i 在 tile 内的起始地址
rowBase = i * 4 * 4096;  // 元素偏移
rowTileAddr = rowBase + tileHidStart;

// 权重张量 tile 起始地址
weightTileAddr = tileHidStart;  // (channel stride = 4096, handled by DataCopyPad)
```

### 7.3 DataCopyPad 参数

**GM→UB: 加载一行 tile (归约阶段)**
```cpp
DataCopyExtParams copyParams;
copyParams.blockCount = 4;                                    // 4 channels
copyParams.blockLen   = tileHidLen * sizeof(float);           // tile 有效数据
copyParams.srcStride  = (4096 - tileHidLen) * sizeof(float);  // 跨 channel 间隔
copyParams.dstStride  = 0;                                    // UB 连续存放
DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
```

**GM→UB: 加载权重 bf16 (逐元素阶段)**
```cpp
copyParams.blockCount = 4;
copyParams.blockLen   = tileHidLen * sizeof(bfloat16_t);
copyParams.srcStride  = (4096 - tileHidLen) * sizeof(bfloat16_t);
```

**GM→UB / UB→GM: 加载/存储梯度 fp32 (逐元素阶段)**
与 grad_w_partial 加载参数相同。

---

## 8. API 映射表

| 操作 | API | 说明 | 验证状态 |
|------|-----|------|---------|
| GM→UB 单行 tile 搬运 | `DataCopyPad` + `DataCopyExtParams` | blockCount=4, 带跨 channel stride | 已验证 (ARA-FullLoad 模式) |
| GM→UB 权重 bf16 搬运 | `DataCopyPad` + `DataCopyExtParams` | blockCount=4, blockLen=tileHidLen*2 | 已验证 |
| UB→GM 梯度输出 | `DataCopyPad` + `DataCopyExtParams` | blockCount=4, dstStride=(4096-tileHidLen)*4 | 已验证 |
| 累加器初始化 | `Duplicate<float>` | 填充 0.0f | 已验证 (api-buffer) |
| 归约累加 | `Add<float>(dst, src0, src1, count)` | 向量加法, dst==accum, src0==accum, src1==rowBuf | 已验证 (vec_binary_intf.h) |
| bf16→fp32 转换 | `Cast<float, bfloat16_t>(dst, src, CAST_NONE, count)` | 无精度损失转换 | 已验证 (api-precision.md 规范) |
| 逐元素乘法 | `Mul<float>(dst, src0, src1, count)` | dst==weightFp32, src0==accum, src1==weightFp32 | 已验证 (vec_binary_intf.h) |
| 原位加法 | `Add<float>(dst, src0, src1, count)` | dst==gradBuf, src0==gradBuf, src1==weightFp32 | 已验证 (in-place 支持) |

### 8.1 API 约束确认

- **Add/Mul 支持 in-place**: `Add<float>(a, a, b, count)` 合法, dst 可与 src0 别名 (api-arithmetic.md)
- **DataCopyPad blockCount 上限**: 4095, 本算子最大 blockCount=4, 不受限
- **Cast RoundMode**: bf16→fp32 使用 `CAST_NONE` (无精度损失), 无需 fp32→bf16 回退 (输出为 fp32)
- **重复使用**: 本算子无 BinaryRepeatParams 广播场景, 使用 Level 2 count 版本即可

---

## 9. 精度策略

### 9.1 数值精度评估

| 计算阶段 | 数据类型 | 精度风险 |
|---------|---------|---------|
| 归约求和 | float32 累加 float32 | 低风险。108 个 float32 相加，累积误差可控 (N*eps ≈ 108 * 1.2e-7 ≈ 1.3e-5) |
| bf16→fp32 转换 | Cast (CAST_NONE) | 无风险。bf16 的 7 位尾数完整保留到 fp32 |
| 逐元素乘法 | float32 * float32 | 低风险。fp32 乘法误差 < 0.5 ULP |
| 原位加法 | float32 += float32 | 低风险 |

### 9.2 精度标准

采用浮点计算社区标准 (ops-precision-standard):

| 指标 | 阈值 | 计算方式 |
|------|------|---------|
| MERE (平均相对误差) | < 2^-13 (≈ 0.000122) | avg(\|actual - golden\| / (\|golden\| + 1e-7)) |
| MARE (最大相对误差) | < 10 * 2^-13 (≈ 0.00122) | max(\|actual - golden\| / (\|golden\| + 1e-7)) |

### 9.3 混合精度说明

- weight_hidden / weight_embed 输入为 bfloat16, 仅 7 位尾数精度
- 乘法 `grad_w_sum(fp32) * weight(fp32_from_bf16)` 的有效精度受限于 bf16 输入
- 这是输入数据类型的固有限制, 无法通过中间计算提升
- 归约和累加全程使用 float32, 不引入额外精度损失

---

## 10. 边界条件与分支场景

### 10.1 Tiling 边界处理

| 场景 | 处理 |
|------|------|
| 最后 tile 的 tileHidLen | tileHidLen 整除 4096 (512/1024/2048), 无尾块 |
| tileHidLen 非整除 | 最后 tile 使用实际剩余元素数 (`lastTileHidLen = 4096 - tileHidStart`) |
| blockCount=1 场景 | DataCopyPad 算法一致, 无特殊处理 |

### 10.2 输入边界

| 场景 | 处理 |
|------|------|
| R=1 (单行归约) | 退化: accum = 唯一行, 直接用于后续计算 |
| R=0 | 非法输入, Host 侧校验拒绝 |
| hidden_size=0 | 非法输入, Host 侧校验拒绝 |
| 非对齐 tileHidLen | 使用 alignedHidLen = Ceil32(tileHidLen * sizeof(float)) / sizeof(float) 对齐 UB 存取 |

### 10.3 多核覆盖

```
totalTiles = ceil(4096 / tileHidLen)    // 沿 hidden dim 切分
numCores  = totalTiles                   // 每 core 一个 tile
usedCores = min(numCores, MAX_CORES)     // 不超过硬件上限
```

当前默认配置: tileHidLen=512 → totalTiles=8 → 8 cores

---

## 11. Host 侧 Tiling 参数

```cpp
struct EngramGateWReduceTiling {
    uint32_t R;               // 归约行数 = 108
    uint32_t numChannels;     // channel 数 = 4
    uint32_t hiddenSize;      // hidden 维度 = 4096
    uint32_t tileHidLen;      // 每个 tile 的 hidden 长度
    uint32_t tileHidStart;    // 当前 core 的 hidden 起始偏移
    uint32_t tileElems;       // 当前 core 处理的元素数 = numChannels * tileHidLen
    uint32_t alignedHidLen;   // 32B 对齐后的 hidden 长度
    uint32_t totalTiles;      // 总 tile 数
    uint32_t tileIdx;         // 当前 core 的 tile 索引
};
```

Host 侧根据 `hiddenSize` 和 UB 容量计算 `tileHidLen`, 下发每个 core 的 `tileHidStart`。

---

## 12. 与经典 ARA 模式的关系

本设计从 ARA-FullLoad 模式推导而来, 核心差异:

| 方面 | 经典 ARA-FullLoad | 本设计 |
|------|-----------------|--------|
| 归约方式 | `ReduceSum<Pattern::Reduce::RA>` 一次 API 调用 | 手动循环 `Add` 逐行累加 |
| 输入 buffer | R * tileA0Len 全部驻留 UB | 仅需 1 (或 2) 行 + 累加器 |
| tile 大小 | 受 R 限制 (~192 elements) | 仅受 Phase 2 Buffer 限制 (~512 hidden) |
| 适用条件 | R ≤ R_max (~255) | R 无限制 |
| 后处理 | 未定义 | 内置逐元素 Mul+Add |

**选取手动迭代原因**: R=108 虽满足 ARA-FullLoad (R_max=255), 但 tileA0Len 严重受限。手动迭代以少量额外指令开销换取 2.7x 更大的 tile, 减少 tile 数量和 DMA 次数。
