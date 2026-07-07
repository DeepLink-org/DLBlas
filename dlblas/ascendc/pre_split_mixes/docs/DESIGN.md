# DESIGN.md — pre_split_mixes 算子技术设计

---

## 1. 数学定义

### 1.1 算子语义

输入 `input_mixes` 形状为 `[batch, seq_len, M3]`，其中 `M3 = mhc_mult * 2 + mhc_mult * mhc_mult`。  
对最后一维按通道施加 per-channel scale + bias，然后将 M3 个通道拆分为三段输出：

```
Let m = mhc_mult
Let M3 = 2*m + m*m

// 1. per-channel scale + bias
scale_cat = [scale[0] 重复 m 次, scale[1] 重复 m 次, scale[2] 重复 m*m 次]  // 长度 M3
x = input_mixes * scale_cat + mhc_base                                      // elementwise, M3 通道

// 2. 三段拆分
pre_mix[b, s, c, 0]  = sigmoid( x[b, s, c] ) + mhc_pre_eps       , c ∈ [0, m)
post_mix[b, s, c, 0] = sigmoid( x[b, s, c] ) * mhc_post_mult_value, c ∈ [m, 2m)
comb_mix[b, s, c_r, c_c] = x[b, s, 2m + c_r * m + c_c]           , c_r ∈ [0, m), c_c ∈ [0, m)
```

### 1.2 输入输出规格

| 张量 | 形状 | dtype |
|------|------|-------|
| `input_mixes` | `[batch, seq_len, M3]` | FP32 |
| `mhc_scale` (权重) | `[3]` | FP32 |
| `mhc_base` (权重) | `[M3]` | FP32 |
| **pre_mix** (输出0) | `[batch, seq_len, m, 1]` | FP32 |
| **post_mix** (输出1) | `[batch, seq_len, m, 1]` | FP32 |
| **comb_mix** (输出2) | `[batch, seq_len, m, m]` | FP32 |

标量参数:

| 参数 | 类型 | 说明 |
|------|------|------|
| `mhc_mult` | int | 通道分区基数，决定 pre/post/comb 各占 m、m、m*m 通道 |
| `mhc_pre_eps` | float | pre_mix 的 eps 偏移量（默认 1e-2） |
| `mhc_post_mult_value` | float | post_mix 的乘法因子（默认 2.0） |

### 1.3 输出展开（flat layout）

为适配 AscendC GlobalTensor 的线性寻址，三个输出按行优先展开：

```
pre_mix_flat   = pre_mix.reshape(batch * seq_len * m)      // N*m 个元素
post_mix_flat  = post_mix.reshape(batch * seq_len * m)     // N*m 个元素
comb_mix_flat  = comb_mix.reshape(batch * seq_len * m * m) // N*m*m 个元素
```

其中 `N = batch * seq_len` 为总行数。

---

## 2. 方案决策

### 2.1 架构信息

| 项目 | 值 | 来源 |
|------|---|------|
| 芯片型号 | Ascend910B2 | 用户提供 |
| NpuArch | DAV_2201 | `/npu-arch` 查表 |
| SocVersion | ASCEND910B | 映射表 |
| `__NPU_ARCH__` | 2201 | 映射表 |
| UB 容量 | 192 KB (196608 B) | `npu-hardware-params.md` §2.3 |
| L1 容量 | 512 KB | 同上 |
| CANN 版本 | 9.0.0 | 用户提供 |

### 2.2 算子类型判定

- 核心计算: per-channel scale+bias (Elementwise), sigmoid (Elementwise), 标量加法/乘法 (Elementwise)
- 所有输出元素与对应输入元素之间无跨元素依赖
- 判定为 **Elementwise 类（带通道分段语义）**

### 2.3 技术路线

按路由决策流程：
```
算子类型 = Elementwise/Vector 类
目标架构 = DAV_2201 (非 DAV_3510)
→ 通用 SIMD/MemBase 路线
```

- 加载 `/ascendc-tiling-design` → Elementwise 场景路由 → `references/elewise/patterns.md` + `tiling.md`
- 复用通用 Elementwise 的 tiling 方法论（多核对齐 512 元素、UB 对齐 256B、首/尾 block 区分）
- 适配本算子的三段拆分语义

### 2.4 RegBase / Blaze 排除理由

- DAV_2201 不支持 RegBase 编程模型（RegBase 为 DAV_3510 专有能力）
- 本算子无 MatMul/Cube 计算，不涉及 Blaze 路线

---

## 3. 多核切分策略

### 3.1 总任务量

将 `batch * seq_len` 视为行数 N，每行固定 `M3` 个元素：

```
totalRows = batch * seq_len
totalElems = totalRows * M3
```

多核按**行**切分：每个核分配若干整行，保证同一行的 M3 个元素在同一核内处理（避免跨核的通道拆分逻辑复杂化）。

### 3.2 核数计算

```
minElemsPerCore = 32768 / 8 = 4096 元素 (4KB, FP32)
coreNum = min(ceil(totalElems / 4096), availableCoreNum)
```

### 3.3 每核行数分配

```
rowsPerCore = ceil(totalRows / coreNum)
rowsPerCoreAligned = align_up(rowsPerCore, 1)  // 行边界天然对齐
```

行边界由 DataCopy 的 block stride 参数控制，每核的 GM 偏移为：

```
input_offset  = blockIdx * rowsPerCore * M3 * sizeof(float)
pre_offset    = blockIdx * rowsPerCore * m * sizeof(float)
post_offset   = blockIdx * rowsPerCore * m * sizeof(float)
comb_offset   = blockIdx * rowsPerCore * m * m * sizeof(float)
```

### 3.4 尾核处理

最后一个核的行数可能小于 `rowsPerCore`：`tailRows = totalRows - (coreNum - 1) * rowsPerCore`，在 kernel 内通过判断 `blockIdx` 区分。

---

## 4. UB 切分策略

### 4.1 核心参数

```
M3 = 2*m + m*m
rowsPerChunk: 单次 UB 可容纳的行数（tiling 时计算）
elemsPerChunk = rowsPerChunk * M3          // 输入元素数
prePerChunk   = rowsPerChunk * m           // pre_mix 输出元素数
postPerChunk  = rowsPerChunk * m           // post_mix 输出元素数
combPerChunk  = rowsPerChunk * m * m       // comb_mix 输出元素数
```

### 4.2 rowsPerChunk 计算

基于 UB 总容量 (192KB) 和所需 Buffer 数量反推。详见第 5 节 Buffer 规划。

### 4.3 Kernel 执行模型

```cpp
// 每核主循环: 以 rowsPerChunk 行为一组迭代
for (rowOffset = 0; rowOffset < myRows; rowOffset += rowsPerChunk) {
    curRows = min(rowsPerChunk, myRows - rowOffset);

    // Step A: 加载输入 → inputBuf
    DataCopy(inputBuf, inputGM + rowOffset * M3, {blockCount: curRows, blockLen: M3, ...});

    // Step B: Scale + Bias → tmpBuf (逐行 broadcast)
    Mul(tmpBuf, inputBuf, scaleBuf, {blockCount: curRows, blockLen: M3, src1RepStride: 0});
    Add(tmpBuf, tmpBuf, biasBuf, {blockCount: curRows, blockLen: M3, src1RepStride: 0});

    // Step C: 三段处理与写出 (详见第 6 节数据流)
    ProcessPreSegment(tmpBuf, curRows, rowOffset);
    ProcessPostSegment(tmpBuf, curRows, rowOffset);
    ProcessCombSegment(tmpBuf, curRows, rowOffset);
}
```

---

## 5. Buffer 规划

### 5.1 常驻 Buffer（一次加载，全程复用）

| Buffer | 大小 (bytes) | 说明 |
|--------|-------------|------|
| `scaleBuf` | `M3 * 4` | 展开后的 per-channel scale |
| `biasBuf` | `M3 * 4` | per-channel bias |

总计：`2 * M3 * 4` bytes。M3 = 24 (m=4) 时仅 192 B，可忽略。

### 5.2 分块 Buffer（每 chunk 复用）

| Buffer | 大小 (bytes) | 生命周期 |
|--------|-------------|---------|
| `inputBuf` | `rowsPerChunk * M3 * 4` | Step A → B |
| `tmpBuf` | `rowsPerChunk * M3 * 4` | Step B → C（可与 inputBuf 别名复用，节省一份） |
| `preBuf` | `rowsPerChunk * m * 4` | Step C (pre 段输出) |
| `postBuf` | `rowsPerChunk * m * 4` | Step C (post 段输出) |
| `combBuf` | `rowsPerChunk * m * m * 4` | Step C (comb 段输出) |
| `sigmoidTmpBuf` | Tiling 时由 `GetSigmoidMaxMinTmpSize()` 确定 | Sigmoid 激活计算 |

### 5.3 UB 总预算

```
UB_total = 192 * 1024 = 196608 B

UB_used = 常驻 + 分块 + sigmoidTmp
        = 2*M3*4 + rowsPerChunk*M3*4         // inputBuf (可复用为 tmpBuf)
                  + rowsPerChunk*m*4          // preBuf
                  + rowsPerChunk*m*4          // postBuf (可复用 preBuf 的空间，串行写)
                  + rowsPerChunk*m*m*4        // combBuf
                  + sigmoidTmpBufSize

// 优化: preBuf 和 postBuf 串行使用，取 max(m, m) = m
UB_used_opt = 2*M3*4 + rowsPerChunk * (M3 + m + m*m) * 4 + sigmoidTmpBufSize

// rowsPerChunk 推导:
rowsPerChunk = floor((UB_total - 2*M3*4 - sigmoidTmpBufSize) / ((M3 + m + m*m) * 4))
```

由于 preBuf 和 postBuf 均只需 `rowsPerChunk * m` 个元素，可共用一块 Buffer 串行处理，进一步节省 UB。

### 5.4 rowsPerChunk 示例

| m | M3 | rowsPerChunk (不含 sigmoidTmp) | rowsPerChunk (sigmoidTmp≈2KB) |
|---|----|-------------------------------|------|
| 4 | 24 | floor(196608 / (24+4+16)/4) ≈ 1117 | ≈ 1111 |
| 8 | 80 | floor(196608 / (80+8+64)/4) ≈ 322 | ≈ 319 |
| 16 | 288 | floor(196608 / (288+16+256)/4) ≈ 87 | ≈ 86 |

实际 `sigmoidTmpBufSize` 由 `GetSigmoidMaxMinTmpSize({M3*rowsPerChunk}, 4, false, maxVal, minVal)` 在 tiling 阶段动态计算。

### 5.5 Double Buffer 考虑

由于本算子属于轻量级 Elementwise 操作（计算密度较低），UB 容量为主要约束，不强制使用 Double Buffer。若 `rowsPerChunk` 较大，可考虑对 inputBuf 做双缓冲流水以掩盖数据搬运延迟。

---

## 6. 数据流设计

### 6.1 阶段一: GM → UB 数据加载

```
inputGM  --[DataCopy]-->  inputBuf (rowsPerChunk * M3 个 FP32)
scaleGM  --[DataCopy]-->  scaleBuf (M3 个 FP32, 仅加载一次)
biasGM   --[DataCopy]-->  biasBuf  (M3 个 FP32, 仅加载一次)
```

其中 scaleGM 在 Host 侧已展开为长度 M3 的完整数组（无需在 Device 侧做 expand 操作）。

### 6.2 阶段二: Scale + Bias

```
inputBuf --[Mul, src1RepStride=0]--> tmpBuf  (inputBuf * scaleBuf)
tmpBuf   --[Add, src1RepStride=0]--> tmpBuf  (tmpBuf + biasBuf)
```

使用 `BinaryRepeatParams` 将 `scaleBuf`/`biasBuf` 沿行方向广播。`src1RepStride = 0` 表示每行重复使用相同的 src1 数据块。

### 6.3 阶段三: 三段处理与写出

#### 6.3.1 Pre 段 (通道 0 .. m-1)

对 tmpBuf 中每行的前 m 个元素:

```
// 提取 pre 段到 preBuf
DataCopy(preBuf, tmpBuf, {blockCount: curRows, blockLen: m,
        srcGap: (M3 - m) * 4, dstGap: 0});

// Sigmoid(preBuf) → preBuf
Sigmoid(preBuf, preBuf, sigmoidTmpBuf, curRows * m);

// preBuf += mhc_pre_eps
Adds(preBuf, preBuf, mhc_pre_eps, curRows * m);   // 标量加法

// 写出到 GM
DataCopy(preGM + rowOffset * m, preBuf, {blockCount: 1, blockLen: curRows * m});
```

#### 6.3.2 Post 段 (通道 m .. 2m-1)

```
// 提取 post 段到 postBuf
DataCopy(postBuf, tmpBuf + m, {blockCount: curRows, blockLen: m,
        srcGap: (M3 - m) * 4, dstGap: 0});

// Sigmoid(postBuf) → postBuf
Sigmoid(postBuf, postBuf, sigmoidTmpBuf, curRows * m);

// postBuf *= mhc_post_mult_value
Muls(postBuf, postBuf, mhc_post_mult_value, curRows * m);  // 标量乘法

// 写出到 GM
DataCopy(postGM + rowOffset * m, postBuf, {blockCount: 1, blockLen: curRows * m});
```

#### 6.3.3 Comb 段 (通道 2m .. M3-1)

```
// comb 段: 直接从 tmpBuf 提取 (无激活函数)
DataCopy(combBuf, tmpBuf + 2*m, {blockCount: curRows, blockLen: m*m,
        srcGap: 0, dstGap: 0});  // comb 段在每行末尾，srcGap = 0
// 写出到 GM
DataCopy(combGM + rowOffset * m*m, combBuf, {blockCount: 1, blockLen: curRows * m*m});
```

### 6.4 数据流总览

```
                      ┌──────────────────┐
                      │   GM: input_mixes │
                      └────────┬─────────┘
                               │ DataCopy
                      ┌────────▼─────────┐
                      │   UB: inputBuf   │
                      └────────┬─────────┘
                               │ Mul(×scale_buf)
                      ┌────────▼─────────┐
                      │   UB: tmpBuf     │
                      └────────┬─────────┘
                               │ Add(+bias_buf)
                      ┌────────▼─────────┐
               ┌──────┤   UB: tmpBuf     ├──────┐
               │      └──────────────────┘      │
          [pre段:0..m)                    [post段:m..2m)        [comb段:2m..M3)
               │                               │                    │
          Sigmoid                          Sigmoid              直接拷贝
               │                               │                    │
          Add(+eps)                       Mul(×post_mult)          │
               │                               │                    │
      ┌────────▼─────────┐           ┌────────▼─────────┐  ┌──────▼──────────┐
      │ GM: pre_mix      │           │ GM: post_mix     │  │ GM: comb_mix    │
      └──────────────────┘           └──────────────────┘  └─────────────────┘
```

---

## 7. API 映射与验证

### 7.1 DataCopy (GM ↔ UB)

| 用途 | API | 验证状态 |
|------|-----|---------|
| GM → UB 数据搬运 | `DataCopy(LocalTensor<T>& dst, GlobalTensor<T>& src, DataCopyParams&)` | 已验证 (header: `kernel_operator_data_copy_intf.h`) |
| UB 内分段提取 | `DataCopy(LocalTensor<T>& dst, LocalTensor<T>& src, DataCopyParams&)` (UB→UB) | 需确认 UB→UB 支持，否则改用分段索引 |

**DataCopyParams 结构**:
```cpp
struct DataCopyParams {
    uint16_t blockCount;  // 数据块数量
    uint16_t blockLen;    // 每块元素数
    uint16_t srcGap;      // 块间源偏移 (bytes)
    uint16_t dstGap;      // 块间目标偏移 (bytes)
};
```

### 7.2 Mul / Add (Elementwise with Broadcast)

| 用途 | API | 验证状态 |
|------|-----|---------|
| input * scale | `Mul(dst, src0, src1, count)` | 已验证 (header: `kernel_operator_vec_binary_intf.h`, L138-156) |
| + bias | `Add(dst, src0, src1, count)` | 已验证 (header: `kernel_operator_vec_binary_intf.h`, L57-78) |
| 带 broadcast 的 Mul/Add | `Mul(dst, src0, src1, mask, repeatTime, BinaryRepeatParams)` | 已验证 (同 header, L137-141) |

**广播实现方式**:
- 通过 `BinaryRepeatParams.src1RepStride = 0` 实现 scale_buf / bias_buf 沿行维度广播
- `repeatTime` = rowsPerChunk, `blockLen` = M3

### 7.3 Sigmoid 激活

| 用途 | API | 验证状态 |
|------|-----|---------|
| sigmoid 激活 | `Sigmoid(dst, src, sharedTmpBuffer, calCount)` | 已验证 (header: `adv_api/activation/sigmoid.h`, L47-50) |
| 临时缓冲区大小查询 | `GetSigmoidMaxMinTmpSize(srcShape, typeSize, isReuseSource, maxVal, minVal)` | 已验证 (header: `activation/sigmoid_tiling.h`, L32-33) |

**Sigmoid API 细节**:
- 支持 FP32 输入/输出
- 需要临时工作缓冲区 `sharedTmpBuffer`（类型为 `LocalTensor<uint8_t>`）
- 缓冲区大小通过 Host 侧 `GetSigmoidMaxMinTmpSize()` 查询，建议使用 maxVal
- 也提供无 workBuf 的重载版本（性能较低，不推荐）

### 7.4 Adds / Muls (标量运算)

| 用途 | API | 验证状态 |
|------|-----|---------|
| dst = src + scalar | `Adds(dst, src, scalar, count)` | 需确认重载签名 |
| dst = src * scalar | `Muls(dst, src, scalar, count)` | 需确认重载签名 |

若 Adds/Muls 不可用，可改用 `Duplicate` + `Add`/`Mul` 两步:

```
Duplicate(scalarBuf, epsValue, curRows * m);   // 填充标量到 UB
Add(preBuf, preBuf, scalarBuf, curRows * m);    // 逐元素加法
```

### 7.5 Duplicate (标量填充)

| 用途 | API | 验证状态 |
|------|-----|---------|
| UB 填充标量 | `Duplicate(dst, scalarValue, count)` | 已验证 (header: `kernel_operator_vec_duplicate_intf.h`, L64-65) |

---

## 8. 精度策略

### 8.1 精度标准

按 `/ops-precision-standard` 决策树:

```
包含数值计算? → 是
输入输出 dtype? → 均为浮点 (FP32)
用户声明商用标准? → 否 (未声明)
→ 浮点计算类社区标准
```

### 8.2 数值分析

| 操作 | 数据类型 | 精度影响 |
|------|---------|---------|
| `input * scale` | FP32 × FP32 → FP32 | 标准 FP32 乘法误差 (~0.5 ULP) |
| `+ bias` | FP32 + FP32 → FP32 | 标准 FP32 加法误差 (~0.5 ULP) |
| `sigmoid(x)` | FP32 → FP32 | AscendC Sigmoid 内部使用查表+插值实现，误差通常在 1e-5 量级 |
| `+ eps` | FP32 + FP32 → FP32 | 若 eps << x，可能发生大数吃小数；eps=1e-2 相对 sigmoid 输出 [0,1] 量级相当，风险低 |
| `* post_mult_value` | FP32 × FP32 → FP32 | 标准乘法误差 |

### 8.3 混合精度评估

FP32 全链路计算，无需混合精度。本算子的 sigmoid 输出值域 [0, 1]，加法/乘法操作不会引入显著的精度问题。

### 8.4 数值稳定性保护

- Sigmoid 对大值输入 (x >> 0) 趋近 1，对小值输入 (x << 0) 趋近 0 — AscendC Sigmoid API 内部已做数值稳定处理
- eps = 1e-2 避免 pre_mix 输出为 0（防止下游 log 等操作出现 -inf）

---

## 9. Host 侧 Tiling 设计

### 9.1 TilingData 结构

```cpp
struct PreSplitMixesTilingData {
    // 问题规格
    int64_t totalRows;        // batch * seq_len
    int32_t mhcMult;          // m
    int64_t mhcMult3;         // M3 = 2m + m*m
    float mhcPreEps;
    float mhcPostMultValue;

    // 多核切分
    int32_t coreNum;
    int64_t rowsPerCore;      // 每核行数（首核）
    int64_t tailRows;         // 尾核行数

    // UB 切分
    int64_t rowsPerChunk;     // 单次 UB 处理行数
    int64_t ubLoopPerCore;    // 首核 UB 循环次数
    int64_t ubTailPerCore;    // 首核 UB 尾部行数
    int64_t ubLoopTailCore;   // 尾核 UB 循环次数
    int64_t ubTailTailCore;   // 尾核 UB 尾部行数

    // Sigmoid 临时空间
    uint32_t sigmoidTmpBufSize;  // bytes
};
```

### 9.2 Tiling 计算步骤

```cpp
TilingData ComputeTiling(int64_t totalRows, int32_t m, int32_t availableCoreNum) {
    // 0. 常量
    constexpr int64_t UB_SIZE = 192 * 1024;  // DAV_2201 UB

    int64_t M3 = 2 * m + m * m;

    // 1. 多核切分（按行）
    int64_t totalElems = totalRows * M3;
    int64_t coreNum = (totalElems * 32 + 32767) / 32768;  // FP32=32bit, 4KB min
    coreNum = min(coreNum, availableCoreNum);
    int64_t rowsPerCore = (totalRows + coreNum - 1) / coreNum;
    int64_t tailRows = totalRows - (coreNum - 1) * rowsPerCore;

    // 2. Sigmoid 临时缓冲区
    uint32_t sigmoidMax, sigmoidMin;
    GetSigmoidMaxMinTmpSize({rowsPerCore * M3}, 4, false, sigmoidMax, sigmoidMin);
    uint32_t sigmoidTmpSize = sigmoidMax;  // 使用最大值以获得最佳性能

    // 3. UB 切分 — 计算 rowsPerChunk
    // UB预算 = UB_SIZE - 常驻Buffer - sigmoidTmpBuf
    int64_t fixedUB = 2 * M3 * 4 + sigmoidTmpSize;  // scaleBuf + biasBuf + sigmoidTmp
    // 分块Buffer = rowsPerChunk * (M3 + m + m*m) * 4  // inputBuf + tmpBuf(共用) + output(共用pre/post)
    //             + rowsPerChunk * m * 4               // comb 需要独立写
    // 合并: rowsPerChunk * (M3 + m + m*m + m) * 4 = rowsPerChunk * (M3 + 2m + m*m) * 4

    int64_t perRowUB = (M3 + 2 * m + m * m) * 4;
    int64_t rowsPerChunk = (UB_SIZE - fixedUB) / perRowUB;

    if (rowsPerChunk < 1) rowsPerChunk = 1;  // 兜底

    // 4. 循环次数
    int64_t ubLoopPerCore = (rowsPerCore + rowsPerChunk - 1) / rowsPerChunk;
    int64_t ubTailPerCore = rowsPerCore - (ubLoopPerCore - 1) * rowsPerChunk;
    int64_t ubLoopTailCore = (tailRows + rowsPerChunk - 1) / rowsPerChunk;
    int64_t ubTailTailCore = tailRows - (ubLoopTailCore - 1) * rowsPerChunk;

    return {...};
}
```

### 9.3 Host 侧预处理

按约束 C9（禁止 Host 侧对算子输入 tensor 做预处理），以下操作**不在 Host 侧完成**：

- scale 的 expand（由 Host 在调用前提供已经 expand 好的 `scale_expanded` 权重或由 Device 侧 vector 指令完成。考虑 `mhc_scale` 仅有 3 个元素，可使用 `Duplicate` 在 UB 中展开）
- 输出的 reshape（输出直接以 flat layout 写入 GM）

**推荐方案**：Host 侧在 tiling 阶段不预处理张量数据。`mhc_scale`(3 个 FP32) 和 `mhc_base`(M3 个 FP32) 作为权重参数直接传给 Device，Device 侧用 `Duplicate` 将 `mhc_scale` 展开为长度 M3 的 `scaleBuf`。

---

## 10. 边界场景覆盖

### 10.1 分支决策

| 条件 | 策略 |
|------|------|
| `totalRows < availableCoreNum` | 减小 coreNum，每核至少处理 4KB |
| `totalRows * M3 < 4096` (极小输入) | 单核处理，不分核 |
| `tailRows < rowsPerCore` (尾核) | 使用尾核专用循环次数 `ubLoopTailCore` |
| `mhc_mult` 较大 (M3 > UB 可行) | 每行分多次处理（当前设计中 rowsPerChunk=1 即逐行模式作为兜底） |
| `mhc_mult` = 1 边界 | M3=3, 每行仅 3 个元素 → rowsPerChunk 极大，向量效率可能较低 |

### 10.2 Shape 分支

| Shape 类别 | batch | seq_len | mhc_mult | 每核行数 | rowsPerChunk |
|-----------|-------|---------|----------|---------|-------------|
| 典型（参考实现） | 1 | 1024 | 4 | ~51 | ~1117 |
| 大 batch | 8 | 1024 | 4 | ~410 | ~1117 |
| 大 m | 1 | 1024 | 16 | 1024 (单核) | ~86 |
| 极小 | 1 | 1 | 4 | 1 (单核) | 1 |

---

## 11. 文件结构

```
operators/pre_split_mixes/
├── docs/
│   ├── DESIGN.md          ← 本文件
│   ├── PLAN.md            ← 开发计划
│   └── environment.md     ← 环境信息 (待创建)
├── CMakeLists.txt
├── op_host/
│   ├── pre_split_mixes_tiling.h     // Tiling 数据结构与计算
│   └── pre_split_mixes_tiling.cpp   // Tiling 实现
├── op_kernel/
│   └── pre_split_mixes_kernel.cpp   // Kernel 实现 (VectorCore)
├── op_test/
│   └── pre_split_mixes_test.cpp     // 单元测试
└── op_runner/
    └── pre_split_mixes_runner.py    // Python 集成测试
```
