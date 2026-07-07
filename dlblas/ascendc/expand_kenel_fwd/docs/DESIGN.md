# DESIGN.md — expand_kenel_fwd 算子架构设计

---

## 1. 算子分析

### 1.1 语义定义

```
PyTorch: x.unsqueeze(-2).expand(*original_shape[:-1], mhc_mult, original_shape[-1]).contiguous()

数学语义:
  输入:  x ∈ R^{..., H}
  输出:  y ∈ R^{..., M, H}   (M = mhc_mult)

  对于任意索引 (i_0, ..., i_{k-1}, m, h):
    y[i_0, ..., i_{k-1}, m, h] = x[i_0, ..., i_{k-1}, h]
```

该算子在倒数第二维插入一个新维度（大小为 mhc_mult），然后将原始数据沿新维度广播复制 mhc_mult 次，最后将结果**物化**为连续内存（对应 `.contiguous()` 调用）。

**关键特征**：
- **纯数据搬运**：不涉及任何数值计算（无加减乘除、无激活函数）
- **输出是输入的 M 份完整拷贝**：每个 (B*S, H) 行被复制 M 次
- **类型保留**：输入 dtype = 输出 dtype，无需精度转换

### 1.2 形状规格

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `B` | batch_size | 1 |
| `S` | seq_len | 1024 |
| `H` | hidden_dim | 1280 |
| `M` | mhc_mult（扩展倍数） | 2, 4, 8 |

输入形状：`(B, S, H)`  → 展平外层为 `(B*S, H)`
输出形状：`(B, S, M, H)` → 展平外层为 `(B*S, M, H)`

### 1.3 内存访问模式

```
输入:  [0..H-1], [H..2H-1], ..., [(B*S-1)*H .. B*S*H-1]         (连续)
输出:  [0..M*H-1], [M*H..2M*H-1], ..., [(B*S-1)*M*H .. B*S*M*H-1] (连续)

对于每行 (row = b*S + s):
  输入 GM 基地址:  x_base = row * H
  输出 GM 基地址:  y_base = row * M * H

  每个输入元素 x[row * H + h] 对应 M 个输出位置:
    y[row * M * H + 0 * H + h]   (副本 0)
    y[row * M * H + 1 * H + h]   (副本 1)
    ...
    y[row * M * H + (M-1) * H + h] (副本 M-1)
```

输入数据和输出数据均连续存放，便于使用 DMA 批量搬运。输出在 UB 内展开后单次 DMA 写出。

---

## 2. 技术路线决策

### 2.1 环境参数

| 参数 | 值 | 来源 |
|------|-----|------|
| 芯片型号 | Ascend910B2 | 需求指定 |
| NpuArch | `DAV_2201` | `/npu-arch` skill |
| `__NPU_ARCH__` | 2201 | `/npu-arch` skill |
| CANN 版本 | 9.0.0 | 需求指定 |
| UB 容量 | 192 KB | `/npu-arch` skill |
| L0C 容量 | 128 KB | `/npu-arch` skill |
| AI Core 数量（Vector） | 24 | `/npu-arch` skill |

### 2.2 路线选择

| 决策维度 | 判定 | 依据 |
|----------|------|------|
| 算子类型 | Broadcast/Conversion（数据搬运） | 无计算，纯数据复制 |
| 目标架构 | `DAV_2201` | Ascend910B2 |
| 候选路线 | SIMD/MemBase | 非 `DAV_3510`，不走 RegBase/Blaze |
| 最终路线 | **SIMD/MemBase + DataCopy DMA** | DAV_2201 默认路线 |

**选择理由**：
1. 目标架构为 `DAV_2201`，不满足 RegBase（需 `DAV_3510`）和 Blaze（需 `DAV_3510` 且为 MatMul 族）的前置条件。
2. 算子本质是"内存搬运 + 广播复制"，不涉及计算，DMA 路径（DataCopy/DataCopyPad）效率最高。
3. ascendc-tiling-design 的 Broadcast 类提供了 DAV_2201 的 UB Broadcast 静态接口设计方法论，本算子的 UB 内扩展可直接复用其 DataCopyPad + Copy 搬运指令广播优化模式。

### 2.3 与 ascendc-tiling-design Broadcast 模式的关系

本算子与标准 Broadcast 算子的核心差异：
- 标准 Broadcast：多个输入 shape 不同，需要维度对齐 + 计算（Add/Mul/...）
- 本算子：单输入，纯数据复制，无跨输入计算

**可复用部分**：
- UB 内广播扩展方法（§3.2 axis=-2 广播：Copy 行复制模式）
- DataCopyPad 对齐策略
- 多核切分策略（按 outer 维度均分）
- UB 切分策略（从内轴向外累乘）

**简化部分**：
- 无多输入合轴，维度固定为 2D（`(B*S, H)` → `(B*S, M, H)`）
- 无计算步骤（跳过 Add/Mul/Sub 环节）
- 无 tmpBuffer 需求（不使用 Broadcast API，使用搬运指令方案）
- Buffer 数量更少（仅输入 + 输出，无中间计算 buffer）

---

## 3. 多核切分策略

### 3.1 切分方式

将 `total_rows = B * S` 沿最外层均匀分配给所有 AI Core：

```
total_rows = B * S
coreNum = min(24, total_rows)    // 实际使用的 AI Core 数
rowsPerCore = ceil(total_rows / coreNum)
tailRows = total_rows - (coreNum - 1) * rowsPerCore
```

每个 Core 处理连续的若干行，确保负载均衡。

### 3.2 核利用率

当 `total_rows < coreNum` 时，只开 `total_rows` 个核，不做空核占位。例如 `B=1, S=1` 时只用一个核。

---

## 4. UB 切分策略

### 4.1 维度建模

展平外层后，问题转化为：

```
输入:  [N, H]  其中 N = B * S
输出:  [N, M, H]
```

将输出视为 `[N*M, H]`，即把 (N, M) 合并为一维。这样：
- 输入总元素数：`N * H`
- 输出总元素数：`N * M * H`

### 4.2 内层 Tiling（沿 H 维度）

当 H 较大时，沿 H 维度分 tile 处理：

```
tileH = min(H, maxTileH)
tilesPerRow = ceil(H / tileH)
tailH = H - (tilesPerRow - 1) * tileH
```

`maxTileH` 由 UB 容量约束确定（见 §5）。

### 4.3 总 Tile 数

```
totalTiles = N * tilesPerRow
```

---

## 5. Buffer 规划

### 5.1 数据流

```
对每个 (row, tileH_chunk):
  ┌─────────────────────────────────────────┐
  │ 1. DMA: input GM → UB inBuf             │  (tileH × sizeof(T) 字节)
  │    GM addr = x_base + row * H + tileOff  │
  ├─────────────────────────────────────────┤
  │ 2. UB Expand: inBuf → outBuf            │  (M × tileH × sizeof(T) 字节)
  │    将 inBuf 的 tileH 个元素复制 M 次     │
  ├─────────────────────────────────────────┤
  │ 3. DMA: UB outBuf → output GM            │
  │    GM addr = y_base + row * M * H       │
  │            + tileOff                     │  (M × tileH × sizeof(T) 字节)
  └─────────────────────────────────────────┘
```

### 5.2 Buffer 列表

| Buffer | 队列位置 | 大小 | 说明 |
|--------|---------|------|------|
| `inBuf[2]` | `VECIN` | `2 × tileH × sizeof(T)` | 双缓冲输入，从 GM 读入 |
| `outBuf` | `VECOUT` | `M × tileH × sizeof(T)` | 扩展后输出，写到 GM |
| `tmpBuf` | `TMP` | `M × tileH × sizeof(T)` | UB 内扩展时临时空间（与 outBuf 复用） |

### 5.3 UB 用量计算

由于本算子无计算步骤，输入读入和输出写出可以串行化。采用**双缓冲输入 + 单缓冲输出**：

```
ubBytes = 2 × tileH × sizeof(T)           // 双缓冲输入
        + M × tileH × sizeof(T)           // 扩展后输出
        = (M + 2) × tileH × sizeof(T)
```

加上临时开销（队列头、对齐 pad），预算：

```
ubBudget = 192 KB - 4 KB(reserved) = 188 KB = 192512 字节
```

`maxTileH` 约束：

```
maxTileH = floor(ubBudget / ((M + 2) × sizeof(T)))
```

FP16 (sizeof(T)=2) 典型值：

| M | maxTileH (FP16) | maxTileH (FP32) |
|---|-----------------|-----------------|
| 2 | 192512 / (4×2) = 24064 | 192512 / (4×4) = 12032 |
| 4 | 192512 / (6×2) = 16042 | 192512 / (6×4) = 8021 |
| 8 | 192512 / (10×2) = 9625 | 192512 / (10×4) = 4812 |

默认 `tileH = 1024`（FP16），远超 H=1280 的典型场景，可以整行处理无需分 tile。

### 5.4 对齐约束

- `tileH` 对齐到 **16 元素**（32 字节），满足 DataCopy 32B 对齐要求
- 尾块 `tailH` 使用 `DataCopyPad` 处理非对齐情况
- 输出写回使用 `DataCopy`（连续 `M × tileH` 元素始终 32B 对齐）

---

## 6. UB 内扩展方案

### 6.1 方案选择

复用 ascendc-tiling-design Broadcast §3.2「axis=-2 广播：Copy 行复制」模式：

```
Step 1: DataCopyPad 搬入输入 tileH 个元素到 UB inBuf
        (blockLen = tileH × sizeof(T), 自动 32B 对齐)

Step 2: Copy 将 inBuf 的一行复制 M 次到 outBuf
        srcStride = 0 (重复读同一行)
        dstStride = tileH (每行间隔 tileH 个元素)

Step 3: DataCopy outBuf → 输出 GM (连续 M × tileH 个元素)
```

### 6.2 适用条件

- `tileH × sizeof(T)` 为 32B 的倍数（满足 DataCopy 对齐约束）
- 对尾块 `tailH`，使用 DataCopyPad 搬入（自动补齐到 32B 对齐），Copy 后 GatherMask 裁剪（如 tailH 非 16 对齐）
- 无 Broadcast API 的 tmpBuffer 开销

### 6.3 替代方案（小 tileH / 非对齐场景）

当 `tileH × sizeof(T)` 不满足 32B 对齐时（如 FP32 且 tileH 为奇数），改用：
- DataCopyPad 搬入（自动补齐到 32B）
- UB 内逐元素循环复制 M 次（标量循环，M 很小，开销可忽略）
- DataCopyPad 搬出

---

## 7. GM 地址计算

### 7.1 Kernel 侧变量

```cpp
int64_t rowStart = blockIdx * rowsPerCore;     // 本 core 起始行
int64_t rowEnd   = min(rowStart + rowsPerCore, total_rows);  // 本 core 结束行
int64_t tileOff  = tileIdx * tileH;            // H 维度偏移
int64_t curTileH = min(tileH, H - tileOff);    // 当前 tile 实际元素数（尾块）
```

### 7.2 地址公式

```
输入 GM 偏移: x_base + row * H + tileOff
输出 GM 偏移: y_base + row * M * H + tileOff
输出连续长度: M × curTileH × sizeof(T)
```

其中 `x_base` 和 `y_base` 为 Host 传入的 GlobalMemory 基地址。

---

## 8. 多级流水线

### 8.1 流水线设计

采用 CopyIn → Expand → CopyOut 三级流水：

```
Tile 0:  [CopyIn] [Expand] [CopyOut]
Tile 1:           [CopyIn] [Expand] [CopyOut]
Tile 2:                    [CopyIn] [Expand] [CopyOut]
```

- CopyIn（MTE2 → UB）与 CopyOut（UB → MTE3）由不同 MTE 引擎执行，可部分重叠
- Expand 在 UB 内进行，使用 Vector 单元

### 8.2 同步点

```cpp
// CopyIn 完成
pipe->InsertSync(HardEvent::MTE2_MTE3);
// Expand 完成 (UB 内操作，无需显式同步)
// CopyOut 完成
pipe->InsertSync(HardEvent::MTE3_MTE2);
```

由于 Expand 是纯 UB 操作，不涉及跨引擎数据依赖，只需在 CopyIn 和 CopyOut 之间同步。

### 8.3 Double Buffer

输入侧使用双缓冲，通过 `EnQue` / `DeQue` 管理：

```cpp
TQue<QuePosition::VECIN, 2> inQue;   // 双缓冲输入队列
TQue<QuePosition::VECOUT, 1> outQue;  // 单缓冲输出队列

// Tile pipeline
for (tile) {
    LocalTensor<T> inBuf = inQue.DeQue<T>();     // 获取空闲输入 buffer
    DataCopy(inBuf, inGM[inOffset], tileH);       // DMA 搬入
    inQue.EnQue(inBuf);                           // 归还
    pipe->InsertSync(HardEvent::MTE2_MTE3);

    LocalTensor<T> outBuf = outQue.AllocTensor<T>();  // 获取输出 buffer
    // UB 内扩展: inBuf → outBuf
    ExpandInUB(outBuf, inBuf, M, curTileH);
    DataCopy(outGM[outOffset], outBuf, M * curTileH); // DMA 搬出
    outQue.EnQue(outBuf);
    pipe->InsertSync(HardEvent::MTE3_MTE2);
}
```

---

## 9. 数据类型与精度

### 9.1 数据类型

| 输入 dtype | 输出 dtype | 中间 dtype |
|-----------|-----------|-----------|
| FP16 | FP16 | FP16 |
| FP32 | FP32 | FP32 |
| BF16 | BF16 | BF16 |

本算子为纯数据搬运，不做类型转换。Host 侧根据输入 Tensor 的 dtype 模板化实例化对应 Kernel。

**设计约束**：按需求，只支持需求中指定的 dtype。当前需求使用 FP32（torch.randn 默认 float32），设计中以 `T` 模板参数支持 FP16/FP32/BF16，便于扩展。实际实现优先支持 FP16（Ascend 原生精度）。

### 9.2 精度标准

**算子类型**：非计算类（纯数据搬运，无数学运算）

根据 `ops-precision-standard` 判定：
- 不包含数值计算 → **非计算类标准**
- 通过标准：**Bitwise Match（二进制一致）**
- 验证方法：`numpy.array_equal(npu_output, cpu_golden)`

由于没有任何数值变换（无加法/乘法/舍入），结果必须和 PyTorch 参考实现逐位完全一致。

### 9.3 NaN/Inf 处理

输入中的 NaN/Inf 值原样复制到输出（无任何变换），不产生额外 NaN/Inf。

---

## 10. 分支场景覆盖

| 分支维度 | 条件 | 策略 |
|----------|------|------|
| **数据类型** | FP16 / FP32 / BF16 | 模板参数 `T`，编译期特化 |
| **M 值** | M=2, 4, 8（小 M） | 默认路径，UB 内一次扩展 |
| **M 值** | M > 16（大 M） | 增大 tileH 或改为多次写出，避免单个 outBuf 过大 |
| **H 边界** | tileH 完整 | DataCopy（32B 对齐） |
| **H 边界** | tailH（尾块） | DataCopyPad + GatherMask 裁剪 |
| **总行数** | N >= coreNum | 均分到所有核 |
| **总行数** | N < coreNum | 只开 N 个核 |
| **32B 对齐** | `tileH × sizeof(T)` 对齐 | DataCopy 快速路径 |
| **32B 对齐** | 非对齐 | DataCopyPad 通用路径 |

---

## 11. Host 侧 API 设计

### 11.1 函数签名

```cpp
// Host 侧算子入口
aclError ExpandKernelFwd(
    const aclTensor *x,          // 输入 Tensor, shape = (..., H)
    int64_t mhc_mult,            // 扩展倍数
    aclTensor *y,                // 输出 Tensor, shape = (..., M, H)
    aclrtStream stream
);
```

### 11.2 Tiling 参数

在 Host 侧计算并传入 Device：

| Tiling 参数 | 含义 | Device 侧类型 |
|------------|------|-------------|
| `total_rows` | 展平后的总行数 (B*S) | `int64_t` |
| `H` | hidden_dim | `int64_t` |
| `M` | mhc_mult | `int64_t` |
| `tileH` | 单次处理 H 元素数 | `int64_t` |
| `rowsPerCore` | 每个核处理的行数 | `int64_t` |
| `usedCoreCnt` | 实际使用核数 | `int64_t` |
| `totalTiles` | 总 tile 数 | `int64_t` |

### 11.3 不允许的 Host 侧预处理

根据设计约束 C9，**禁止** Host 侧对输入 Tensor 做转置、重排等预处理。Kernel 必须直接处理原始内存布局。

---

## 12. API 映射表

| API | 用途 | 约束 | 验证状态 |
|-----|------|------|---------|
| `DataCopy` | GM→UB / UB→GM 连续数据搬运 | 数据地址 32B 对齐 | 头文件确认 |
| `DataCopyPad` | 非对齐数据搬运（尾块处理） | 无对齐要求 | 头文件确认 |
| `TQue<QuePosition::VECIN, 2>` | 双缓冲输入队列管理 | DAV_2201 支持 | 头文件确认 |
| `TQue<QuePosition::VECOUT, 1>` | 单缓冲输出队列管理 | DAV_2201 支持 | 头文件确认 |
| `Duplicate<T>` | UB 内向量复制（扩展用） | DAV_2201 支持 | 头文件确认 |
| `InsertSync(HardEvent::MTE2_MTE3)` | MTE2→MTE3 同步屏障 | 标准同步原语 | 头文件确认 |
| `GetBlockIdx()` | 获取当前 AI Core 索引 | 标准内置函数 | 头文件确认 |
| `SetGlobalBuffer()` | 绑定 GlobalMemory 指针 | GM 地址有效 | 头文件确认 |

所有 API 均来自 AscendC `kernel_operator.h` 标准头文件（`/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/include/ascendc/basic_api/kernel_operator.h`）。

---

## 13. 设计检查清单

- [x] 算子类型明确：Broadcast/Conversion 数据搬运
- [x] 技术路线确定：SIMD/MemBase + DMA (DAV_2201)
- [x] 多核切分：按 B*S 行维度均分
- [x] UB 切分：沿 H 维度分 tile，tileH 受 UB 容量约束
- [x] Buffer 规划：双缓冲输入 + 单缓冲输出，总 (M+2) × tileH × sizeof(T)
- [x] 分支覆盖：dtype / M 值 / H 对齐 / 核数 四类分支
- [x] API 验证：所有 API 来自 kernel_operator.h 标准头文件
- [x] 精度标准：Bitwise Match（非计算类）
- [x] 禁止 Host 侧预处理：Kernel 直接读取原始内存布局
- [x] UB 内扩展方案：复用 Broadcast 类 Copy 行复制模式，省 tmpBuffer
