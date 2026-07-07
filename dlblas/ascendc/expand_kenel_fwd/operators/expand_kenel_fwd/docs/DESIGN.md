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
- **输出是输入的 M 份完整拷贝**：每个 `(B*S, H)` 行被复制 M 次
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

输入数据和输出数据均连续存放。输出 GM 中相邻副本间隔为 H 个元素。

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
| 算子类型 | Conversion/Broadcast（数据搬运） | 无计算，纯数据复制 |
| 目标架构 | `DAV_2201` | Ascend910B2 |
| 候选路线 | SIMD/MemBase | 非 `DAV_3510`，不走 RegBase/Blaze |
| 最终路线 | **SIMD/MemBase + DataCopy DMA** | DAV_2201 默认路线 |

**选择理由**：
1. 目标架构为 `DAV_2201`，不满足 RegBase（需 `DAV_3510`）和 Blaze（需 `DAV_3510` 且为 MatMul 族）的前置条件。
2. 算子本质是"内存搬运 + 广播复制"，不涉及计算，DMA 路径（DataCopy/DataCopyPad）效率最高。
3. ascendc-tiling-design 的 Broadcast 类提供了 DAV_2201 的 UB Broadcast 静态接口设计方法论。本算子的 UB 内扩展可复用其多核切分和 UB 切分的通用思路。

### 2.3 与 ascendc-tiling-design 的关系

本算子与标准 Broadcast 算子的核心差异：
- 标准 Broadcast：多个输入 shape 不同，需要维度对齐 + 计算（Add/Mul/...）
- 本算子：单输入，纯数据复制，无跨输入计算

**可复用部分**：
- 多核切分策略（按 outer 维度均分）
- UB 切分策略（从内轴向外累乘，受 UB 容量约束）
- 32B 对齐约束

**独立设计部分**：
- UB 内逐副本 DataCopy（无 Broadcast API 的多输入合轴）
- 输出按副本逐份 DataCopyPad 写出（GM stride H 与 UB stride tileH 不同）

---

## 3. 多核切分策略

### 3.1 切分方式

将 `total_rows = B * S` 沿最外层均匀分配给所有 AI Core：

```
total_rows = B * S
coreNum = min(GetVectorCoreNum(), total_rows)
rowsPerCore = ceil(total_rows / usedCoreCnt)
```

每个 Core 处理连续的若干行，尾核处理全部剩余行，确保负载均衡。

### 3.2 核利用率

当 `total_rows < coreNum` 时，只开 `total_rows` 个核，不做空核占位。Kernel 入口处通过 `blockIdx >= usedCoreCnt` 判断空闲核并提前返回。

---

## 4. UB 切分策略

### 4.1 维度建模

展平外层后，问题转化为：

```
输入:  [N, H]  其中 N = B * S
输出:  [N, M, H]
```

将输出视为 `[N*M, H]`（按副本展平），但以整行为处理粒度。

### 4.2 tileH 确定

由于 `H=1280` 等典型值远小于 UB 容量约束下的 maxTileH，多数场景下可整行处理（`tilesPerRow=1`）：

```
tileH = AlignUp16(H)   // 对齐到 16 元素（32B），满足 DataCopy 对齐要求
tailH = H               // 当 tilesPerRow=1 时，tailH = H（实际数据量）
```

当 H 极大超过 UB 容量时，沿 H 维度分 tile：

```
maxTileH = floor(UB_BUDGET / ((M + 2) * sizeof(T)))
maxTileH = AlignDown16(maxTileH)
tileH = min(AlignUp16(H), maxTileH)
tilesPerRow = ceil(H / tileH)
tailH = H - (tilesPerRow - 1) * tileH
```

### 4.3 H 维度对齐约束

**设计约束**：`H % 16 == 0`。

**原因**：输出 GM 中相邻副本间隔 `M * H * sizeof(T)` 字节。当 H 不是 16 的倍数时：
- UB 内相邻副本地址间隔 `tileH * sizeof(T)` 始终 32B 对齐（tileH 已对齐到 16）
- 但 GM 中相邻副本间隔 `H * sizeof(T)` 可能不是 32B 对齐
- 导致后续副本的 GM 写入目标地址非 32B 对齐，DataCopyPad(UB→GM) 产生数据错误

**缓解方案**：Host 侧输入校验 `H % 16 != 0` 时拒绝执行并返回明确错误信息。所有 LLM 常见 hidden size（768, 1024, 1280, 2048, 2560, 4096, 5120, 8192, 13824, 16384）均为 16 的倍数。

---

## 5. Buffer 规划

### 5.1 数据流

```
对每个 (row, tileH_chunk):
  ┌─────────────────────────────────────────┐
  │ 1. CopyIn:  GM → UB inBuf               │  (curTileH × sizeof(T) 字节)
  │    DataCopyPad(xGm[row*H+tileOff],       │
  │               inBuf, curTileH)           │
  ├─────────────────────────────────────────┤
  │ 2. Expand:  UB inBuf → UB outBuf        │  (M × tileH × sizeof(T) 字节)
  │    for m in 0..M-1:                     │
  │      DataCopy(outBuf[m*tileH], inBuf,    │
  │               curTileH)                  │
  ├─────────────────────────────────────────┤
  │ 3. CopyOut: UB outBuf → GM yGm          │
  │    for m in 0..M-1:                     │
  │      DataCopyPad(yGm[row*M*H+m*H        │
  │                    +tileOff],            │
  │                  outBuf[m*tileH],        │
  │                  curTileH)               │
  └─────────────────────────────────────────┘
```

### 5.2 UB Buffer 布局

```
outBuf (M * tileH 元素):
  ┌──────────────┬───────┬──────────────┬───────┬─────┬──────────────┬───────┐
  │ replica[0]   │ pad   │ replica[1]   │ pad   │ ... │ replica[M-1] │ pad   │
  │ curTileH elm │ ...   │ curTileH elm │ ...   │     │ curTileH elm │ ...   │
  └──────────────┴───────┴──────────────┴───────┴─────┴──────────────┴───────┘
  |<── tileH ──>|        |<── tileH ──>|               |<── tileH ──>|

UB 内副本按 tileH 步长对齐排列（tileH >= H, 16 对齐）。
Padding 区域数据无意义，CopyOut 只读取有效 curTileH 元素。
```

### 5.3 Buffer 列表

| Buffer | 队列位置 | 队列深度 | 大小 | 说明 |
|--------|---------|---------|------|------|
| `inBuf` | `VECIN` | 2（双缓冲） | `tileH × sizeof(T)` | 从 GM 读入，Ping-Pong 交替 |
| `outBuf` | `VECOUT` | 1（单缓冲） | `M × tileH × sizeof(T)` | 扩展后输出，写到 GM |

### 5.4 UB 用量计算

```
ubBytes = 2 × tileH × sizeof(T)            // 双缓冲输入
        + M × tileH × sizeof(T)            // 扩展后输出
        = (M + 2) × tileH × sizeof(T)

ubBudget = 192 KB - 4 KB(reserved) = 188 KB = 192512 字节
maxTileH = floor(ubBudget / ((M + 2) × sizeof(T)))
```

FP16 (sizeof(T)=2) 典型值：

| M | maxTileH (FP16) | 
|---|-----------------|
| 2 | 192512 / (4×2) = 24064 |
| 4 | 192512 / (6×2) = 16042 |
| 8 | 192512 / (10×2) = 9625 |

默认 `tileH = AlignUp16(H)`。对于 H=1280 的典型场景，tileH=1280，远小于 maxTileH，可整行处理。

---

## 6. UB 内扩展方案

### 6.1 核心思路

UB 内扩展采用**逐副本 DataCopy** 方案：

```cpp
for (int64_t m = 0; m < M; m++) {
    DataCopy(outBuf[m * tileH], inBuf, static_cast<uint32_t>(curTileH));
}
```

每条 `DataCopy(LocalTensor, LocalTensor, count)` 将 inBuf 的 curTileH 个元素复制到 outBuf 的第 m 个副本位置。

### 6.2 签名验证

```cpp
// 来源: kernel_operator_data_copy_intf_impl.h, line 756
// 约束: count % GetC0Count(sizeof(T)) == 0  (即 count * sizeof(T) 为 32B 倍数)
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T> &dst, const LocalTensor<T> &src, const uint32_t count);
```

- **适用条件**：`curTileH × sizeof(T)` 为 32B 倍数。由于 tileH=AlignUp16(H)，对于`tilesPerRow=1`的情况，`curTileH=H`且`H%16==0`（由 Host 侧校验保证），始终满足对齐约束。
- 对于分 tile 场景（`tilesPerRow>1`），中间完整 tile 的 `curTileH=tileH=AlignDown16(maxTileH)`，满足对齐。尾块 `tailH` 的 DataCopy 在 UB 内进行，对齐到 `tileH` 的步长下 UB 地址始终 32B 对齐，DataCopy 的 count 为 `curTileH`，当`curTileH%16!=0`时可能在 NPU mode 下静默截断为非预期行为。此时需将 curTileH 也对齐到 16 后再做 DataCopy（多余 padding 元素由 CopyOut 阶段裁剪，因为 CopyOut 只写 curTileH 实际元素）。

### 6.3 关键约束汇总

| 约束 | 说明 |
|------|------|
| UB-to-UB DataCopy | count × sizeof(T) 必须为 32B 倍数 |
| UB-to-GM DataCopyPad | GM 目的地址建议 32B 对齐（非对齐可能导致数据错误） |
| GM-to-UB DataCopyPad | 自动处理非对齐 GM 源地址（内部补齐到 32B） |

---

## 7. 流水线与同步

### 7.1 流水线深度

采用 CopyIn → Expand → CopyOut 三级流水，单行单 tile 内串行执行：

```
Tile 0:  [CopyIn(MTE2)] [Expand(VEC)] [CopyOut(MTE3)]
Tile 1:                  [CopyIn(MTE2)] [Expand(VEC)] [CopyOut(MTE3)]
```

### 7.2 同步机制

使用 TQue 队列的 EnQue/DeQue 隐式同步 + 显式 InsertSync：

```cpp
// CopyIn 阶段 (MTE2 → UB)
inBuf = inQue.AllocTensor<T>();                  // 从缓冲池获取
DataCopyPad(inBuf, xGm[offset], params, pad);    // MTE2 异步搬运
inQue.EnQue(inBuf);                              // 入队，隐含 MTE2→VEC 依赖

// Expand 阶段 (UB 内 Vector)
inBuf = inQue.DeQue<T>();                        // 出队，等待 MTE2 完成
outBuf = outQue.AllocTensor<T>();
for (m in 0..M-1) {
    DataCopy(outBuf[m*tileH], inBuf, curTileH);  // VEC 搬运
}
outQue.EnQue<T>(outBuf);                         // 入队，隐含 VEC→MTE3 依赖
inQue.FreeTensor(inBuf);                         // 归还缓冲池

// CopyOut 阶段 (UB → GM via MTE3)
outBuf = outQue.DeQue<T>();                      // 出队，等待 VEC 完成
for (m in 0..M-1) {
    DataCopyPad(yGm[gmBase], outBuf[ubBase], cpParams); // MTE3 异步搬运
}
outQue.FreeTensor(outBuf);                       // 归还缓冲池
```

**依赖链分析**：

| 依赖 | 生产者 | 消费者 | 同步机制 |
|------|--------|--------|----------|
| inBuf: MTE2 DMA → VEC 读取 | MTE2 (DataCopyPad) | VEC (DataCopy) | VECIN EnQue→DeQue |
| outBuf: VEC 写入 → MTE3 DMA | VEC (DataCopy) | MTE3 (DataCopyPad) | VECOUT EnQue→DeQue |

双缓冲输入（VECIN depth=2）允许当前 tile CopyIn 和上一 tile Expand 在不同 buffer 上重叠，当 `tilesPerRow > 1` 时生效。`tilesPerRow = 1` 时单缓冲与双缓冲功能等价。

---

## 8. GM 地址计算与 CopyOut 策略

### 8.1 地址公式

```
输入 GM:  xGm[row * H + tileOff]
输出 GM:  yGm[row * M * H + m * H + tileOff]    (第 m 份副本)
```

### 8.2 CopyOut 逐副本写出策略

**为什么不能一次写出整个 outBuf？**

UB 中 outBuf 按 `tileH` 步长排列（含 padding），GM 中副本按 `H` 步长排列。两个步长不同，因此必须逐副本写出：

```
UB:  outBuf[0*tileH], outBuf[1*tileH], ..., outBuf[(M-1)*tileH]
GM:  yGm[0*H],        yGm[1*H],        ..., yGm[(M-1)*H]
```

**GM 地址对齐**：每份副本的 GM 目标地址为 `yGm[row*M*H + m*H + tileOff]`。当 `H % 16 == 0` 且 `tileOff % 16 == 0` 时，目标地址 32B 对齐。

---

## 9. 数据类型与精度

### 9.1 数据类型

本算子为纯数据搬运，不做类型转换。Kernel 模板化实例化：

| 输入 dtype | 输出 dtype | sizeof(T) | Kernel 实例 |
|-----------|-----------|-----------|-------------|
| FP16 | FP16 | 2 | `KernelExpand<half>` |
| FP32 | FP32 | 4 | `KernelExpand<float>` |
| BF16 | BF16 | 2 | `KernelExpand<half>` (同 FP16) |

### 9.2 精度标准

**算子类型**：非计算类（纯数据搬运，无数学运算）。

根据 `ops-precision-standard` 判定：不包含数值计算 → **非计算类标准**。
- 通过标准：**Bitwise Match（二进制一致）**
- 验证方法：`numpy.array_equal(npu_output, cpu_golden)`

由于没有任何数值变换（无加法/乘法/舍入），结果必须和 PyTorch 参考实现逐位完全一致。

### 9.3 NaN/Inf 处理

输入中的 NaN/Inf 值原样复制到输出（无任何变换），不产生额外 NaN/Inf。

---

## 10. 分支场景覆盖

| 分支维度 | 条件 | 策略 |
|----------|------|------|
| **数据类型** | FP16 / FP32 | Kernel 模板参数 `T`，Host 侧根据 dtypeSize 选择实例 |
| **H 对齐** | H % 16 == 0 | 主路径（DataCopy + DataCopyPad 全对齐） |
| **H 对齐** | H % 16 != 0 | Host 侧拒绝，返回明确错误信息 |
| **M 值** | M = 2, 4, 8（小 M） | 默认路径，UB 内一次扩展 |
| **M 值** | M > 16（大 M） | Tiling 自动减小 tileH 适配 UB 容量 |
| **总行数** | N >= coreNum | 均分到所有核 |
| **总行数** | N < coreNum | 只开 N 个核，空闲核提前返回 |
| **H 分 tile** | H <= maxTileH | tilesPerRow = 1，整行处理 |
| **H 分 tile** | H > maxTileH | tilesPerRow = CeilDiv(H, tileH)，沿 H 分 tile |

---

## 11. Host 侧设计

### 11.1 函数签名

```cpp
// Host 侧算子入口（直调模式）
int32_t main(int32_t argc, char* argv[]);  // 命令行: <B> <S> <H> <M> [dtype]

// PyTorch 扩展入口
at::Tensor expand_kenel_fwd(const at::Tensor& x, int64_t mhc_mult);
```

### 11.2 Tiling 参数结构体

```cpp
struct ExpandTilingData {
    int64_t totalRows;      // 展平后总行数 = B * S
    int64_t H;              // hidden_dim (最后一维大小)
    int64_t M;              // mhc_mult 扩展倍数
    int64_t tileH;          // UB buffer 中 tileH (AlignUp16(H))
    int64_t rowsPerCore;    // 每个 AI Core 处理的行数
    int64_t usedCoreCnt;    // 实际使用的 AI Core 数量
    int64_t totalTiles;     // 总 tile 数 (= totalRows * tilesPerRow)
    int64_t tailH;          // 尾块 H 元素数 (= H, tilesPerRow=1)
    uint32_t dtypeSize;     // sizeof(T) in bytes (2=FP16/BF16, 4=FP32)
};
```

### 11.3 参数校验

```
1. 输入 Tensor 维度 >= 2
2. mhc_mult > 0
3. 输入 dtype 与输出 dtype 一致
4. H % 16 == 0  (32B 对齐要求)   ← 关键校验
5. 输出形状 = (*input_shape[:-1], mhc_mult, input_shape[-1])
```

### 11.4 不允许的 Host 侧预处理

根据设计约束 C9，**禁止** Host 侧对输入 Tensor 做转置、重排等预处理。Kernel 必须直接处理原始内存布局。

---

## 12. API 映射表

| API | 用途 | 签名（已验证） | 约束 |
|-----|------|---------------|------|
| `DataCopy(Local, Local, count)` | UB 内逐副本复制 | `kernel_operator_data_copy_intf_impl.h:756` | count×sizeof(T) 为 32B 倍数 |
| `DataCopyPad(Local, Global, DataCopyParams, DataCopyPadParams)` | GM→UB 搬入（含 padding） | `kernel_operator_data_copy_intf_impl.h:1256` | DAV_2201 AIV 专用，dst 必须 UB/L1 |
| `DataCopyPad(Global, Local, DataCopyParams)` | UB→GM 搬出 | `kernel_operator_data_copy_intf_impl.h:1300` | DAV_2201 AIV 专用，GM 目标地址建议 32B 对齐 |
| `TQue<TPosition::VECIN, 2>` | 双缓冲输入队列 | 模板声明，编译期检查 | depth=2 |
| `TQue<TPosition::VECOUT, 1>` | 单缓冲输出队列 | 模板声明，编译期检查 | depth=1 |
| `TPipe::InitBuffer` | 分配队列缓冲区 | 标准接口 | 须在 `Init()` 中调用 |
| `GetBlockIdx()` | 获取当前 AI Core 索引 | 标准内置函数 | — |
| `SetGlobalBuffer()` | 绑定 GlobalMemory 指针 | `GlobalTensor` 成员方法 | GM 地址有效 |
| `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` | 获取可用 Vector Core 数 | aclrt 运行时 API | Host 侧 |
| `aclrtMalloc` / `aclrtMemcpy` | GM 内存分配 / Host↔Device 拷贝 | aclrt 运行时 API | Host 侧 |

**禁止使用的 API**：

| API | 原因 |
|-----|------|
| `GlobalTensor::SetValue()` / `GetValue()` | 效率极低，逐元素标量操作 |
| `Duplicate()` | 仅支持标量广播，不适用于向量数据复制 |

---

## 13. 设计检查清单

- [x] 算子类型明确：Conversion/Broadcast 数据搬运
- [x] 技术路线确定：SIMD/MemBase + DMA (DAV_2201)
- [x] 多核切分：按 B*S 行维度均分
- [x] UB 切分：tileH = AlignUp16(H)，tilesPerRow=1 为主路径
- [x] Buffer 规划：双缓冲输入 (VECIN, 2) + 单缓冲输出 (VECOUT, 1)
- [x] UB 用量：`(M+2) × tileH × sizeof(T) ≤ UB_BUDGET`
- [x] H 对齐约束：H%16==0，Host 侧校验拒绝非法输入
- [x] CopyOut 策略：逐副本 DataCopyPad(UB→GM)，UB stride tileH vs GM stride H
- [x] 流水线同步：TQue EnQue/DeQue 隐式同步，依赖链完整
- [x] 分支覆盖：dtype / M 值 / H 对齐 / 核数 / H 分 tile 五类分支
- [x] API 验证：所有 API 签名已通过 CANN 9.0.0 头文件验证
- [x] 精度标准：Bitwise Match（非计算类）
- [x] 禁止 Host 侧预处理：Kernel 直接读取原始内存布局
- [x] AIC 核限制：DataCopyPad 在 DAV_2201 上仅 AIV 核可用，算子运行在 Vector 核上
