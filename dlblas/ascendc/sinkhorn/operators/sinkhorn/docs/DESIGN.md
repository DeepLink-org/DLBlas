# Sinkhorn 算子 AscendC 实现设计方案 (DESIGN.md)

> **算子名称**: Sinkhorn Normalize
> **目标芯片**: Ascend910B2 (DAV_2201, `__NPU_ARCH__=2201`)
> **CANN 版本**: 9.0.0
> **技术路线**: SIMD/MemBase (通用 Vector API 路线)

---

## 1. 数学定义

### 1.1 算子语义

Sinkhorn 归一化将方阵迭代归一化为双随机矩阵（行和=1，列和=1）。

**输入**: `x` shape `[1, 1024, 4, 4]`, dtype `float32`，其中 `[1024, 4, 4]` 可视为 1024 个独立的 4x4 矩阵 batch。

**算法**:

```
Step 0: x = softmax(x, dim=-1) + eps
Step 1: x = x / (sum_{dim=-2}(x) + eps)           # 列归一化
Step 2..repeat: for i in [2, repeat]:
             x = x / (sum_{dim=-1}(x) + eps)       # 行归一化
             x = x / (sum_{dim=-2}(x) + eps)       # 列归一化
```

**输出**: shape `[1, 1024, 4, 4]`, dtype `float32`，"近似"双随机矩阵。

**常量**: repeat=10, eps=1e-6。

### 1.2 核心运算拆解

| 运算 | 维度 | 轴 | 每矩阵操作数 |
|------|------|-----|-------------|
| Softmax | dim=-1 (4元素) | 行归约 | ReduceMax(4) + Subs(4) + Exp(4) + ReduceSum(4) + Muls(4) |
| 加 eps | 逐元素 | 无 | Adds(16) |
| 列归约求和 | dim=-2 (4元素, stride=4) | 列归约 | 4次 ReduceSum(4) |
| 列归一化除法 | 逐元素(broadcast列和) | 无 | Mul(4) per row |
| 行归约求和 | dim=-1 (4元素, 连续) | 行归约 | ReduceSum(4) |
| 行归一化除法 | 逐元素(broadcast行和) | 无 | Muls(4) per row |

---

## 2. 技术路线决策

### 2.1 决策矩阵

| 决策维度 | 选项 | 选择 | 理由 |
|----------|------|------|------|
| 芯片架构 | DAV_2201 / DAV_3510 | **DAV_2201** | 环境信息: Ascend910B2 |
| 编程模型 | SIMD-Regbase / SIMD-MemBase | **SIMD-MemBase** | RegBase 仅 DAV_3510 支持 |
| 算子路线 | Blaze / RegBase / MemBase | **MemBase** | 非 Matmul 类，非 DAV_3510 |
| 矩阵运算 | 4x4 极小矩阵，纯 Vector | **Vector API** | Cube 单元开销大于收益 |

### 2.2 设计特征

- **算子类型**: 融合算子 (归约 + 逐元素)。核心是归约（Softmax, Sum, 归一化），归属 Reduction 类，采用 AR 模板逐行处理方案。
- **数据特点**: 1024 个 4x4 极小矩阵，每矩阵仅 16 个 float32 元素 (64 字节)。总数据量 64 KB，可完全放入 UB。
- **并行策略**: 沿 batch 维度多核并行，各核独立处理自己的矩阵子集，无需核间通信。
- **UB 策略**: 每核一次性加载全部 tile 矩阵到 UB，全在 UB 内完成所有计算后一次性写回。不启用 Double Buffer（数据量太小，流水线启动开销大于收益）。

---

## 3. 环境信息

| 参数 | 值 | 来源 |
|------|-----|------|
| 芯片型号 | Ascend910B2 | 用户输入 |
| NpuArch | DAV_2201 | `/npu-arch` skill 映射表 |
| `__NPU_ARCH__` | 2201 | `/npu-arch` skill 映射表 |
| `--npu-arch` 候选 | `2201_vec`, `2201_cube` | 选 `2201_vec`（纯 Vector 计算） |
| UB 容量 | 192 KB (196608 B) | `/npu-arch` skill |
| L1 容量 | 512 KB | `/npu-arch` skill |
| CANN 版本 | 9.0.0 | 用户输入 |
| CPU 架构目录 | `aarch64-linux` | `/usr/local/Ascend/cann-9.0.0/aarch64-linux/` |

---

## 4. 多核切分策略

### 4.1 切分维度

沿 **batch 维度**（dim=1, size=1024）均匀切分：

```
tile_batch = ceil(batch / usedCoreNum)
```

### 4.2 参数计算

```
usedCoreNum = min(blockDim, batch)       // 不超过 batch 总数
tile_batch = ceil(batch / usedCoreNum)   // 每核处理的矩阵数
tail_batch = batch - (usedCoreNum-1) * tile_batch  // 尾核矩阵数（可能少于 tile_batch）
```

### 4.3 数据分配

- Core i 处理矩阵范围: `[i * tile_batch, min((i+1) * tile_batch, batch))`
- 每核独立完成所分配矩阵的 Sinkhorn 全流程
- **无核间通信**：各矩阵完全独立

### 4.4 核利用率

| usedCoreNum | tile_batch | 每核数据量 | UB 占用 (含 tmp) |
|------------|------------|-----------|-----------------|
| 1 | 1024 | 64 KB | ~128 KB |
| 2 | 512 | 32 KB | ~96 KB |
| 8 | 128 | 8 KB | ~40 KB |
| 24 | 43 | 2.7 KB | ~35 KB |

全部场景 UB 占用均在 192 KB 限制内。

---

## 5. UB 切分与 Buffer 规划

### 5.1 策略：全载（Full Load）

由于每个矩阵仅 16 元素，且 tile_batch 数据量极小，采用**一次性全载**策略：
- 将所有 tile 矩阵从 GM 一次性 CopyIn 到 UB
- 全部计算在 UB 内完成
- 计算结果一次性 CopyOut 回 GM

**不启用 Double Buffer**：数据量太小，流水线流水线 setup 开销大于收益。

### 5.2 Buffer 清单

| Buffer 名称 | 大小 (字节) | 说明 |
|------------|------------|------|
| `xBuf` | `tile_batch * 16 * sizeof(float)` | 主数据 Buffer，原位计算（输入→软最大→归一化→输出） |
| `rowTmpBuf` | `8 * sizeof(float)` = 32B | 单行 Reduce 结果暂存 (8B 对齐) |
| `colTmpBuf` | `8 * sizeof(float)` = 32B | 单列 Reduce 结果暂存 (8B 对齐) |
| `gatherBuf` | `8 * sizeof(float)` = 32B | 列元素收集 Buffer (32B 对齐) |
| `multBuf` | `4 * sizeof(float)` = 16B | 列乘数 Buffer (4 floats) |
| `reduceBuf` | 32 KB | Reduce 系列 API 通用临时 Buffer |

### 5.3 对齐设计

- 所有 Buffer 起始地址 32 字节对齐
- `rowTmpBuf`: 使用 stride=2 存储 (每结果占 8 字节)，满足 Reduce API 的 8B 对齐要求
- 矩阵数据: 每矩阵 64 字节，天然 32B 对齐
- 行数据: 每行 4 float = 16 字节，不满足 32B 对齐要求，使用 `DataCopyPad` 搬运（不启用 Double Buffer 时单块搬运，blockLen = 有效数据长度）

**UB 内行偏移**: 因为不做行间 padding（所有矩阵紧凑连续存储），行偏移直接用真实偏移 `m*16 + r*4`，无需 `rLengthAlign`。

### 5.4 UB 容量校验

```
totalUB = tile_batch * 64 + 32 + 32 + 32 + 16 + 32768
        = tile_batch * 64 + 32880

worst_case (usedCoreNum=1, tile_batch=1024):
    totalUB = 65536 + 32880 = 98416 B ≈ 96 KB < 192 KB ✓
```

---

## 6. Kernel 计算流程

### 6.1 核心数据结构

单核 UB 内数据布局 (tile_batch 个矩阵连续存储):

```
[x_0 的 16 元素][x_1 的 16 元素]...[x_{tile_batch-1} 的 16 元素]
```

每个矩阵内按 row-major: `[r0c0, r0c1, r0c2, r0c3, r1c0, ..., r3c3]`

### 6.2 算法伪代码

```
Kernel(tile_batch, repeat, eps):

  // === Phase 0: 数据加载 ===
  DataCopyPad(xBuf, xGm[tile_offset], tile_batch * 16 floats)

  // === Phase 1: Softmax(dim=-1) + eps ===
  for m in 0..tile_batch-1:
    for r in 0..3:
      base = m * 16 + r * 4

      // 1a. ReduceMax (数值稳定)
      ReduceMax(rowTmp, xBuf[base], reduceBuf, 4)
      maxVal = rowTmp.GetValue(0)

      // 1b. 减去最大值
      Adds(xBuf[base], xBuf[base], -maxVal, 4)

      // 1c. Exp
      Exp(xBuf[base], xBuf[base], 4)

      // 1d. ReduceSum + 归一化
      ReduceSum(rowTmp, xBuf[base], reduceBuf, 4)
      sumVal = rowTmp.GetValue(0) + eps
      Muls(xBuf[base], xBuf[base], 1.0f/sumVal, 4)

    // 1e. 加 eps
    Adds(xBuf[m*16], xBuf[m*16], eps, 16)

  // === Phase 2: 列归一化 (dim=-2) ===
  for m in 0..tile_batch-1:
    // 2a. 收集列和
    for c in 0..3:
      for r in 0..3:
        gatherBuf.SetValue(r, xBuf.GetValue(m*16 + r*4 + c))
      ReduceSum(colTmp, gatherBuf, reduceBuf, 4)
      multBuf.SetValue(c, 1.0f / (colTmp.GetValue(0) + eps))

    // 2b. 逐行乘以列倒数 (Mul = 逐元素乘法)
    for r in 0..3:
      Mul(xBuf[m*16 + r*4], xBuf[m*16 + r*4], multBuf, 4)

  // === Phase 3: 迭代行归一化 + 列归一化 ===
  for iter in 1..repeat-1:
    // 3a. 行归一化 (dim=-1, 连续访问)
    for m in 0..tile_batch-1:
      for r in 0..3:
        base = m * 16 + r * 4
        ReduceSum(rowTmp, xBuf[base], reduceBuf, 4)
        sumVal = rowTmp.GetValue(0) + eps
        Muls(xBuf[base], xBuf[base], 1.0f/sumVal, 4)

    // 3b. 列归一化 (同 Phase 2)
    for m in 0..tile_batch-1:
      for c in 0..3:
        for r in 0..3:
          gatherBuf.SetValue(r, xBuf.GetValue(m*16 + r*4 + c))
        ReduceSum(colTmp, gatherBuf, reduceBuf, 4)
        multBuf.SetValue(c, 1.0f / (colTmp.GetValue(0) + eps))
      for r in 0..3:
        Mul(xBuf[m*16 + r*4], xBuf[m*16 + r*4], multBuf, 4)

  // === Phase 4: 写回 ===
  DataCopyPad(xGm[tile_offset], xBuf, tile_batch * 16 floats)
```

### 6.3 列归约详细说明

列归约 (sum dim=-2) 是唯一需要处理非连续数据的步骤。

**问题**: 矩阵内列元素 stride=4（每行 4 元素），不连续。

**解法**: 逐列收集 + ReduceSum:
1. 对每列 c，用 `LocalTensor::GetValue` 收集矩阵 m 中的 4 个元素到 `gatherBuf`
2. 在 `gatherBuf` 上执行 ReduceSum (数据已连续)
3. 计算倒数存到 `multBuf`

**为什么使用 GetValue/SetValue**:
- 黑名单仅限制 `GlobalTensor::GetValue/SetValue` (GM 端)，不限制 `LocalTensor` (UB 端)
- 每矩阵仅 4 列 * 4 元素 = 16 次访问，总量极小

### 6.4 精度策略

| 策略 | 说明 |
|------|------|
| 数据类型 | float32 全程 (输入/计算/输出统一) |
| Softmax 数值稳定 | 标准 max-subtract 方法: `exp(x - max) / sum(exp(x - max))` |
| 除零保护 | `sum + eps`，eps=1e-6 |
| 除法优化 | 预计算 `1.0f/(sum+eps)`，使用 Muls (乘法) 替代 Div (除法) |

### 6.5 流水线设计

**不使用 Double Buffer / 多级流水线**。理由:
- tile 数据仅一次 CopyIn / 一次 CopyOut，中间无额外 GM↔UB 交互
- 数据量极小，流水线 setup/teardown 开销大于收益
- 使用 `InitBuffer(xxx, 1, size)` 单 Buffer 模式

---

## 7. API 映射表

以下 API 均基于 CANN 9.0.0 头文件验证，参数签名已确认。

| 功能 | API | 头文件路径 (相对 asc/include/) | 已验证签名 |
|------|-----|------|-----------|
| GM→UB 搬运 | `DataCopyPad` | `basic_api/kernel_operator_data_copy_intf.h` | `DataCopyPad(LocalTensor&, GlobalTensor&, DataCopyExtParams&, DataCopyPadExtParams<T>&)` |
| UB→GM 搬运 | `DataCopyPad` | 同上 | 同上 (dst=GlobalTensor, src=LocalTensor) |
| 行归约最大值 | `ReduceMax` (Level 2) | `adv_api/reduce/reduce.h` | `ReduceMax(dst, src, tmp, int32_t count, bool calIndex=false)` |
| 行归约求和 | `ReduceSum` (Level 2) | 同上 | `ReduceSum(dst, src, tmp, int32_t count, ...)` |
| 指数运算 | `Exp` | `adv_api/math/exp.h` | `Exp(dst, src, uint32_t calCount)` |
| 标量加法 | `Adds` | `basic_api/kernel_operator_vec_binary_scalar_intf.h` | `Adds(dst, src, T scalar, int32_t count)` |
| 标量乘法 | `Muls` | 同上 | `Muls(dst, src, T scalar, int32_t count)` |
| 逐元素乘法 | `Mul` | `basic_api/kernel_operator_vec_binary_intf.h` | `Mul(dst, src0, src1, uint64_t count)` |
| 逐元素减法 | `Sub` | 同上 | `Sub(dst, src0, src1, uint64_t count)` |

---

## 8. Host 侧 Tiling 设计

### 8.1 TilingData 结构

```cpp
struct SinkhornTilingData {
    uint32_t batch;          // 总 batch 数 = 1024
    uint32_t mhc;            // 矩阵维度 = 4
    uint32_t repeat;         // 迭代次数 = 10
    float eps;               // epsilon = 1e-6
    uint32_t tileBatch;      // 每核处理的矩阵数
    uint32_t tailBatch;      // 尾核矩阵数
    uint32_t usedCoreNum;    // 实际使用的核数
};
```

### 8.2 Tiling 计算

```cpp
// 多核切分
uint32_t usedCoreNum = std::min(blockDim, batch);
uint32_t tileBatch = (batch + usedCoreNum - 1) / usedCoreNum;
uint32_t tailBatch = batch - (usedCoreNum - 1) * tileBatch;
if (tailBatch == 0) tailBatch = tileBatch;

// 数据搬运参数 (每核)
uint32_t matrixSize = mhc * mhc;  // = 16
uint32_t tileElements = blockIdx == usedCoreNum - 1
    ? tailBatch * matrixSize
    : tileBatch * matrixSize;
uint32_t tileBytes = tileElements * sizeof(float);
```

### 8.3 Context 传递

通过 `TilingContext` 将 `SinkhornTilingData` 传递给每个 AI Core 的 kernel 实例。

---

## 9. 数据流图

```
Host (CPU)                          Device (AI Core × N)
────────────                        ─────────────────────
                                    ┌─────────────────────┐
TilingData ────────────────────────>│ Block 0             │
  tileBatch, eps, ...               │                     │
                                    │ CopyIn              │
  xGm [1,1024,4,4]                  │  ← xGm[0:tile*16]  │
    │                               │                     │
    ├──────────────────────────────>│ Softmax dim=-1      │
    │ slice 0                        │  + eps              │
    │                               │                     │
    ├──────────────────────────────>│ Column normalize    │
    │ slice 1                        │                     │
    │                               │ Loop × (repeat-1):  │
    │  ...                           │  Row normalize      │
    │                               │  Column normalize   │
    ├──────────────────────────────>│                     │
    │ slice N-1                      │ CopyOut             │
    │                               │  → yGm[0:tile*16]  │
    │                               └─────────────────────┘
    │                               ┌─────────────────────┐
    │                               │ Block 1             │
    │                               │  (同上)              │
    │                               └─────────────────────┘
    ▼                               ┌─────────────────────┐
  yGm [1,1024,4,4] <────────────────│ Block N-1           │
                                    │  (同上)              │
                                    └─────────────────────┘
```

---

## 10. 精度验证标准

按照浮点计算类社区标准 (`ops-precision-standard`):

| 指标 | 阈值 |
|------|------|
| MERE (平均相对误差) | < 2^-13 ≈ 0.000122 |
| MARE (最大相对误差) | < 10 × 2^-13 ≈ 0.00122 |

**数值稳定性保护**:
- Softmax: max-subtract 方法，避免 exp 溢出
- 除零保护: 所有求和后 + eps
- 小值域: 如 golden 中某元素 < 2^-14，启用小值域标准

---

## 11. 边界条件

| 场景 | 处理 |
|------|------|
| batch 不可整除 coreNum | 尾核处理 `tail_batch` (可能 < `tile_batch`) 个矩阵 |
| 全零输入 | Softmax → 均匀分布 (0.25 + eps)，迭代收敛 |
| 极大值输入 | max-subtract 保证 exp 不溢出 |
| 极小值输入 (全 -inf) | Softmax → NaN，需在文档中标注 |
| eps=0 | 除零风险，使用 eps=1e-6 固定值 |
| tile_batch * 16 * 4 非 32B 对齐 | DataCopyPad 自动处理 |
| Reduce dst 8B 对齐 | rowTmpBuf/colTmpBuf 使用 stride=2 float 存储，满足对齐 |
| blockCount > 4095 | tile_batch ≤ 1024，blockCount = tile_batch * 16 ≤ 16384。如果 blockCount 超限，分批 CopyIn |

### blockCount 限制处理

`DataCopyPad` 的 `blockCount` 最大值为 4095。当 `tile_batch * 16 > 4095` 时，需要分批搬运。约束条件：`tile_batch ≤ 255`。在 Tiling 计算时确保 `tile_batch = min(ceil(batch/usedCoreNum), 255)`。对于 batch=1024，这要求 `usedCoreNum ≥ 4`。在 Host 侧进行 clip 处理:

```cpp
constexpr uint32_t MAX_TILE_MATRICES = 255;
uint32_t tileBatch = std::min(rawTileBatch, MAX_TILE_MATRICES);
```

---

## 12. 文件结构

```
operators/sinkhorn/
├── CMakeLists.txt                      # 顶层 CMake
├── op_host/
│   ├── sinkhorn_tiling.h               # Tiling 数据结构
│   ├── sinkhorn.cpp                    # Host 侧算子入口 (Tiling + Kernel 启动)
│   └── CMakeLists.txt
├── op_kernel/
│   ├── sinkhorn_kernel.h               # Device 侧 Kernel 实现
│   └── CMakeLists.txt
├── scripts/
│   └── gen_test_data.py                # 测试数据生成脚本
└── docs/
    ├── DESIGN.md                       # 本文件
    └── PLAN.md                         # 开发计划
```
