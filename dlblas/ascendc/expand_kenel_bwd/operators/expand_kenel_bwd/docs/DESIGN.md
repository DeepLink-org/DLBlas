# Expand Kernel Backward 算子架构设计文档 (DESIGN.md)

## 1. 需求分析与算子语义

### 1.1 算子来源

本算子实现 **expand_to_mhc** 操作的反向传播。前向 expand 沿 dim=-2 维度将输入广播扩展 `mhc_mult` 倍，其反向即为沿该广播维度做 reduce sum。

### 1.2 数学定义

```
前向: input(n0, n1, h) → broadcast(dim=-2, factor=mhc_mult) → output(n0, n1, mhc_mult, h)

反向 (本算子):
  o_grad(n0, n1, mhc_mult, h) → sum(o_grad, dim=-2) → output(n0, n1, h)
```

逐元素公式:

```
output[n0_idx][n1_idx][h_idx] = Σ_{m=0}^{mhc_mult-1} o_grad[n0_idx][n1_idx][m][h_idx]
```

### 1.3 典型 Shape

| 维度 | 符号 | 典型值 | 说明 |
|------|------|--------|------|
| dim 0 | n0 | 2 | batch 维度之一 |
| dim 1 | n1 | 1024 | 序列长度或 batch 维度 |
| dim 2 | mhc_mult | 4 | 广播倍数 (归约轴) |
| dim 3 | h (hidden_dim) | 1280 | 隐藏层维度 |

- **输入 shape**: `(2, 1024, 4, 1280)`，FP16，~20 MB
- **输出 shape**: `(2, 1024, 1280)`，FP16，~5 MB

### 1.4 目标平台

| 属性 | 值 |
|------|-----|
| 芯片型号 | Ascend 910B2 |
| NPU 架构 | DAV_2201 |
| CANN 版本 | 9.0.0 |
| Vector 核数 | 48 (AIV) |
| UB 容量 | 192 KB (196608 bytes) |
| `VECTOR_REG_WIDTH` | 256 bits (32 bytes) |

---

## 2. 技术路线选择

### 2.1 路线决策总表

| 决策维度 | 选择 | 依据 |
|----------|------|------|
| 算子类型 | **Reduction (归约类)** | 沿 dim=-2 做 ReduceSum |
| 架构路线 | **SIMD / MemBase** | DAV_2201 不支持 RegBase / Blaze |
| 数据流模式 | **ARA-FullLoad** | 归约轴 R=4 极小，可全载入 UB |
| 归约实现 | **Elementwise Add (3 次)** | R=4 极小时比 Reduce API 更高效 |
| 中间精度 | **FP32 显式提升** | 避免 FP16 中间截断误差累积 |
| Tiling 策略 | **自适应 Tiling** | 运行时根据 shape 和核数动态计算 |

### 2.2 排除的技术路线

| 路线 | 不适用原因 |
|------|-----------|
| RegBase (SIMD-Regbase) | 仅 DAV_3510 支持；DAV_2201 无此能力 |
| Blaze / tensor_api | 仅 DAV_3510 支持；本算子为纯 vector 归约，非 Matmul |
| Cube (MatMulImpl) | 归约操作无需 Cube 计算单元 |
| `ReduceSum` (Level 2 API) | 要求归约轴元素连续存储；ARA 模式下 R=4 个元素由 `tileA0Len` 间隔，不连续 |
| `WholeReduceSum` (vcadd) | `srcBlkStride` 语义要求块内元素连续；ARA 跨行元素不连续 |
| `BlockReduceSum` | 同上，块内元素需连续 |
| 固定 Tiling (硬编码 blockDim/blockIdx) | 违背可移植性要求；不同卡配置核数不同 |

### 2.3 Elementwise Add vs Reduce API 的详细分析

ARA 合轴后 UB 布局为:

```
UB 连续地址:
  [row0[0..tileA0Len-1] | row1[0..tileA0Len-1] | row2[0..tileA0Len-1] | row3[0..tileA0Len-1]]

归约轴 R=4 的 4 个元素 (例如列 j):
  位置: j, j+tileA0Len, j+2*tileA0Len, j+3*tileA0Len
  间隔: tileA0Len (不连续)
```

Reduce API 要求归约轴元素在内存中连续，而 ARA 布局下它们被 `tileA0Len` 个元素隔开。因此改用 Elementwise Add:
1. 通过 `LocalTensor::operator[]` 创建 4 个行视图 (row0~row3)
2. 逐元素 Add: `result = row0 + row1 + row2 + row3`
3. 仅需 3 次 Add，且支持硬件向量化

---

## 3. 合轴分析 (Axis Merging)

### 3.1 合轴过程

```
原始 shape: [2, 1024, 4, 1280]
axes = [2]  (dim=-2, 即 mhc_mult 维度)

标记各维属性:
  dim0 (n0=2):       A (保留)
  dim1 (n1=1024):    A (保留)
  dim2 (mhc_mult=4): R (归约)
  dim3 (h=1280):     A (保留)

消除冗余维: 无 size=1 的维度, 跳过

合并相邻同类型轴:
  dim0(A) + dim1(A) → A1 = 2 × 1024 = 2048
  dim2(R)           → R  = 4
  dim3(A)           → A0 = 1280

合轴后 shape: (A1=2048, R=4, A0=1280) → ARA 模式
```

### 3.2 ARA 布局示意图

```
GM 视角 (输入 o_grad):
┌──────────────────────────────────────────────────────┐
│ A1=2048 行, 每行 R×A0=4×1280 个 half                 │
│                                                      │
│ A1[0]:   [row0(1280) | row1(1280) | row2(1280) | row3(1280)]  │
│ A1[1]:   [row0(1280) | row1(1280) | row2(1280) | row3(1280)]  │
│ ...                                                  │
│ A1[2047]: [...]                                      │
└──────────────────────────────────────────────────────┘

GM 视角 (输出):
┌──────────────────────────────────────────────────────┐
│ A1=2048 行, 每行 A0=1280 个 half                     │
└──────────────────────────────────────────────────────┘
```

---

## 4. 向量化策略

### 4.1 数据搬移向量化

使用 `DataCopyPad` API 进行 GM-UB 数据搬移，通过 `DataCopyExtParams` 配置块式搬运:

**CopyIn (GM -> UB):**
```
blockCount = R = 4          # 搬 4 行
blockLen   = tileA0Len × sizeof(half)  # 每行长度 (bytes)
srcStride  = (A0 - tileA0Len) × sizeof(half)  # GM 中行间间隔
dstStride  = 0              # UB 中连续存放
```

当 `tileA0Len = A0 = 1280` (全载) 时，`srcStride = 0`，等价于一次连续搬运 `R × A0` 个 half。

**CopyOut (UB -> GM):**
```
blockCount = 1              # 单块
blockLen   = tileA0Len × sizeof(half)
srcStride  = 0
dstStride  = 0
```

### 4.2 计算向量化

归约计算通过逐元素 `Add` 实现，每条指令处理 `VECTOR_REG_WIDTH / sizeof(half) = 128` 个 half 元素 (即 `a0TileBase`):

- **一次 Add 指令覆盖 128 个元素**，`tileA0Len=1280` 时需 `1280/128 = 10` 条向量指令
- **3 次 Add 共 30 条向量指令**
- 配合 Double Buffer，向量计算与数据搬移流水重叠

### 4.3 FP32 中间累加

```
精度流: half (输入) → float (Cast, CAST_NONE) → float (Add ×3) → half (Cast, CAST_ROUND)

UB Buffer 布局:
  castBuf: [accF32 (tileA0Len floats) | tmpF32 (tileA0Len floats)]
             ↑ 累加缓冲区                ↑ 临时类型转换缓冲区
```

**为什么必须显式 FP32 累加?** 直接 3 次 `Add<half>` 会在每次加法后截断为 FP16，累加 4 个值时误差累积可达 `(R-1) × 0.5 ULP ≈ 1.5 ULP`。显式升为 FP32 累加后单次截断，误差仅 `0.5 ULP` (即观测到的 max_diff = 7.81e-03 = 1/128)。

---

## 5. 内存规划

### 5.1 UB Buffer 分配

| Buffer | 类型 | 深度 | 用途 | 单份大小 (bytes) | 总大小 (bytes) |
|--------|------|------|------|-----------------|---------------|
| `inQueueX` | `TQue<VECIN, 1>` | Double (2) | 输入: R × tileA0Len 个 half | 4 × 1280 × 2 = 10240 | 20480 |
| `outQueueY` | `TQue<VECOUT, 1>` | Double (2) | 输出: tileA0Len 个 half | 1280 × 2 = 2560 | 5120 |
| `castBuf` | `TBuf<>` | Single (1) | FP32 累加 + 临时缓冲 | 2 × 1280 × 4 = 10240 | 10240 |

**UB 总用量**: 20480 + 5120 + 10240 = **35840 bytes ≈ 35 KB**

**UB 空闲**: 192 KB - 35 KB = **157 KB** (81.8% 空闲)

### 5.2 Buffer 初始化代码

```cpp
pipe_->InitBuffer(inQueueX, DOUBLE_BUFFER, R * tileA0Len * sizeof(half));
pipe_->InitBuffer(outQueueY, DOUBLE_BUFFER, tileA0Len * sizeof(half));
pipe_->InitBuffer(castBuf, 2 * tileA0Len * sizeof(float));
```

### 5.3 GM 内存

| 数据 | 地址空间 | 大小 |
|------|---------|------|
| 输入 o_grad | GM | A1 × R × A0 × 2 = 20,971,520 bytes (~20 MB) |
| 输出 result | GM | A1 × A0 × 2 = 5,242,880 bytes (~5 MB) |
| Tiling 数据 | GM | sizeof(ExpandKernelBwdTilingData) = 80 bytes |

**无需额外 workspace** (算子无中间 GM 暂存需求)。

### 5.4 内存访问模式

```
每 tile 的 GM 访问:
  GM Read:  R × tileA0Len × sizeof(half) = 4 × 1280 × 2 = 10240 bytes
  GM Write: tileA0Len × sizeof(half)     = 1280 × 2     = 2560 bytes

全量 GM 访问 (2048 tiles):
  Total Read:  2048 × 10240 = 20,971,520 bytes (输入全量, 每元素读 1 次)
  Total Write: 2048 × 2560  = 5,242,880 bytes  (输出全量, 每元素写 1 次)

计算密集度 = FLOPs / Bytes = 10,485,760 / 26,214,400 ≈ 0.4
→ 内存带宽瓶颈 (远低于 DAV_2201 向量计算上限)
```

---

## 6. 并行策略

### 6.1 多核并行: A1 维度切分

```
totalTiles = A1 × a0Outer = 2048 × 1 = 2048

多核分配 (以 48 核为例):
  tilesPerCore  = ceil(2048 / 48) = 43
  usedCoreNum   = ceil(2048 / 43) = 48
  tailCoreTiles = 2048 % 43       = 27

分配结果:
  Core 0 ~ 46:  各 43 tiles
  Core 47:      27 tiles (尾核)
```

### 6.2 单核内流水并行: Double Buffer

```
时间轴 ─────────────────────────────────────────────────►

Tile N:
  MTE2: [CopyIn  tile N] ───────────
  VEC:                     [Compute tile N] ──────────
  MTE3:                                   [CopyOut tile N] ───

Tile N+1:
  MTE2:         [CopyIn  tile N+1] ───────────
  VEC:                            [Compute tile N+1] ──────────
  MTE3:                                          [CopyOut tile N+1] ───

图例: MTE2(GM→UB) ⋂ VEC(Compute) ⋂ MTE3(UB→GM)
     三者流水重叠: MTE2(61.3%) + VEC(37.3%) + MTE3(26.8%) > 100%
```

**同步机制**: 零 `PipeBarrier` 调用，全部通过 `TQue::EnQue/DeQue` 隐式同步:

| 同步点 | 生产者 | 消费者 | 机制 |
|--------|--------|--------|------|
| CopyIn → Compute | MTE2 (EnQue) | VEC (DeQue) | `inQueueX` TQue |
| Compute → CopyOut | VEC (EnQue) | MTE3 (DeQue) | `outQueueY` TQue |

### 6.3 核类型约束

使用 `__vector__` 属性确保 kernel 仅在 AIV (Vector) 核上运行:

```cpp
extern "C" __global__ __vector__ void expand_kenel_bwd_kernel(...)
```

CANN 9.0.0 已移除 `GetBlockType()` / `BlockType` API，`__vector__` 是官方推荐的替代方案。

---

## 7. Tiling 策略

### 7.1 参数定义

| 参数 | 符号 | 含义 | 典型值 |
|------|------|------|--------|
| 外层保留轴 | A1 | n0 × n1 | 2048 |
| 归约轴 | R | mhc_mult | 4 |
| 内层保留轴 | A0 | h | 1280 |
| 最小对齐单位 | `A0_TILE_BASE` | VECTOR_REG_WIDTH / sizeof(half) | 128 |
| UB 切片大小 | tileA0Len | A0 维度每次处理的元素数 | 1280 |
| A0 切片份数 | a0Outer | ceil(A0 / tileA0Len) | 1 |
| 总 tile 数 | totalTiles | A1 × a0Outer | 2048 |

### 7.2 自适应 Tiling 算法

```cpp
// Step 1: 确定 tileA0Len (对齐到 A0_TILE_BASE = 128)
// 全载判定: 2×R×tileA0Len×2 + 2×tileA0Len×2 + 2×tileA0Len×4 ≤ 192KB
// 代入 R=4: 8×tileA0Len + 4×tileA0Len + 8×tileA0Len = 20×tileA0Len bytes
// 20 × tileA0Len ≤ 196608 → tileA0Len ≤ 9830
// A0=1280 < 9830, 全载条件满足
uint64_t tileA0Len = ((A0 + A0_TILE_BASE - 1) / A0_TILE_BASE) * A0_TILE_BASE;

// Step 2: A0 切片数
uint64_t a0Outer = (A0 + tileA0Len - 1) / tileA0Len;

// Step 3: 总 tile 数
uint64_t totalTiles = A1 * a0Outer;

// Step 4: 多核分配 (向上取整)
uint64_t tilesPerCore = (totalTiles + blockDim - 1) / blockDim;
uint64_t usedCoreNum  = (totalTiles + tilesPerCore - 1) / tilesPerCore;
uint64_t tailCoreTiles = totalTiles % tilesPerCore;
if (tailCoreTiles == 0 && totalTiles > 0) {
    tailCoreTiles = tilesPerCore;
}
```

### 7.3 Tiling 数据结构

```cpp
struct ExpandKernelBwdTilingData {
    uint64_t A1;            // 外层保留轴总大小 = n0 × n1
    uint64_t R;             // 归约轴大小 = mhc_mult
    uint64_t A0;            // 内层保留轴总大小 = h
    uint64_t tileA0Len;     // UB 切片 A0 大小 (对齐到 128)
    uint64_t a0Outer;       // A0 切片份数
    uint64_t totalTiles;    // 总 tile 数 = A1 × a0Outer
    uint64_t tilesPerCore;  // 每核 tile 数
    uint64_t tailCoreTiles; // 尾核 tile 数
    uint64_t usedCoreNum;   // 使用的核数
    uint32_t inputSize;     // 输入总大小 (bytes)
    uint32_t outputSize;    // 输出总大小 (bytes)
};
```

### 7.4 Tiling 数据传递

CANN 9.0.0 直调模式 (direct-invoke) 不支持 `REGISTER_TILING_DEFAULT` 等 auto-tiling 宏，因此 Tiling 数据通过 `__gm__` 指针直接传递:

```cpp
// Host 侧: malloc device memory, memcpy tiling data, 传指针给 kernel
// Kernel 侧: 直接 reinterpret_cast 读取
op.Init(oGrad, out, reinterpret_cast<__gm__ ExpandKernelBwdTilingData*>(tiling));
```

---

## 8. 精度考虑

### 8.1 数据类型与精度路径

| 阶段 | 数据类型 | 说明 |
|------|---------|------|
| GM 输入 | FP16 | 算子输入 |
| GM→UB 搬移 | FP16 | 无损搬移 |
| 类型提升 | FP16 → FP32 | `Cast(accF32, row, CAST_NONE)` |
| 累加计算 | FP32 | `Add(accF32, accF32, tmpF32, count)` × 3 |
| 类型截断 | FP32 → FP16 | `Cast(outLocal, accF32, CAST_ROUND)` |
| UB→GM 搬移 | FP16 | 无损搬移 |
| GM 输出 | FP16 | 算子输出 |

### 8.2 精度分析

**FP16 精度极限**:
- FP16 尾数: 10 bits → 相对精度约 1/1024 ≈ 9.77e-04
- 本算子 R=4，累加 4 个 FP16 值
- 单次 FP32→FP16 截断误差: ≤ 0.5 ULP ≈ 1/2048 ≈ 4.88e-04 (相对)

**为什么不会出现大数吃小数**:
- R=4 极小，累加项数远小于 FP16 精度阈值 (~1024)
- FP32 中间累加提供了足够的动态范围 (23-bit 尾数)
- 无需求助于二分累加或 Kahan 补偿

**实测精度** (25+ 用例):
- 最大绝对误差: 7.81e-03 (= 1/128)
- 误差来源: FP32 累加后单次截断为 FP16 的量化误差
- 采用精度标准: `rtol=1e-3, atol=1e-4` (FP16 浮点计算类社区标准)

### 8.3 设计偏离: 显式 FP32 vs 隐式升精度

原设计计划依赖 `Add<half>` API 的隐式升精度。实测发现 3 次连续的 `Add<half>` 在每次操作后截断回 FP16，导致中间结果精度损失。改为显式 `Cast→FP32 → Add(FP32) → Cast→FP16`，仅单次截断，精度更优。

---

## 9. API 映射表

| 步骤 | API | 关键参数 |
|------|-----|---------|
| Tiling 计算 | 自定义 `ComputeTiling()` | 运行时根据 A1/R/A0/blockDim 计算 |
| GM→UB 搬入 | `DataCopyPad` (MTE2) | `blockCount=R, blockLen=tileA0Len*2, srcStride=(A0-tileA0Len)*2` |
| 类型提升 | `Cast` (VEC) | `RoundMode::CAST_NONE`, FP16→FP32 |
| 逐元素加 | `Add` (VEC) | FP32, count=tileA0Len |
| 类型截断 | `Cast` (VEC) | `RoundMode::CAST_ROUND`, FP32→FP16 |
| UB→GM 搬出 | `DataCopyPad` (MTE3) | `blockCount=1, blockLen=tileA0Len*2` |

---

## 10. 边界与约束

| 约束 | 说明 | 处理方式 |
|------|------|---------|
| R 硬编码为 4 | 当前仅需支持 mhc_mult=4 | 3 次 Add；如需通用化改为 `for(i=0; i<R-1; i++)` |
| A0 需对齐 128 | `A0_TILE_BASE = VECTOR_REG_WIDTH / sizeof(half)` | Tiling 自动向上对齐并 Padding |
| 仅支持 FP16 | sizeof(half)=2 硬编码在 Tiling 和 Buffer 计算中 | 扩展需增加 dtype 分支 |
| `repeatTimes ≤ 2^31-1` | Add API count 上限 | tileA0Len=1280 远小于上限 |
| UB ≤ 192 KB | DAV_2201 单核 UB 容量 | 实际使用 ~35 KB，远小于上限 |
| `tilesPerCore ≤ 255` (部分约束) | 固定开销 | 当前 43 tiles，满足 |

---

## 11. 性能模型

### 11.1 理论分析

| 指标 | 数值 |
|------|------|
| 总 FLOPs | A1 × R × A0 = 2048 × 4 × 1280 = 10,485,760 |
| 总输入字节 | A1 × R × A0 × 2 = 20,971,520 |
| 总输出字节 | A1 × A0 × 2 = 5,242,880 |
| 计算密集度 | 10.49 MFLOPs / 25.6 MB ≈ 0.4 FLOPs/Byte |
| 瓶颈判定 | **内存带宽瓶颈** (计算密集度远低于 ~32 FLOPs/Byte 的向量计算上限) |

### 11.2 实测性能

| 指标 | 数值 |
|------|------|
| Task Duration | 30.061 us |
| AIV Core Time | 24.527 us |
| Head Overhead | 5.534 us (18.4%) |
| BlockDim | 48 cores |
| Pipe MTE2 (GM→UB) | 61.30% |
| Pipe VEC (Compute) | 37.30% |
| Pipe MTE3 (UB→GM) | 26.80% |
| Bank Conflict | 1.90% |

**解读**:
- MTE2(61.3%) + VEC(37.3%) > 100%，证实 Double Buffer 流水有效重叠
- 18.4% 头开销对于 30us 级极短 kernel 是合理的 (dispatch/scheduling/teardown 固定开销 ~5-6us)
- 瓶颈确认与理论分析一致: 内存带宽瓶颈

---

## 12. 架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                        Host (CPU)                                │
│  shape 解析 → 合轴 (A1,R,A0) → ComputeTiling() → <<<>>> launch  │
└──────────────────────────┬───────────────────────────────────────┘
                           │ aclrtStream
     ┌─────────────────────┼──────────────────────────┐
     ▼                     ▼                          ▼
┌──────────┐        ┌──────────┐               ┌──────────┐
│  Core 0  │        │  Core 1  │    ...        │ Core 47  │
│ A1[0:43) │        │ A1[43:86)│               │A1[2021:2048)│
│  43 tiles│        │  43 tiles│               │  27 tiles │
└──────────┘        └──────────┘               └──────────┘
     │                     │                          │
     └─────────────────────┴──────────────────────────┘
                           │
               ┌───────────┴───────────┐
               │   Per-Tile Pipeline   │
               │                       │
               │  MTE2: CopyIn         │
               │    ↓ (TQue EnQue)     │
               │  VEC:  Cast+Add+Cast  │
               │    ↓ (TQue EnQue)     │
               │  MTE3: CopyOut        │
               └───────────────────────┘
```
