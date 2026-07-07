# DESIGN.md — head_compute_mix_fwd 算子架构设计

> Architect: Ascend C 算子架构设计专家 | Date: 2026-07-01

---

## 1. 环境信息

| 项目 | 值 |
|------|-----|
| 芯片型号 | Ascend910B2 |
| NpuArch | DAV_2201 |
| SocVersion | Ascend910B2 |
| `__NPU_ARCH__` | 2201 |
| CANN 版本 | 9.0.0 |
| UB 容量 | 192 KB (196608 bytes) |
| L0C 容量 | 128 KB |
| BT 容量 | 1 KB |
| AI Core 数量 | 48 |

---

## 2. 算子语义

### 2.1 数学公式

```
output = sigmoid(input_mix * mhc_scale + mhc_base) + mhc_pre_eps
```

展开为：

```
output[i,j,k] = sigmoid(input_mix[i,j,k] * mhc_scale[0] + mhc_base[k]) + mhc_pre_eps
```

其中 `i ∈ [0, batch_size)`, `j ∈ [0, n1)`, `k ∈ [0, mhc_mult)`.

### 2.2 输入输出规格

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| input_mix | [batch_size, n1, mhc_mult] | FP16 | 主输入张量 |
| mhc_scale | [1] | FP16 | 标量缩放因子 |
| mhc_base | [mhc_mult] | FP16 | 逐通道偏置（mhc_mult=4） |
| mhc_pre_eps | scalar | FP32 | eps 常量 |
| output | [batch_size, n1, mhc_mult] | FP16 | 输出 |

默认 shape: `batch_size=16, n1=16384, mhc_mult=4`, 总元素数 = 1,048,576.

### 2.3 计算特性

- **逐元素独立计算**：每个输出元素仅依赖同位置的输入元素 + 对应的 mhc_base[k]
- **无跨元素依赖**：不需要归约、排序等跨元素操作
- **含超越函数**：sigmoid 涉及 `exp()`，需要 FP32 中间精度

---

## 3. 技术路线决策

### 3.1 决策过程

| 步骤 | 判断 | 结论 |
|------|------|------|
| Step 0: 架构检查 | NpuArch = DAV_2201 | 非 DAV_3510，RegBase/Blaze 不可用 |
| Step 0.5: 算子类型 | 逐元素 + 广播 | Elementwise/Broadcast 混合 |
| Step 0.5: 路线决策 | DAV_2201 + vector 类 | **通用 SIMD/MemBase 路线** |

**决策理由**：
- DAV_2201 不支持 RegBase（DAV_3510 独有）和 Blaze/tensor_api（DAV_3510 的 Matmul/Cube 路径）
- 算子核心为逐元素计算（乘加 + sigmoid），属 vector 类，走通用 SIMD/MemBase

### 3.2 算子类别判定

算子含两种形态的输入：
- `input_mix` [16, 16384, 4] — 3D 主张量
- `mhc_scale` [1] — 标量（→ 作为 tiling 参数传递）
- `mhc_base` [4] — 1D 偏置（→ 需要广播到 3D）
- `mhc_pre_eps` — 标量（→ 作为 tiling 参数传递）

按 Broadcast patterns.md 的合轴流程：

```
补维后 shapes:
  input_mix:  [16, 16384, 4]   strides: [65536, 4, 1]
  mhc_base:   [1,  1,     4]   strides: [0, 0, 1]

Flag 计算 (bit0=mhc_base, bit1=input_mix):
  轴0 (16):    input_mix≠1, mhc_base=1 → flag=01
  轴1 (16384): input_mix≠1, mhc_base=1 → flag=01
  轴2 (4):     input_mix≠1, mhc_base≠1 → flag=00

合轴（轴0与轴1 flag 相同 → 合并）:
  output dims: [262144, 4]
  input_mix:   [262144, 4]  strides: [4, 1]
  mhc_base:    [1, 4]       strides: [0, 1]
```

### 3.3 策略选择：展平 1D Elementwise

mhc_base 的内维大小仅为 4，远小于 UB 容量。mhc_scale 和 mhc_pre_eps 为标量。因此：

**采用展平 1D Elementwise 策略**：
- 将 3D 张量展平为 1D: `dim0 = batch_size * n1 * mhc_mult = 1,048,576`
- mhc_base[4] 预加载到 UB 并扩展至 tile 大小，用步进式 Duplicate 实现
- mhc_scale / mhc_pre_eps 通过 TilingData 作为标量传入 kernel
- 计算链路完全对标 Elementwise 1D tiling 方法论

**为什么不走 Broadcast DAV_2201 路径**：
- Broadcast 静态接口要求 srcShape[1] × sizeof(T) 为 32B 对齐
- mhc_base 内维大小=4，FP16 下 4×2=8B，FP32 下 4×4=16B，均不满足 32B 对齐
- 虽然可通过 DataCopyPad + Copy + GatherMask 兜底，但复杂度高
- 展平 1D + Duplicate 扩展更简洁，在 UB 容量（192KB）充足的前提下无额外代价

---

## 4. Tiling 设计

### 4.1 多核切分

遵循 Elementwise tiling 的标准公式（elewise/tiling.md）：

```cpp
constexpr int64_t MIN_TILING_BITS = 32768;  // 4KB，单位 bits
constexpr int64_t ELEM_ALIGN_FACTOR = 512;  // 多核元素对齐因子

dim0 = batch_size * n1 * mhc_mult;  // 1,048,576  (FP16: minDtypeBits=16)

coreNum = min(
    (dim0 * 16 + MIN_TILING_BITS - 1) / MIN_TILING_BITS,
    availableCoreNum  // 最大 24
);

blockFormer = ((dim0 + coreNum - 1) / coreNum + ELEM_ALIGN_FACTOR - 1)
              / ELEM_ALIGN_FACTOR * ELEM_ALIGN_FACTOR;
blockNum = (dim0 + blockFormer - 1) / blockFormer;
blockTail = dim0 - (blockNum - 1) * blockFormer;
```

**示例计算**（FP16, dim0=1,048,576, availableCoreNum=24）:
```
coreNum = min((1048576*16 + 32767)/32768, 24) = min(512, 24) = 24
blockFormer = ceil(ceil(1048576/24) / 512) * 512 = ceil(43691/512) * 512 = 86 * 512 = 44032
blockNum = ceil(1048576 / 44032) = 24
blockTail = 1048576 - 23 * 44032 = 1048576 - 1012736 = 35840
```

### 4.2 UB 切分

```cpp
constexpr int64_t ALIGN_256 = 256;  // UB 256B 对齐

// bufferNum = 计算图中存活 buffer 份数（按 FP32 等效宽度计）
// ubFormer = UB 单次处理元素数（按 FP16 对齐）

bufferDivisor = 等效 buffer 总字节数（见第 5 节 Buffer 规划）
maxElemNum = (ubSize * 8) / bufferDivisor;
alignFactor = ALIGN_256 * 8 / minDtypeBits;  // FP16: 256*8/16 = 128
ubFormer = (maxElemNum / alignFactor) * alignFactor;

// 首 block 循环
ubLoopOfFormerBlock = (blockFormer + ubFormer - 1) / ubFormer;
ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer;

// 尾 block 循环
ubLoopOfTailBlock = (blockTail + ubFormer - 1) / ubFormer;
ubTailOfTailBlock = blockTail - (ubLoopOfTailBlock - 1) * ubFormer;
```

**关键对齐要求**：ubFormer 必须同时是 4 的倍数（因 mhc_mult=4，保证 Duplicate 扩展后的 pattern 对齐）。

### 4.3 TilingData 结构

```cpp
struct TilingData {
    int64_t dim0;                    // 元素总数
    int32_t coreNum;                 // 实际使用核数
    int64_t blockFormer;             // 每核基础元素数（512 对齐）
    int64_t blockNum;                // block 总数
    int64_t blockTail;               // 尾 block 元素数
    int64_t ubFormer;                // UB tile 基础大小（256B 对齐）
    int64_t ubLoopOfFormerBlock;     // 首 block 内循环次数
    int64_t ubTailOfFormerBlock;     // 首 block 尾部元素数
    int64_t ubLoopOfTailBlock;       // 尾 block 内循环次数
    int64_t ubTailOfTailBlock;       // 尾 block 尾部元素数
    float  mhc_scale_f32;            // FP32 mhc_scale
    float  mhc_pre_eps_f32;          // FP32 mhc_pre_eps
    half   mhc_base_f16[4];          // mhc_base (FP16 原始)
};
```

---

## 5. UB Buffer 规划

### 5.1 Buffer 清单

采用 **Double Buffer** 策略，输入/输出流水线化。

| Buffer 名称 | 数据类型 | 元素数 | 字节数 | 用途 |
|------------|---------|--------|--------|------|
| `inQueue` (×2) | half | ubFormer | 2 × uF × 2B | 输入双缓冲（TQue, VECIN） |
| `f32WorkBuf` (×2) | float | ubFormer | 2 × uF × 4B | FP32 计算双缓冲（TBuf, VECCALC）|
| `baseF32Expanded` | float | ubFormer | uF × 4B | mhc_base 扩展结果（只写一次） |
| `outQueue` (×2) | half | ubFormer | 2 × uF × 2B | 输出双缓冲（TQue, VECOUT） |

### 5.2 UB 容量计算

```
bufferDivisor = 2 * sizeof(half)      // inQueue 双缓冲 (每份)
              + 2 * sizeof(float)     // f32WorkBuf 双缓冲 (每份)
              + 1 * sizeof(float)     // baseF32Expanded (每份)
              + 2 * sizeof(half)      // outQueue 双缓冲 (每份)
              = 4 + 8 + 4 + 4 = 20 bytes per element

ubSize = 192 * 1024 = 196608 bytes
maxElemNum = 196608 * 8 / (20 * 8) = 196608 / 20 ≈ 9830
alignFactor = 256 / sizeof(half) = 128   // 256B / 2B per half
ubFormer = (9830 / 128) * 128 = 76 * 128 = 9728

验证 4 的倍数: 9728 / 4 = 2432 ✓
```

### 5.3 Double Buffer 流水线

```
Time ──────────────────────────────────────────────►

Stream 0:  [CopyIn tile_0][CopyIn tile_2]...
Stream 1:  [CopyIn tile_1][CopyIn tile_3]...

Compute:           [Compute tile_0][Compute tile_1][Compute tile_2]...

Stream 2:                    [CopyOut tile_0][CopyOut tile_1]...
Stream 3:                                    [CopyOut tile_2]...
```

使用 `EnQue` / `DeQue` 同步 MTE 搬入搬出与 Vector 计算流水线。

---

## 6. API 映射

### 6.1 算术运算

| 数学运算 | AscendC API | 调用方式 | 验证状态 |
|---------|------------|---------|---------|
| `input * scale` | `AscendC::Muls` | `Muls(dst, src, scalar, count)` | 已验证（api-arithmetic.md） |
| `vec + base` | `AscendC::Add` | `Add(dst, src0, src1, count)` | 已验证（api-arithmetic.md） |
| `-x` | `AscendC::Muls` | `Muls(dst, src, -1.0f, count)` | 已验证（用 Muls 替代 Neg） |
| `exp(x)` | `AscendC::Exp` | `Exp(dst, src, count)` | 已验证（api-restrictions.md §1.1） |
| `scalar + vec` | `AscendC::Adds` | `Adds(dst, src, scalar, count)` | 已验证（api-arithmetic.md §场景1） |
| `1.0 / vec` | `AscendC::Div` | `Div(dst, ones_vec, vec, count)` | 已验证（api-arithmetic.md；需 ones_vec 缓冲） |

### 6.2 数据搬运

| 操作 | AscendC API | 说明 |
|------|------------|------|
| GM→UB 搬入 | `DataCopyPad` | 推荐，自动处理对齐/非对齐 |
| UB→GM 搬出 | `DataCopyPad` | 同上 |
| half↔float 转换 | `AscendC::Cast` | `Cast<float, half>(..., CAST_NONE)` / `Cast<half, float>(..., CAST_ROUND)` |

### 6.3 向量扩展

| 操作 | 方案 | 说明 |
|------|------|------|
| mhc_base[4]→uF 扩展 | 步进 `Duplicate` | 4→N×2→...→ubFormer，每步 repeatTimes≤255 |
| ones 向量填充 | `Duplicate` | `Duplicate(ones_buf, 1.0f, ubFormer)` |

### 6.4 Sigmoid 计算链

```
// x = input * scale + base  (已在 f32WorkBuf 中)

// Step 1: neg = -x
Muls(f32WorkBuf, f32WorkBuf, -1.0f, ubFormer);

// Step 2: exp_val = exp(-x)
Exp(f32WorkBuf, f32WorkBuf, ubFormer);

// Step 3: denom = 1.0 + exp(-x)
Adds(f32WorkBuf, f32WorkBuf, 1.0f, ubFormer);

// Step 4: sigmoid = 1.0 / denom
//        (ones_buf 已预先填充 1.0f)
Div(f32WorkBuf, onesF32Buf, f32WorkBuf, ubFormer);

// Step 5: result = sigmoid + eps
Adds(f32WorkBuf, f32WorkBuf, mhc_pre_eps_f32, ubFormer);
```

**数值稳定性考量**：
- Exp 在 FP16 下对极端输入可能溢出；全链路 FP32 中间计算避免此问题
- Sigmoid 对大正输入趋向 1，大负输入趋向 0，FP32 足够稳定
- 可以额外考虑对大负数做 clip（exp(-x) 极大时溢出），但 AscendC Exp 在 FP32 下输入范围约 [-87, 88]，输入 `x = input * scale + base` 若来自归一化后的 attention score（通常 [-5, 5]），不会达到溢出边界

---

## 7. Kernel 执行流程

### 7.1 初始化阶段（Kernel 入口）

```
1. 解析 TilingData，提取所有参数
2. 计算当前 core 的 GM 偏移: offset = blockFormer * blockIdx * sizeof(half)
3. 加载 mhc_base_f16[4] → Cast → mhc_base_f32[4]
4. 步进 Duplicate 扩展 mhc_base_f32 到 ubFormer 元素 → baseF32Expanded
5. 用 Duplicate 填 onesF32Buf (全部 1.0f, ubFormer 元素)
```

### 7.2 主循环（Double Buffer 流水线）

```
for tile_idx in [0, ubLoopOfFormerBlock):
    currentSize = (isLastTile && isTailBlock) ? ubTail : ubFormer

    // --- CopyIn Phase ---
    DataCopyPad(inQueue[ping], inputGm[offset], currentSize * sizeof(half))
    EnQue(inQueue[ping])

    // --- Compute Phase (on previous tile) ---
    DeQue(inQueue[pong])
    Cast<float, half>(f32WorkBuf, inQueue[pong], CAST_NONE, currentSize)
    Muls(f32WorkBuf, f32WorkBuf, mhc_scale_f32, currentSize)  // * scale
    Add(f32WorkBuf, f32WorkBuf, baseF32Expanded, currentSize) // + base
    [Sigmoid 计算链: §6.4, Steps 1-5]
    Cast<half, float>(outQueue[ping], f32WorkBuf, CAST_ROUND, currentSize)
    EnQue(outQueue[ping])

    // --- CopyOut Phase (on previous-previous tile) ---
    DeQue(outQueue[pong])
    DataCopyPad(outputGm[offset_prev], outQueue[pong], prevSize * sizeof(half))

    // Advance offset, ping-pong flip
    offset += currentSize
```

### 7.3 收尾

处理最后 1-2 个 tile 的 DeQue + CopyOut，确保所有数据已写出。

---

## 8. 精度策略

### 8.1 混合精度方案

```
FP16 输入 → Cast(FP32, CAST_NONE)
  → FP32 中间计算（Mul + Add + Sigmoid）
  → Cast(FP16, CAST_ROUND)
  → FP16 输出
```

**依据**（api-precision.md）：
- Sigmoid 中 Exp 在 FP16 下极易溢出（exp(11) ≈ 59874 > 65504 FP16 max）
- AscendC 半精度加减法默认升精度策略
- 非"同量级"场景，不能直接用 FP16 直算

### 8.2 精度标准

- 算子类型：浮点计算类社区标准（ops-precision-standard）
- 对标：PyTorch FP32 参考实现 → FP16 AscendC 实现
- 容许误差：绝对误差 ≤ 1e-3, 相对误差 ≤ 1e-2（sigmoid 的数值特性下合理预期）

---

## 9. 常见陷阱与规避

| 陷阱 | 规避方案 |
|------|---------|
| Duplicate repeatTimes > 255 | 步进式扩展（4→512→ubFormer），每步 repeat≤255 |
| Broadcast 静态接口 32B 对齐不满足 | 不使用 Broadcast，改用展平 1D + Duplicate |
| Exp FP16 溢出 | 全链路 FP32 中间计算 |
| Tail block 尺寸 ≠ ubFormer | 区分首/尾 block，使用正确的 currentSize |
| Double buffer ping-pong 错位 | 严格遵循 EnQue/DeQue 配对 |

---

## 10. 文件结构

```
operators/head_compute_mix_fwd/
├── docs/
│   ├── DESIGN.md          ← 本文件
│   └── PLAN.md
├── CMakeLists.txt
├── head_compute_mix_fwd_tiling.h    # TilingData 结构 + ComputeTiling()
├── head_compute_mix_fwd.cpp         # Host 侧入口
└── head_compute_mix_fwd_kernel.h    # Device 侧 Kernel 实现
```
