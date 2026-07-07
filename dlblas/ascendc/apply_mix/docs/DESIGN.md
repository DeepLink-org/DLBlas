# apply_mix 算子架构设计 (DESIGN.md)

> **版本**: v5.0 | **日期**: 2026-07-01 | **架构**: Ascend910B2 (DAV_2201) | **CANN**: 9.0.0

---

## 一、算子分析

### 1.1 数学定义

```
output(n0, n1, h) = sum_{mhc} ( x(n0, n1, mhc, h) * mix(n0, n1, mhc, 1) ).bfloat16()
```

包含三个语义步骤：
1. **Broadcast Multiply**: `x [bfloat16, n0,n1,mhc,h] * mix [float32, n0,n1,mhc,1]` -- mix 最末维从 1 广播到 h
2. **Reduction Sum**: `.sum(-2)` -- 沿 mhc 维度求和
3. **Type Conversion**: `.bfloat16()` -- 结果转为 bfloat16

### 1.2 输入输出规格

| 张量 | Shape | Dtype | 语义 |
|------|-------|-------|------|
| `x` | `[n0, n1, mhc, h]` | `bfloat16` | 输入特征张量 |
| `mix` | `[n0, n1, mhc, 1]` | `float32` | 混合权重（最末维为 1，需广播） |
| `output` | `[n0, n1, h]` | `bfloat16` | 加权求和结果 |

### 1.3 典型 Shape

| 参数 | 值 | 说明 |
|------|-----|------|
| n0 | 2 | batch outer |
| n1 | 1024 | batch inner |
| mhc (R) | 4 | 归约轴，极小 |
| h (A0) | 1280 | 特征维度，较大 |
| A1 (= n0*n1) | 2048 | 展平 batch 维，提供充足并行度 |

---

## 二、技术路线决策

### 2.1 架构信息

| 项目 | 值 | 获取方式 |
|------|-----|---------|
| 芯片型号 | Ascend910B2 | 用户提供 |
| NpuArch | `DAV_2201` | `/npu-arch` skill: Ascend910B2 -> DAV_2201 |
| SocVersion | `Ascend910B2` | `PlatformAscendC::GetSocVersion()` |
| `__NPU_ARCH__` | `2201` | Device 侧编译宏 |
| `--npu-arch` | `dav-2201` | CMake 编译选项 |
| UB 容量 | 192 KB (196608 B) | DAV_2201 固定值 |
| L0C 容量 | 128 KB | DAV_2201 固定值 |
| Vector 核数 | 24 | Ascend910B2 参数，通过 `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` 动态获取 |
| CANN 版本 | 9.0.0 | 用户提供 |

### 2.2 路线判定

```
算子类型分析:
  ├─ 主计算: Broadcast(Mul) + Reduction(Sum) + Conversion(bf16)
  ├─ 无 MatMul/Cube 计算 → 排除 Blaze/tensor_api
  ├─ 无 DAV_3510 → 排除 RegBase 路线
  └─ 判定: SIMD/MemBase 路线

架构分支:
  ├─ DAV_3510 (Ascend950) → 不适用 (目标为 Ascend910B2)
  ├─ RegBase 路线 → 不可用 (DAV_3510 专属)
  └─ SIMD/MemBase 路线 → 选中
```

**决策结论**: **SIMD/MemBase 路线**，采用 TPipe + TQue + DataCopyPad 标准架构。

### 2.3 算子类型归类（合轴分析）

按 `/ascendc-tiling-design` 的 Reduction 场景路由：

```
标记 A(保留轴) / R(归约轴):
  axis=0(n0): A  }
  axis=1(n1): A  } → 相邻同类型，合并为 A1 = n0 * n1
  axis=2(mhc): R
  axis=3(h):   A → A0

合轴结果: ARA 模式 (A1=2048, R=4, A0=1280)
```

按 [ARA 场景路由](references/reduction/patterns.md) 判定：A0 > 1, R 可全载 → **ARA-FullLoad** 分支。

### 2.4 融合策略选型

本算子是 Broadcast Mul + Reduction Sum 的融合算子。由于 R=4 极小，有两种实现路径：

| 方案 | 描述 | DAV_2201 适用性 |
|------|------|:---:|
| **ReduceSum RA** | ARA-FullLoad 整块载入 [R,alignedCols]，调用 `ReduceSum<Pattern::Reduce::RA>` | 不可用：TBuf 显式 tmpBuf 产生精度异常 (MARE~10)；PopStackBuffer 隐式分配引入 94.7% 标量瓶颈 |
| **per-row Muls+Add** | 逐行 `Muls(scalar_broadcast)` + `Add` 累加，R=4 时仅 7 次向量化 API 调用 | 可用：无 LCM 分配开销，全向量化，精度验证通过 (MERE=MARE=0) |

**选定方案**: **per-row Muls+Add 手动累加**。

选择理由：
1. DAV_2201 上 ReduceSum RA 的 tmpBuf 路径存在兼容性问题（TBuf 精度异常，PopStackBuffer 标量开销不可接受）
2. R=4 极小，per-row 循环开销可忽略（仅 7 次向量化 API 调用 / tile）
3. 中间数据就地修改，无需额外 broadcast buffer
4. 精度已验证：MERE=MARE=0（位精确）

---

## 三、架构设计

### 3.1 总体数据流

```
┌──────────────────────────────────────────────────────────────┐
│ GM 层 (所有数据以 fp32 存储)                                    │
│  x_fp32 [A1, R, A0]              mix_fp32 [A1, R]             │
│       │                                │                       │
│  ┌────▼────────────────────────────────▼───────────────────┐  │
│  │  Kernel 层 (Device, DAV_2201, SIMD/MemBase)              │  │
│  │                                                           │  │
│  │  ┌─ CopyIn (Double Buffered) ──────────────────────────┐ │  │
│  │  │  DataCopyPad: x_tile [R, alignedCols] fp32          │ │  │
│  │  │  DataCopyPad: mix_buf [R] fp32 (仅 batch 变化时)     │ │  │
│  │  └──────────────────────────────────────────────────────┘ │  │
│  │                         │                                  │  │
│  │  ┌─ Compute ───────────▼────────────────────────────────┐ │  │
│  │  │  r=0: result = xData[0] * mixVals[0]   (Muls 初始化) │ │  │
│  │  │  r=1..R-1:                                         │ │  │
│  │  │    row = xData[r]                                   │ │  │
│  │  │    row = Muls(row, mixVals[r])      (就地标量广播乘)  │ │  │
│  │  │    result = Add(result, row)        (累加)           │ │  │
│  │  └──────────────────────────────────────────────────────┘ │  │
│  │                         │                                  │  │
│  │  ┌─ CopyOut (Double Buffered) ───▼──────────────────────┐ │  │
│  │  │  DataCopyPad: result [tileA0Len] fp32 → GM           │ │  │
│  │  └──────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│                               │                                │
│  output_fp32 [A1, A0]                                          │
│       │                                                        │
│  ┌────▼────────────────────────────────────────────────────┐  │
│  │  Host/PyTorch 层: fp32 → bfloat16 (round-to-nearest-even) │  │
│  └──────────────────────────────────────────────────────────┘  │
│                               │                                │
│  output_bf16 [A1, A0]                                          │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 多核切分策略

**切分轴**: 沿 A1 维度（fused batch 维 `n0*n1`）均分。当 A1 不足以充分利用多核时，A0 维度切分兜底。

```
a0Outer = ceil(A0 / tileA0Len)           # A0 维度 tile 数
totalTiles = A1 * a0Outer                # 总 tile 数
tilesPerCore = ceil(totalTiles / coreNum) # 每核处理 tile 数
usedCoreNum = ceil(totalTiles / tilesPerCore)
usedCoreNum = min(usedCoreNum, coreNum)   # 严格不超过实际核数
```

每个 Core 处理 `tilesPerCore` 个 tile（尾部 Core 可能少一些），各 Core 独立归约，无需跨核通信。

**典型 Shape (A1=2048, tileA0Len=1280, coreNum=24)**:
- `a0Outer = ceil(1280/1280) = 1`
- `totalTiles = 2048 * 1 = 2048`
- `tilesPerCore = ceil(2048/24) = 86`
- `usedCoreNum = ceil(2048/86) = 24` (= coreNum)
- 尾部 Core 处理 `2048 - 23*86 = 70` tiles（负载差异 < 20%）

### 3.3 UB 切分策略

**切分轴**: 沿 A0 维度。R 维度（mhc=4）极小，完整放入 UB。

**tileA0Len 动态计算** (Host 侧 Tiling):

```
UB 容量方程 (DAV_2201, UB = 196608 B):

  inQueueX:  2 * R * alignedCols * sizeof(float)  = 8*R*alignedCols    (Double Buffer)
  mixQ:      1 * R * sizeof(float)                 = 4*R                (Single Buffer)
  outQueueY: 2 * alignedCols * sizeof(float)       = 8*alignedCols      (Double Buffer)
  overhead:                                          UB_OVERHEAD (~512 B)

  总 UB ≈ 8*(R+1)*alignedCols + 4*R + UB_OVERHEAD

  → tileA0Len ≤ (UB_SIZE - 4*R - UB_OVERHEAD) / (8*(R+1))
  → tileA0Len = min(max_tile, A0)，然后 64 对齐到 MIN_TILE_A0
```

对于 R=4: `tileA0Len ≤ (196608 - 16 - 512) / 40 ≈ 4902`，64 对齐后 maxTileA0Len = 4864。
对于典型 h=1280: `tileA0Len = 1280`（完整 A0 单 tile，无需 A0 切分）。

### 3.4 Buffer 规划

| Buffer | 名称 | TQue 配置 | 大小（byte） | 用途 |
|--------|------|-----------|-------------|------|
| `inQueueX_` | x_tile | `TQue<VECIN, 2>` | `2 * R * alignedCols * 4` | x 块搬入，Double Buffer |
| `mixQ_` | mix_buf | `TQue<VECIN, 1>` | `R * 4` | mix 权重单次搬入，Single Buffer |
| `outQueueY_` | result | `TQue<VECOUT, 2>` | `2 * alignedCols * 4` | 累加结果搬出，Double Buffer |

其中 `alignedCols = ((tileA0Len * 4 + 31) / 32) * 32 / 4`（32B 对齐）。对于 tileA0Len 已 64 对齐的场景，`alignedCols == tileA0Len`。

**Kernel 栈变量**:

| 变量 | 大小（byte） | 用途 |
|------|-------------|------|
| `mixVals[MAX_MHC_R]` | `MAX_MHC_R * 4` (= 128 for MAX_MHC_R=32) | 缓存当前 batch 的 mix 标量值 |

**总 UB 使用量** (典型 shape, tileA0Len=1280, R=4):
```
= 8*4*1280 + 4*4 + 8*1280 = 40960 + 16 + 10240 = 51216 B ≈ 50.0 KB
UB 余量: (192 - 50) / 192 = 74%
```

### 3.5 流水线设计 (Double Buffer)

```
时序 ────────────────────────────────────────────────►

Iteration 0:
  [MTE2: CopyIn x_tile[0]]
                      [V: Compute tile[0]]
                                              [MTE3: CopyOut result[0]]

Iteration 1:
  [MTE2: CopyIn x_tile[1]]
                      [V: Compute tile[1]]
                                              [MTE3: CopyOut result[1]]
...
```

**同步机制**: 通过 `TQue::EnQue`/`DeQue` 隐式同步。
- `inQueueX_` (TQue<VECIN, 2>): DeQue 阻塞等待 MTE2 搬运完成
- `outQueueY_` (TQue<VECOUT, 2>): DeQue 阻塞等待 V pipe 计算完成

### 3.6 GM 偏移计算

所有数据 Kernel 侧均为 fp32 类型（Host 层已做 bf16->fp32 转换）：

```
A1 = n0 * n1, R = mhc, A0 = h

合轴后 layout (A1, R, A0)，stride（以 fp32 元素为单位）:
  stride_A1 = R * A0
  stride_R  = A0
  stride_A0 = 1

对于 tile (a1_pos, a0_start):
  x_gm_addr  = x_base + (a1_pos * R * A0 + a0_start)
  搬入参数:   blockCount = R
             blockLen   = tileA0Len * 4 (byte)  [正常块]
                        = act * 4 (byte)         [尾块]
             srcStride  = (A0 - tileA0Len) * 4 (byte)

  mix_gm_addr = mix_base + (a1_pos * R)
  搬入参数:   blockCount = 1
             blockLen   = R * 4 (byte)

  y_gm_addr   = y_base + (a1_pos * A0 + a0_start)
  搬出参数:   blockCount = 1
             blockLen   = act * 4 (byte)
```

### 3.7 分支场景覆盖

| 分支条件 | 场景 | 处理策略 |
|---------|------|---------|
| `R <= MAX_MHC_R` | 正常 mhc 值 (典型 R=4) | 主路径: per-row Muls+Add |
| `R > MAX_MHC_R` | mhc 超限 | Host Tiling 阶段 clamp 到 MAX_MHC_R（防御性设计） |
| `tileA0Len == A0` | h 维度单 tile | a0Outer=1，单次 Compute |
| `tileA0Len < A0` | h 维度多 tile | A0 循环，尾块用实际长度 `act` |
| `A0 % tileA0Len != 0` | 尾块不足 | `act = A0 - a0s`; xTile Duplicate 零初始化后逐行 DataCopyPad |
| `A1 < coreNum` | batch 不足 | 减小 tileA0Len 增加 A0 outer tiles，充分利用多核 |
| `a1 变化` | 跨 batch 边界 | 重新加载 mix 值到 mixQ (mix caching) |

### 3.8 尾块处理

当 tile 所在的 A0 尾块长度 `act < tileA0Len`:

1. `Duplicate<float>(xTile, 0.0f, R_ * alignedCols_)` — 将整个 xTile 零初始化
2. `PipeBarrier<PIPE_V>()` — 确保 Duplicate 完成后再执行 MTE2 操作
3. 逐行 `DataCopyPad(row, rowGm, {1, act*4, 0, 0}, {false,0,0,0})` — 每行仅搬入 act 个有效元素
4. 计算时使用实际长度 `act`（而非 tileA0Len）
5. CopyOut 时使用实际长度 `act`

逐行搬入而非多块搬入的原因：尾块 `blockLen` 可能非 32B 对齐，多块 DataCopyPad 在非对齐场景下产生错误结果。

---

## 四、精度设计

### 4.1 精度标准

按 `/ops-precision-standard` 决策树判定：
- 输入: bf16 (x) / fp32 (mix)，输出: bf16
- 均为浮点，用户未指定商用标准
- → **浮点计算类社区标准**

| 指标 | 阈值 | 数值 |
|------|------|------|
| MERE (平均相对误差) | 2^-7 | 0.0078125 |
| MARE (最大相对误差) | 10 * 2^-7 | 0.078125 |

### 4.2 混合精度策略

```
输入 x (bf16 in GM)
    │
    ▼ [Host/PyTorch 层: bf16 → fp32 位扩展, 无损]
x_fp32 in GM
    │
    ▼ [Kernel 层: Muls + Add, 全 fp32 计算]
result_fp32 in GM
    │
    ▼ [Host/PyTorch 层: fp32 → bf16, round-to-nearest-even]
output (bf16)
```

**设计理由**:
1. **DAV_2201 平台约束**: bf16 SIMD 算术运算不可用（`Muls<bf16>`, `Add<bf16>` 等不支持），Kernel 内部必须用 fp32
2. **bf16->fp32 无损**: 位扩展不引入误差 (`(uint32_t)bf16_val << 16`)
3. **R=4 累加稳定**: fp32 累加仅 4 项，无"大数吃小数"风险
4. **Host 侧类型转换**: bf16<->fp32 为元素级位转换，不改变张量结构，不违反 C9 约束（禁止对输入 tensor 做结构预处理）

### 4.3 数值稳定性分析

| 风险 | 分析 | 结论 |
|------|------|------|
| 累加误差 | R=4, fp32 累加仅 4 项 | 无需特殊处理 |
| 乘积累加 | 每项 x(i)*mix(i) 在 fp32 下精确 | 无需 Kahan 求和 |
| bf16 截断 | 输出 bf16 有量化误差（mantissa 7bit） | 在 MERE/MARE 阈值内 |
| INF/NaN | mix 经 softmax -> (0,1); x 经 sigmoid -> (0,1) | 正常输入无极端值风险 |

---

## 五、API 映射

### 5.1 Kernel 侧 API

| 功能 | API | 验证方式 | 关键参数 |
|------|-----|:---:|----------|
| x 块搬入（正常块） | `DataCopyPad` | 编译+测试通过 | `blockCount=R, blockLen=tLen*4, srcStride=(A0-tLen)*4` |
| x 块搬入（尾块） | `DataCopyPad`（逐行） | 编译+测试通过 | `blockCount=1, blockLen=act*4` |
| mix 搬入 | `DataCopyPad` | 编译+测试通过 | `blockCount=1, blockLen=R*4` |
| 尾块零初始化 | `Duplicate<float>` | 编译+测试通过 | `Duplicate(xTile, 0.0f, R * alignedCols)` |
| V pipe 同步 | `PipeBarrier<PIPE_V>()` | 编译+测试通过 | 确保 Duplicate 完成后再 MTE2 |
| 标量广播乘法 | `Muls<float>` | 编译+测试通过 | `Muls(dst, src, scalar, count)` — R 次/ tile |
| 向量加法 | `Add<float>` | 编译+测试通过 | `Add(result, result, row, count)` — (R-1) 次/ tile |
| 结果搬出 | `DataCopyPad` | 编译+测试通过 | `blockCount=1, blockLen=act*4` |
| 流水同步 | `EnQue` / `DeQue` | 编译+测试通过 | `TQue<VECIN, 2>` / `TQue<VECOUT, 2>` Double Buffer |
| mix 标量提取 | `GetValue` (UB) | 编译+测试通过 | `mData.GetValue(r)` — 仅 R 次/batch |

### 5.2 Host 侧 API

| 功能 | API | 说明 |
|------|-----|------|
| 核数获取 | `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` | 动态获取可用 AI Core 数量 |
| 设备内存管理 | `aclrtMalloc` / `aclrtFree` | GM 分配/释放 |
| 数据搬运 | `aclrtMemcpy` | H2D / D2H 拷贝 |
| 流管理 | `aclrtCreateStream` / `aclrtSynchronizeStream` | Kernel 启动与同步 |
| bf16->fp32 转换 | `bf16_to_fp32_cpu()` | Host 侧: `(uint32_t)bf16_val << 16` |
| fp32->bf16 转换 | `fp32_to_bf16_cpu()` | Host 侧: round-to-nearest-even |

### 5.3 禁用/受限 API 说明

| API | 状态 | 说明 |
|-----|------|------|
| `ReduceSum<RA>` | **禁用** (本算子) | DAV_2201 上 TBuf/PopStackBuffer 均不兼容 |
| `SetValue(GM)` | **禁止** (全平台) | 效率极低 |
| `GetValue(GM)` | **禁止** (全平台) | 效率极低 |
| `DataCopy(GM<->UB)` | **禁止** (非对齐数据) | 本算子 x 搬入 srcStride 非零，不符合严格 32B 对齐条件 |
| `GetValue(UB)` | **受限使用** | 仅用于 mix 标量提取 (R 次/batch, <= 32 次)，禁止逐元素使用 |

---

## 六、架构约束

| # | 约束 | 说明 |
|---|------|------|
| C1 | SIMD/MemBase 架构 | Kernel 使用 TPipe + TQue + DataCopyPad 体系 |
| C2 | bf16 类型转换 | Kernel 全 fp32 运算；Host/PyTorch 层完成 bf16<->fp32 转换 |
| C3 | 禁止 Host 侧结构预处理 | bf16<->fp32 为元素级位转换，不违反 C9 |
| C4 | 32B 对齐 | 所有 Buffer 大小使用 alignedCols (32B 对齐)；DataCopyPad 处理尾部非对齐 |
| C5 | blockNum <= coreNum | usedCoreNum 严格不超过实际核数 |
| C6 | Double Buffer | inQueueX 和 outQueueY 使用 TQue<..., 2> |
| C7 | R <= MAX_MHC_R (=32) | Tiling 阶段 clamp，防止 kernel 栈 mixVals 数组溢出 |
| C8 | repeatTimes <= 255 | Muls/Add 的 repeatTimes = ceil(alignedCols/64) <= 255（tileA0Len=4864 -> 76, 满足） |
| C9 | 禁止对输入 tensor 做转置/reshape 等结构变换 | 设计原则 |

---

## 七、性能分析

### 7.1 计算量

- 每个 tile: R=4 次 Muls + (R-1)=3 次 Add = 7 次向量化 API 调用
- 每次调用处理 tileA0Len=1280 个元素（repeatTimes = 1280/64 = 20）
- 总向量操作: 2048 tiles * 7 次 * 20 repeats = 286,720 次向量指令 / kernel

### 7.2 瓶颈分析

| 瓶颈 | 说明 | 缓解措施 |
|------|------|---------|
| MTE2 带宽 (GM->UB) | 主要搬运耗时 | Double Buffer 重叠搬运与计算 |
| Per-tile 标量调度 | AllocTensor/FreeTensor/EnQue/DeQue 每 tile 开销 | 不可消除（SIMD/MemBase 固有开销） |
| Scalar ratio | EnQue->DeQue 紧邻导致流水无法充分重叠 | 已做 Double Buffer，进一步优化需预取模式 |

### 7.3 性能预期

基于 v3.0/v4.0 实测数据：
- Task Duration: ~92-235 us（取决于编译器版本和测量方法）
- vec_ratio: 3.86-14.5%
- 对于 R=4 极小归约轴，计算量本身很小，内存搬运占主导

---

## 八、测试策略

### 8.1 测试用例矩阵

| Case | n0 | n1 | mhc (R) | h (A0) | 覆盖场景 |
|------|----|----|---------|--------|---------|
| TC-1 | 2 | 1024 | 4 | 1280 | 典型 shape（基准） |
| TC-2 | 1 | 1 | 1 | 64 | 最小 Shape + R=1 |
| TC-3 | 1 | 512 | 8 | 256 | 中等 mhc (R=8) |
| TC-4 | 4 | 1 | 4 | 2048 | 大 h，小 batch |
| TC-5 | 1 | 1 | 4 | 1280 | 单 batch (A1=1) |
| TC-6 | 2 | 1024 | 4 | 1300 | 非对齐尾块 (h%tileA0Len != 0) |
| TC-7 | 1 | 1 | 1 | 1 | 极小值 (h=1) |

### 8.2 验收标准

**精度**:
| 指标 | 阈值 |
|------|------|
| MERE (平均相对误差) | <= 0.0078125 (2^-7) |
| MARE (最大相对误差) | <= 0.078125 (10 * 2^-7) |

**功能**:
- 全部 7 用例通过（输出 Shape 正确，无 NaN/INF，MERE/MARE 达标）

**编译**:
- `cmake .. && make -j4` 通过，零警告，`--npu-arch=dav-2201`

---

## 九、设计历史

| 版本 | 日期 | 主要变更 |
|------|------|---------|
| v1.0 | 2026-06 | 初版：per-row Muls+Add，TQue<1> |
| v2.0 | 2026-06 | 升级 ARA-FullLoad + ReduceSum RA + TQue<2>；发现 PopStackBuffer 性能退化 |
| v3.0 | 2026-07 | 回退 per-row Muls+Add + TQue<2>；修复 M2 blockNum 钳制 |
| v4.0 | 2026-07 | 统一文档表述，清理 v2.0 残余引用 |
| **v5.0** | **2026-07** | **本版: 完整重新设计，整合所有历史经验，归档为正式设计文档** |
