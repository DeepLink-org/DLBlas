# MHC Pre-processing Fused Kernel (big_fuse) — 架构设计

> **NpuArch**: DAV_2201 | **芯片**: Ascend910B2 | **CANN**: 9.0.0 | **__NPU_ARCH__**: 2201
> **CubeCore**: 24 | **VectorCore**: 48 | **UB**: 192 KB

---

## 0. 方案决策

### 0.1 算子类型判定

该算子是 MHC (Multi-Head Composition) 预处理融合算子，将四个独立计算阶段融合为单算子流水线：

| 阶段 | 计算类型 | 核心操作 |
|------|---------|---------|
| S1 | **MatMul + Reduction** | `[512, 5120] @ [5120, 24]` 矩阵乘 + RMS 归一化 |
| S2 | **Elementwise + Broadcast** | Scale/Bias 广播 + Split + Sigmoid |
| S3 | **Iterative Reduction** | Sinkhorn 双重随机归一化 (10 次迭代) |
| S4 | **Reduction + Elementwise** | 加权求和沿 mhc_mult 维度 (4 元素) |

**算子类型**: MatMul + Vector Fusion（复杂融合管线）

### 0.2 路线决策

| 决策因子 | 结论 |
|---------|------|
| 目标架构 | DAV_2201 (Ascend910B2)，**非** DAV_3510 |
| 主计算形态 | MatMul (Cube) + 多步 Vector 后处理 |
| 路线选择 | **SIMD/MemBase 通用路线** |
| MatMul 子路线 | Ascend C **MatmulImpl 高阶 API** (DAV_2201 标准) |
| RegBase | **不适用** — RegBase 要求 DAV_3510 |
| Blaze/tensor_api | **不适用** — Blaze 要求 DAV_3510 |

> **决策理由**：
> 1. DAV_2201 架构不满足 RegBase 和 Blaze 的前置条件（均需 DAV_3510）
> 2. DAV_2201 上 MatMul 标准路径为 `MatmulImpl` + `MatmulApiTiling` 高阶 API（位于 `adv_api/matmul/`）
> 3. AIC (Cube) 和 AIV (Vector) 职责天然分离：Cube 计算由 CubeCore 执行，Vector 计算由 VectorCore 执行
> 4. 中间张量体积极小（10 MB fp32 residual_flat，48 KB fp32 raw_mixes），GM 搬运开销可忽略

### 0.3 核函数拆分策略

| 方案 | 核函数数 | 优点 | 缺点 |
|------|---------|------|------|
| **A. 三核流水线 (采用)** | 3 | 每核专职、无 C9 违规、AIC 只管 Cube/AIV 只管 Vector、调试简单 | 2 次额外 kernel launch 开销 (~20us) |
| B. 两核 + Host 预处理 | 2 | 少两次 launch | **违反 C9 约束**（禁止 Host 侧对输入 tensor 做预处理） |
| C. 单核融合 | 1 | 零 launch 开销 | DAV_2201 不支持 AIC/AIV 单核混用 Cube+Vector |

**采用方案 A**：三核流水线。理由：
1. C9 禁止 Host 侧对算子输入 tensor 做预处理（含 dtype 转换、reshape 等），K0 在 device 侧 AIV 上完成 bf16→fp32 转换
2. AIC 核专做 MatMul（`MatmulImpl`），AIV 核专做 Vector 计算，职责清晰
3. 额外 kernel launch 开销 (~20us) 相对总延迟 (~1662us) 约 1.2%，可忽略

---

## 1. 数学定义

### 1.1 输入规格

| 张量 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `residual` | `[1, 512, 4, 1280]` | bf16 | 残差输入 |
| `fn` (mhc_fn) | `[24, 5120]` | fp32 | 投影权重矩阵 |
| `mhc_scale` | `[3]` | fp32 | 缩放系数 |
| `mhc_base` | `[24]` | fp32 | 偏置向量 |

### 1.2 输出规格

| 张量 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `post_mix` | `[1, 512, 4, 1]` | fp32 | 后混合系数 |
| `comb_mix` | `[1, 512, 4, 4]` | fp32 | 组合混合矩阵 (Sinkhorn 归一化) |
| `layer_input` | `[1, 512, 1280]` | bf16 | 加权混合层输入 |

### 1.3 常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `mhc_mult` | 4 | 多头组合数 |
| `hidden_size` | 1280 | 隐藏维度 |
| `rgs = mhc_mult * hidden_size` | 5120 | 展开维度 |
| `mhc_mult3 = 2*mhc_mult + mhc_mult^2` | 24 | 混合输出维度 |
| `n_tokens = batch * seq_len` | 512 | token 数 |
| `rms_eps` | 1e-6 | RMS eps |
| `mhc_pre_eps` | 1e-6 | pre_mix 微调 |
| `mhc_sinkhorn_eps` | 1e-6 | Sinkhorn eps |
| `sinkhorn_repeat` | 10 | 迭代次数 |
| `mhc_post_mult_value` | 1.0 | post_mix 因子 |

### 1.4 公式

**S1 — RMS-normalized 线性投影**:
```
x = flatten(residual)                           bf16→fp32, [1,512,4,1280] → [512,5120]
raw_mixes = x @ fn^T                            fp32, [512,5120] @ [5120,24] = [512,24]
sqrsum = sum(x^2, dim=-1)                       [512,1]
mixes = raw_mixes * rsqrt(sqrsum/rgs + eps)     [512,24]
```

**S2 — Split 混合 logits**:
```
scale = [s0*4, s1*4, s2*16]                     expand [3] → [24]
input_mixes = mixes * scale + base              [512,24]
pre_mix  = sigmoid(input[:,:4]) + eps           [512,4,1]
post_mix = sigmoid(input[:,4:8]) * mult         [512,4,1]
comb_mix = input[:,8:].reshape(512,4,4)         [512,4,4]
```

**S3 — Sinkhorn 双重随机归一化**:
```
x = softmax(comb, dim=-1) + eps
x = x / (sum(x, dim=-2) + eps)
repeat 9 times:
    x = x / (sum(x, dim=-1) + eps)              row normalize
    x = x / (sum(x, dim=-2) + eps)              col normalize
```

**S4 — 加权混合**:
```
layer_input = sum(residual_fp32 * pre_mix, dim=-2).to(bf16)   [512,1280]
```

---

## 2. 整体架构

### 2.1 三核流水线

```
GM (HBM)
  ├── residual [1,512,4,1280] bf16
  ├── fn [24,5120] fp32
  │
  ▼ K0 (AIV, 全核)
┌──────────────────────────────────────────┐
│  bf16→fp32 Conversion + Flatten          │
│  · residual bf16 → residual_flat fp32    │
│  · [1,512,4,1280] → [512,5120]          │
│  · 输出: residual_flat fp32 → GM         │
└──────────────────────────────────────────┘
  │  residual_flat [512,5120] fp32 (10 MB)
  ▼ K1 (AIC, 多核)
┌──────────────────────────────────────────┐
│  MatMul (Cube)                            │
│  · MatmulImpl: [512,5120] @ [5120,24]    │
│  · A=fp32, B=fp32, C=fp32                │
│  · 输出: raw_mixes [512,24] fp32 → GM    │
└──────────────────────────────────────────┘
  │  raw_mixes [512,24] fp32 (48 KB)
  ▼ K2 (AIV, 安全核数)
┌──────────────────────────────────────────┐
│  Vector Post-process                      │
│  · S1: RMS Norm (使用 K0 的 residual_flat)│
│  · S2: Split + Sigmoid                    │
│  · S3: Sinkhorn 10 迭代                   │
│  · S4: Weighted Sum → layer_input         │
│  · 输出: post_mix, comb_mix, layer_input  │
└──────────────────────────────────────────┘
```

### 2.2 中间张量

| 张量 | Shape | dtype | 大小 | 生产者 | 消费者 |
|------|-------|-------|------|--------|--------|
| `residual_flat` | [512, 5120] | fp32 | 10 MB | K0 | K1, K2 |
| `raw_mixes` | [512, 24] | fp32 | 48 KB | K1 | K2 |

---

## 3. Kernel 0: bf16→fp32 转换 (AIV, Vector)

### 3.1 功能

将 bf16 残差输入展平并转换为 fp32，供后续 K1 (MatMul) 和 K2 (Weighted Apply) 使用。

### 3.2 多核切分

| 项目 | 值 |
|------|-----|
| 切分维度 | token 维 (512 tokens) |
| VectorCore | `PlatformAscendCManager::GetCoreNumAiv()` 动态获取 |
| tokensPerCore | ceil(512 / vecCoreNum) |
| 尾核 tokens | nTokens - (vecCoreNum-1) * tokensPerCore |

### 3.3 UB 切分

单 token 数据量: 4 x 1280 = 5120 elements = 10 KB (bf16) / 20 KB (fp32)。

| 参数 | 值 | 推导 |
|------|-----|------|
| tokensPerTile (T) | 4 | 4 tokens: bf16_in=40 KB, fp32_out=80 KB, 合计 ~123 KB < 192 KB |
| 每核 tile 数 | ceil(tokensPerCore / 4) | 尾 tile: tokensPerCore % 4 |
| 沿 hidden_size 不切分 | 1280 可一次处理 | DataCopyPad 支持大批量搬运 |

### 3.4 Buffer 规划

| Buffer | 类型 | 大小 (元素) | 字节 | 用途 |
|--------|------|-----------|------|------|
| `resBf16UB` | bf16 | T x 4 x 1280 = 20480 | 40,960 | bf16 残差输入 |
| `resFp32UB` | fp32 | T x 5120 = 20480 | 81,920 | fp32 展平输出 |
| **合计** | | | **~123 KB** | < 192 KB |

### 3.5 数据流

```
for each tile (T tokens):
  1. DataCopyPad: residual[T, 4, 1280] bf16 GM → resBf16UB
  2. Cast: resBf16UB (bf16) → resFp32UB (fp32), 同时展平 [T,4,1280] → [T,5120]
  3. DataCopy: resFp32UB[T, 5120] fp32 UB → residual_flat GM (32B 对齐)
```

> **注意**: `DataCopy(GlobalTensor, LocalTensor, uint32_t count)` 的 count 参数是**元素数**，非字节数。

### 3.6 API 映射 (已验证)

| 功能 | API | 参数/约束 | 验证来源 |
|------|-----|----------|---------|
| GM→UB 搬运 | `DataCopyPad` | bf16 非对齐安全搬运 | api-datacopy.md |
| 类型转换 | `Cast<float, bf16>` | CAST_NONE，bf16→fp32 无损 | api-precision.md |
| UB→GM 搬运 | `DataCopy` | fp32 32B 天然对齐 | api-datacopy.md |
| AIC 守卫 | `ASCEND_IS_AIC → return` | AIV-only kernel | api-restrictions.md |

---

## 4. Kernel 1: MatMul (AIC, Cube)

### 4.1 多核切分

| 项目 | 值 |
|------|-----|
| 切分方式 | M x N 二维切分 |
| CubeCore | `PlatformAscendCManager::GetCoreNumAic()` 动态获取 |
| singleCoreM/N | `MatmulApiTiling::GetTiling` 自动推导 → Host 端扩展 |

### 4.2 MatMul 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| M | 512 | n_tokens |
| K | 5120 | rgs |
| N | 24 | mhc_mult3 |
| transA | false | residual_flat [M, K] |
| transB | true | fn 存储为 [N, K] = [24, 5120] |
| A dtype | fp32 | K0 产出 |
| B dtype | fp32 | fn 权重 (原始精度) |
| C dtype | fp32 | raw_mixes 输出 |
| bias | 无 | — |
| 精度模式 | `MatrixMadType::NORMAL` | 全 fp32 |

### 4.3 Tiling (MatmulApiTiling)

```
1. MatmulApiTiling::SetBufferSpace(L1=512KB, L0A=64KB, L0B=64KB, UB=192KB)
2. MatmulApiTiling::SetShape(M=512, N=24, K=5120)
3. MatmulApiTiling::SetAType(GM, ND, DT_FLOAT, trans=false)
4. MatmulApiTiling::SetBType(GM, ND, DT_FLOAT, trans=true)
5. MatmulApiTiling::SetCType(GM, ND, DT_FLOAT)
6. MatmulApiTiling::SetBias(false)
7. MatmulApiTiling::GetTiling(cubeTiling)
8. 覆写 M/N/Ka/Kb → 多核切分扩展 (优先缩小 singleCoreM)
9. 小算子: M*N*dtype = 48KB << L2/2 → 跳过 L2 切分
```

**对齐**: baseM/baseN/singleCoreM/singleCoreN 均为 ALIGNED_H=16 整数倍。

**多核扩展逻辑**: 若 `totalBlock < aicCoreNum` 且 `singleCoreM > minM`，将 `singleCoreM` 减半（保持 16 对齐），重新计算 `mCnt`、`totalBlock`，直到 `totalBlock >= aicCoreNum` 或 `singleCoreM <= minM`。

### 4.4 数据流

```
Host:
  1. PlatformAscendCManager → aicCoreNum, l1Size, l0Size, ubSize (动态获取)
  2. MatmulApiTiling::GetTiling(cubeTiling)
  3. 多核切分扩展 (singleCoreM 减半)
  4. blockDim = min(usedCoreNum, totalBlock)

Device (AIC):
  for each (mIdx, nIdx) in serpentine:
    curM = tail ? mBaseTail : singleCoreM
    curN = tail ? nBaseTail : singleCoreN
    mm_.SetSingleShape(curM, curN, K)
    mm_.SetTensorA(residualFlatGm[mOff], false)
    mm_.SetTensorB(fnGm[nOff], true)
    mm_.IterateAll(rawMixesGm[off])
  PipeBarrier<PIPE_ALL>()
  SetAtomicNone()
```

### 4.5 API 映射 (已验证)

| 功能 | API | 参数/约束 | 验证来源 |
|------|-----|----------|---------|
| Cube 引擎 | `matmul::MatmulImpl<A,B,C,Bias,MM_CFG>` | A/B=fp32, C=fp32, MM_CFG 含 enUnitFlag | matmul_impl.h |
| 类型定义 | `MatmulType<GM, ND, float, trans>` | TPosition, CubeFormat, DataType | matmul.h |
| MM_CFG | `GetMDLConfig(...)` | enUnitFlag=true (IterateAll 必需) | matmul_config.h |
| Host Tiling | `matmul_tiling::MatmulApiTiling` | GetTiling + 手动扩展 | matmul_tiling.h |
| 精度模式 | `SetMadType(NORMAL)` | 全 fp32 | matmul_tiling.h |
| 结果写回 | `IterateAll(cGm[...])` | 含 enUnitFlag | matmul_impl.h |
| 守卫 | `ASCEND_IS_AIV → return` | AIC 专用 | api-restrictions.md |
| Kernel 类型 | `__global__ __cube__` | Cube 专用 kernel (避免编译器警告) | — |

---

## 5. Kernel 2: Vector Post-process (AIV, Vector)

### 5.1 多核切分

| 项目 | 值 |
|------|-----|
| 切分维度 | token 维 (512) |
| VectorCore | `PlatformAscendCManager::GetCoreNumAiv()` 动态获取 |
| tokensPerCore | ceil(512/vecCoreNum) |

**安全核数计算 (Singleton Tile 防护)**:

为确保每个 core 的 tokensPerCore 为偶数（防止尾 tile curT=1 的 singleton tile），实际使用的 k2CoreNum 可能小于 vecCoreNum：

```
k2CoreNum = vecCoreNum
while k2CoreNum > 1:
    tpc = ceil(512 / k2CoreNum)
    if lastStart < 512 AND tpc % 2 == 0: break
    k2CoreNum--
```

以 vecCoreNum=48 为例: tpc=11 (奇数) → 缩减至 43，tpc=12 (偶数)。尾核 core 42 起始于 token 504，处理 8 个 tokens。

### 5.2 UB 切分

| 参数 | 值 | 推导 |
|------|-----|------|
| tokensPerTile (T) | **2** | 2 tokens: UB ~108 KB < 192 KB, 配合偶数 tpc 消除 singleton tile |
| 每核 tile 数 | ceil(tokensPerCore / 2) | 尾 tile curT 恒为 2 (偶数保证) |
| Singleton tile | **已消除** | 偶数 tokensPerCore 确保每 core 尾 tile curT >= 2 |

> **Singleton Tile Bug 记录**: T=2 + 奇数 tokensPerCore 导致尾 tile curT=1，DataCopyPad 将 16B bf16 数据 padding 至 32B 对齐时溢出至相邻 token 区域，导致该 tile 输出全零。通过缩减 k2CoreNum 确保 tokensPerCore 为偶数来修复。

### 5.3 Buffer 规划 (T=2)

以 T=2 为例：

| Buffer | 类型 | 元素数 | 字节 | 用途 |
|--------|------|--------|------|------|
| `residualBf16Que` | bf16 | 2x4x1280=10240 | 20,480 | 残差 bf16 输入 (Phase D 用) |
| `residualFp32Que` | fp32 | 2x5120=10240 | 40,960 | 残差 fp32 (Phase A sqrsum + Phase D) |
| `tmpCalcQue` | fp32 | 2x5120=10240 | 40,960 | 临时计算 (sqr, mixes, weighted) |
| `splitBuf` | fp32 | 2x24=48 | 192 | pre/post/comb split 缓冲 |
| `layerOutQue` | bf16 | 2x1280=2560 | 5,120 | layer_input bf16 输出 |
| **合计** | | | **~107.7 KB** | < 192 KB |

> **Buffer 复用**:
> - `residualFp32Que` 和 `tmpCalcQue` 分阶段分配/释放，不同时占用
> - `splitBuf` 先后分配给 sqrsum (T x 1 fp32) 和 splBuf (T x 24 fp32)
> - 所有 TQue 使用 BUFFER_NUM=1（单缓冲），UB=192KB 无法容纳双缓冲
> - `PipeBarrier<PIPE_ALL>` 保证所有 scalar↔vector 转换点的数据一致性

### 5.4 数据流

```
对于每个 tile (T tokens):

Phase A — RMS Norm:
  A1. DataCopyPad: residual_flat [T, 5120] fp32 GM → residualFp32Que
  A2. Square + scalar accumulate: sqrsum = sum(x^2, dim=-1) → splitBuf [T]
  A3. DataCopyPad: raw_mixes [T, 24] fp32 GM → tmpCalcQue
  A4. Rsqrt(sqrsum/rgs + eps) → splitBuf
  A5. Broadcast Mul: tmpCalcQue *= splitBuf (Mul + BinaryRepeatParams{src1RepStride=0})
  A6. PipeBarrier<PIPE_ALL>  ← scalar→vector 一致性保护

Phase B — Split + Sigmoid:
  B1. Broadcast Mul (Muls): tmpCalcQue *= scaleVec[24]; Broadcast Add: tmpCalcQue += baseVec[24]
      Split into pre(T*4), post(T*4), comb(T*16) in splitBuf
  B2. pre = AscendC::Sigmoid<float>(pre) + eps
      post = AscendC::Sigmoid<float>(post) * postMultValue
  B3. PipeBarrier<PIPE_ALL>  ← vector→scalar 一致性保护
  B4. DataCopyPad: post_mix → GM (Phase D 之后执行，避免 buffer 竞争)

Phase C — Sinkhorn (10 迭代):
  C1. Softmax dim=-1: max-sub + AscendC::Exp + scalar ReduceSum + Div + eps
  C2. Col norm: comb /= (sum(dim=-2) + eps)
  C3. Loop 9 times: alternating row/col norm (scalar, M4=4 小于向量 ReduceSum 最小尺寸)
  C4. PipeBarrier<PIPE_ALL>
  C5. DataCopyPad: comb_mix → GM

Phase D — Weighted Apply:
  D1. DataCopyPad: residual_bf16 [T, 4, 1280] GM → residualBf16Que
  D2. Cast<bf16, float>: residualBf16Que (bf16) → residualFp32Que (fp32), CAST_NONE
  D3. Weighted Multiply: Muls per row with pre_mix values (Muls + BinaryRepeatParams{HS=1280})
  D4. ReduceSum dim=-2: 4 rows → 1 row per token (3x Add per token, HS=1280)
  D5. Cast<bf16, float>: residualFp32Que (fp32) → layerOutQue (bf16), CAST_ROUND
  D6. DataCopyPad: layerOutQue → layer_input GM
```

### 5.5 Sinkhorn 实现细节

4x4 矩阵极小，展开为 batched scalar 操作：

```
// Iteration 0: softmax + col norm
SoftmaxLastDim(comb[T,4,4]) + eps
ColNormalize(comb[T,4,4])

// Iterations 1..9: alternating row/col norm
for i in 1..9:
    RowNormalize(comb[T,4,4])   // sum over dim=-1, broadcast divide
    ColNormalize(comb[T,4,4])   // sum over dim=-2, broadcast divide
```

每轮仅需 2 次 scalar ReduceSum + 2 次 Broadcast Div，在 4x4 上开销极小。M4=4 小于 DAV_2201 Vector ReduceSum 最小有效尺寸（~64 elements），scalar 实现反而是最优选择。

### 5.6 API 映射 (已验证)

| 步骤 | 功能 | API | 约束 |
|------|------|-----|------|
| A1,A3,D1 | GM→UB 搬运 | `DataCopyPad` | bf16/fp32 非对齐安全 |
| A2 | 平方+标量累加 | `Mul` + scalar accumulate | DAV_2201 vcadd 限制 ~64 fp32/token, RGS=5120 需 scalar 兜底 |
| A4 | 倒数平方根 | `Rsqrt` + `Muls` | fp32 精度 |
| A5,B1 | 广播乘/加 | `Mul`/`Muls`/`Adds` | BinaryRepeatParams: src1RepStride=0 |
| B2 | Sigmoid | `AscendC::Sigmoid<float>` | 硬件加速，clamp x to [-88,88] |
| B3,C4 | 管线同步 | `PipeBarrier<PIPE_ALL>` | scalar↔vector 数据一致性 |
| B4,C5,D6 | UB→GM | `DataCopyPad` | bf16/fp32 安全搬运 |
| C1-3 | Sinkhorn | scalar `GetValue`/`SetValue` | M4=4，矢量化收益为负 |
| D2 | bf16→fp32 | `Cast<float, bf16>` | CAST_NONE (无损) |
| D3 | 加权乘 | `Muls` + BinaryRepeatParams | HS=1280 矢量 |
| D4 | 归约 dim=-2 | `Add` 3次/token | 4元素归约，加法链 |
| D5 | fp32→bf16 | `Cast<bf16, float>` | CAST_ROUND |
| — | AIC 守卫 | `ASCEND_IS_AIC → return` | AIV-only kernel |

---

## 6. Host 侧 Tiling 设计

### 6.1 Tiling 数据结构

```cpp
#pragma pack(push, 8)

struct TilingHeaderK0 {
    int32_t nTokens;           // 512
    int32_t mhcMult;           // 4
    int32_t hiddenSize;        // 1280
    int32_t rgs;               // 5120
    int32_t tokensPerCore;     // ceil(512/vecCoreNum)
    int32_t tokensPerTile;     // 4
    int32_t vecCoreNum;        // PlatformAscendCManager::GetCoreNumAiv()
    int32_t reserved[1];
};

struct TilingHeaderK1 {
    AscendC::tiling::TCubeTiling cubeTiling;  // MatmulApiTiling::GetTiling 产出
    int32_t mTotalCnt;          // M 方向 tile 数
    int32_t nTotalCnt;          // N 方向 tile 数
    int32_t totalBlock;         // 总 block 数 = mCnt * nCnt
    int32_t mBaseTail;          // 尾 tile M
    int32_t nBaseTail;          // 尾 tile N
    int32_t convTileK;          // 保留
    int32_t reserved[3];
};

struct TilingHeaderK2 {
    int32_t nTokens;              // 512
    int32_t mhcMult;              // 4
    int32_t hiddenSize;           // 1280
    int32_t mhcMult3;             // 24
    int32_t rgs;                  // 5120
    int32_t tokensPerCore;        // ceil(512/k2CoreNum), 恒为偶数
    int32_t tokensPerTile;        // 2
    int32_t vecCoreNum;           // k2CoreNum (缩减后)
    int32_t sinkhornRepeat;       // 10
    float   rmsEps;               // 1e-6
    float   mhcPreEps;            // 1e-6
    float   mhcSinkhornEps;       // 1e-6
    float   mhcPostMultValue;     // 1.0
    float   scaleVec[24];         // expanded [3] → [24]
    float   baseVec[24];
    int32_t reserved[4];
};

#pragma pack(pop)
```

### 6.2 Host 端执行流程

```
1. 校验输入: dtype, shape, K 维度匹配
2. PlatformAscendCManager → aicCoreNum/vecCoreNum (动态获取)
3. 分配 GM 内存:
   - resFp32Dev: [512, 5120] fp32 (10 MB, K0 输出)
   - rawDev: [512, 24] fp32 (48 KB, K1 输出)
   - postDev/cmbDev/layDev: 输出 buffer
4. ComputeK0Tiling → K0 启动 (blockDim=vecCoreNum, AIV)
5. aclrtSynchronizeStream
6. ComputeK1Tiling → MatmulApiTiling::GetTiling → 多核扩展 → K1 启动 (blockDim=cubeTiling.usedCoreNum, AIC)
7. aclrtSynchronizeStream
8. ComputeK2Tiling (含 scale/base 展开 + k2CoreNum 安全计算) → K2 启动 (blockDim=k2CoreNum, AIV)
9. aclrtSynchronizeStream
10. 拷贝输出回 Host → 释放 GM
```

---

## 7. 精度策略

### 7.1 精度标准

| 项目 | 值 |
|------|-----|
| 标准类型 | **浮点计算社区标准** (float_compute_community) |
| fp32 输出 (post_mix, comb_mix) | MERE < 2^-13 (~1.22e-4) |
| bf16 输出 (layer_input) | Max Abs Error < 2^-6 (~0.0156, 2 ULP bf16) |
| 标杆构造 | PyTorch CPU fp32 参考实现 |

### 7.2 混合精度路径

| 阶段 | 输入 dtype | 计算 dtype | 输出 dtype | 策略 |
|------|-----------|-----------|-----------|------|
| K0 转换 | bf16 | — | fp32 | bf16→fp32 无损 (CAST_NONE) |
| K1 MatMul | fp32 | fp32 | fp32 | 全 fp32 Cube MMAD (MatrixMadType::NORMAL) |
| K2 RMS Norm | fp32 | fp32 | fp32 | 全 fp32 |
| K2 Split+Sigmoid | fp32 | fp32 | fp32 | 全 fp32 |
| K2 Sinkhorn | fp32 | fp32 | fp32 | 全 fp32 (10迭代无精度放大) |
| K2 Weighted | bf16→fp32 | fp32 | bf16 | 累加 fp32，最终 CAST_ROUND |

### 7.3 数值稳定性保护

| 风险 | 保护措施 |
|------|---------|
| RMS norm 分母为零 | eps=1e-6 |
| Sigmoid exp 溢出 | clamp x to [-88, 88] (fp32 exp 安全域) |
| Sinkhorn 除零 | 每次除法 + sinkhorn_eps=1e-6 |
| Softmax 稳定性 | max 减法保护 (x - max(x)) |
| Scalar↔Vector coherency | PipeBarrier\<PIPE_ALL\> 在所有转换点 |

---

## 8. 关键 Bug 修复记录

### 8.1 Singleton Tile 零输出 (已修复)

| 项目 | 详情 |
|------|------|
| **Bug ID** | K2-SINGLETON-ZERO |
| **现象** | 部分 token 的 post_mix/comb_mix/layer_input 全部为零 |
| **根因** | tokensPerCore=11 (奇数) + tokensPerTile=2 → 尾 tile curT=1; DataCopyPad 将 16B bf16 数据 padding 至 32B 时溢出到相邻 token |
| **修复** | k2CoreNum 缩减至 43 (从 48)，确保 tokensPerCore=12 (偶数)，所有 tile curT=2 |
| **状态** | 已修复并验证 |

### 8.2 Scalar/Vector Coherency (已修复)

| 项目 | 详情 |
|------|------|
| **Bug ID** | K2-SCALAR-VECTOR-COHERENCY |
| **现象** | PIPE_V 不足以保证 scalar GetValue/SetValue 对 vector ops 的数据可见性 |
| **根因** | DAV_2201 AIV 核心上 scalar 和 vector 使用独立执行管线 |
| **修复** | 所有 scalar↔vector 转换点使用 PipeBarrier\<PIPE_ALL\> |
| **状态** | 已修复并验证 |

---

## 9. 设计检查清单

### K0 (bf16→fp32 转换)

- [x] token 维切分: 512 tokens / vecCoreNum cores
- [x] T=4, UB ~123 KB < 192 KB
- [x] DataCopyPad 用于 bf16 非对齐搬运
- [x] Cast bf16→fp32 使用 CAST_NONE (无损)
- [x] DataCopy fp32 写回 32B 天然对齐
- [x] 尾 tile 处理 (curT < 4)
- [x] AIC 守卫: `ASCEND_IS_AIC → return`

### K1 (MatMul)

- [x] M=512, K=5120, N=24, transB=true
- [x] A=fp32, B=fp32, C=fp32 (MatrixMadType::NORMAL)
- [x] MatmulApiTiling::GetTiling + 手动多核扩展
- [x] ALIGNED_H=16 对齐约束
- [x] small matrix skip L2 tiling (48 KB << L2/2)
- [x] PipeBarrier + SetAtomicNone
- [x] AIV 守卫: `ASCEND_IS_AIV → return`
- [x] Kernel 类型: `__global__ __cube__`

### K2 (Vector Post-process)

- [x] token 维切分 + k2CoreNum 安全缩减
- [x] T=2, tokensPerCore 偶数, UB ~108 KB < 192 KB
- [x] bf16→fp32 升精度 (Phase D)
- [x] Sigmoid clamp 防 exp 溢出
- [x] Sinkhorn +eps 除零保护
- [x] PipeBarrier\<PIPE_ALL\> scalar↔vector 一致性
- [x] AIC 守卫: `ASCEND_IS_AIC → return`
- [x] DataCopyPad 用于所有非对齐搬运

### Host

- [x] PlatformAscendCManager 动态获取核数
- [x] K1 MatmulApiTiling 调用
- [x] K2 k2CoreNum 安全计算 (偶数 tokensPerCore)
- [x] Host 不预处理输入 tensor (C9 合规)
- [x] aclrtSynchronizeStream 核间同步

---

## 10. 参考资源

| 资源 | 用途 |
|------|------|
| `adv_api/matmul/matmul.h` | MatmulImpl 公开 API |
| `adv_api/matmul/matmul_config.h` | MatmulConfig / CubeFormat / LayoutMode |
| `adv_api/matmul/matmul_tiling.h` | MatmulApiTiling Host Tiling |
| `impl/adv_api/detail/matmul/matmul_impl.h` | MatmulImpl 实现细节 |
| `ascendc-tiling-design/references/matmul/` | MatMul Tiling 策略 |
| `ascendc-tiling-design/references/reduction/patterns.md` | 归约模式 |
| `ascendc-api-best-practices/references/api-arithmetic.md` | 广播 Mul/Adds/Muls |
| `ascendc-api-best-practices/references/api-reduce.md` | ReduceSum/ReduceMax |
| `ascendc-api-best-practices/references/api-precision.md` | Cast bf16↔fp32 |
| `ascendc-api-best-practices/references/api-datacopy.md` | DataCopy/DataCopyPad |
| `ascendc-api-best-practices/references/api-pipeline.md` | PipeBarrier/EnQue/DeQue |
| `npu-arch/references/npu-hardware-params.md` | UB/L1/L0 容量 |
| `ops-precision-standard/reference/float_compute_community.md` | 浮点精度标准 |
