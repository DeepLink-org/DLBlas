# BigFuse 算子架构设计文档 (DESIGN.md)

---

## 1. 环境与架构信息

| 项目 | 值 |
|------|-----|
| 芯片型号 | Ascend910B2 |
| SocVersion | ASCEND910B |
| NpuArch | DAV_2201 |
| `__NPU_ARCH__` | 2201 |
| `--npu-arch` 编译参数 | dav2201_vec |
| CANN 版本 | 9.0.0 |
| UB 容量 | 192 KB (196608 bytes) |
| L1 容量 | 512 KB |
| VectorCore 数量 | 48 |

---

## 2. 路线决策

### 2.1 决策依据

| 判断维度 | 结论 |
|----------|------|
| 算子类型 | 多阶段向量融合算子（线性投影 + RMS Norm + Sigmoid + Sinkhorn + 加权求和） |
| 目标 NpuArch | DAV_2201（非 DAV_3510） |
| 是否 Cube/Matmul | 否（线性投影规模极小，[512, 5120] x [5120, 24]，远达不到 Cube 收益阈值） |

### 2.2 最终路线

**SIMD/MemBase 路线（通用 Ascend C Vector API）**

理由：
- DAV_2201 不支持 RegBase 和 Blaze 路线
- 算子核心是向量规约 + 逐元素计算，天然适合 Vector API
- UB 容量 192KB 足够支持 token-batch 处理

---

## 3. 数学定义与等价性分析

### 3.1 算子概述

BigFuse 是一个将 MHC（Multi-Head Composition）预处理四阶段融合为单一算子的实现。输入为四维残差张量 [B, S, M, H]，输出 post_mix [B, S, M, 1]、comb_mix [B, S, M, M]、layer_input [B, S, H]。

### 3.2 参数约定

| 符号 | 含义 | 典型值 |
|------|------|--------|
| `B` | batch_size (n0) | 1 |
| `S` | seq_len (n1) | 512 |
| `M` | mhc_mult | 4 |
| `H` | hidden_size | 1280 |
| `D` | 投影维度 = M * H | 5120 |
| `K` | mhc_mult3 = 2M + M^2 | 24 |
| `rms_eps` | RMS 归一化 epsilon | 1e-6 |
| `pre_eps` | pre_mix epsilon | 1e-6 |
| `sinkhorn_eps` | Sinkhorn epsilon | 1e-6 |
| `post_mult` | post_mix 乘数 | 1.0 |
| `sinkhorn_repeat` | Sinkhorn 迭代次数 | 10 |

### 3.3 Stage 1: RMS-Normalized Linear Projection

**输入**: `R` = residual `[B, S, M, H]` (bf16), `W` = fn weight `[K, D]` (fp32)

```
Step 1.1: Flatten & Cast
    X = R.reshape(B*S, D).to(fp32)          // [B*S, D]

Step 1.2: Linear Projection
    mixes = X @ W^T                           // [B*S, K]
    即: mixes[t, k] = sum_{d=0}^{D-1} X[t, d] * W[k, d]

Step 1.3: RMS Normalization
    sqrsum[t] = sum_{d=0}^{D-1} X[t, d]^2   // [B*S, 1]
    rms[t] = rsqrt(sqrsum[t] / D + rms_eps)  // [B*S, 1]
    mixes[t, k] = mixes[t, k] * rms[t]       // [B*S, K]
```

**等价性**: 数学恒等，仅将两阶段合并，无近似引入。

### 3.4 Stage 2: Split Mixes + Sigmoid

**输入**: `mixes` `[B, S, K]` (fp32), `scale` `[3]`, `base` `[K]`

```
Step 2.1: Broadcast scale
    scale_expanded = [scale[0] repeated M times,
                      scale[1] repeated M times,
                      scale[2] repeated M*M times]   // [K]
    biased = mixes * scale_expanded + base            // [B, S, K]

Step 2.2: Split & Sigmoid
    pre_mix[t, m] = sigmoid(biased[t, m]) + pre_eps              // [B, S, M, 1]
    post_mix[t, m] = sigmoid(biased[t, M+m]) * post_mult          // [B, S, M, 1]
    comb_mix[t, i, j] = biased[t, 2M + i*M + j]                   // [B, S, M, M]

其中 sigmoid(x) = 1 / (1 + exp(-x))
```

### 3.5 Stage 3: Sinkhorn Doubly-Stochastic Normalization

**输入**: `comb_mix` `[B, S, M, M]` (fp32)

```
Step 3.1: Initial Softmax
    C[t, i, j] = softmax_j(comb_mix[t, i, j]) + eps
    即: C[t, i, j] = exp(comb_mix[t, i, j] - max_j(comb_mix)) /
                     sum_{j'} exp(comb_mix[t, i, j'] - max_j(comb_mix)) + eps

Step 3.2: Iterative Row/Col Normalization (repeat R-1 times)
    For r in 1..R-1:
        C[t, i, j] = C[t, i, j] / (sum_{i'} C[t, i', j] + eps)    // col norm
        C[t, i, j] = C[t, i, j] / (sum_{j'} C[t, i, j'] + eps)    // row norm
```

其中 softmax 沿最后一维（j 轴），M=4 时矩阵为 4x4，每 token 独立计算。

### 3.6 Stage 4: Weighted Sum (Apply Mix)

**输入**: `residual` `[B, S, M, H]` (bf16), `pre_mix` `[B, S, M, 1]` (fp32)

```
    R_cast = residual.to(fp32)                                   // [B, S, M, H]
    weighted = R_cast * pre_mix                                   // [B, S, M, H]
    layer_input[t, h] = sum_{m=0}^{M-1} weighted[t, m, h]       // [B, S, H]
    output = layer_input.to(bf16)                                 // [B, S, H]
```

### 3.7 整体等价性

全部计算为精确数学变换的组合：
- RMS Norm: 精确代数变换，无近似
- Sigmoid: exp 计算引入浮点舍入，属正常浮点运算
- Sinkhorn: exp + 除法迭代，属正常浮点运算
- 加权求和: 精确代数

**结论**: 融合不改变数学语义，输出与 PyTorch 逐阶段计算在浮点误差范围内一致。

---

## 4. API 映射表

### 4.1 已验证的 API

| 数学操作 | Ascend C API | 级别 | 签名验证 |
|---------|-------------|------|---------|
| bf16 -> fp32 类型转换 | `Cast<float, bfloat16_t>` | Level 2 | `Cast(dst, src, count)` |
| fp32 -> bf16 类型转换 | `Cast<bfloat16_t, float>` | Level 2 | `Cast(dst, src, count)` |
| 逐元素乘法 | `Mul` | Level 2 | `Mul(dst, src0, src1, count)` |
| 逐元素加法 | `Add` | Level 2 | `Add(dst, src0, src1, count)` |
| 逐元素除法 | `Div` | Level 2 | `Div(dst, src0, src1, count)` |
| 向量乘标量 | `Muls` | Level 2 | `Muls(dst, src, scalar, count)` |
| 向量加标量 | `Adds` | Level 2 | `Adds(dst, src, scalar, count)` |
| 向量除标量 | `Divs` | Level 2 | `Divs(dst, src, scalar, count)` |
| 指数函数 | `Exp` | Level 2 | `Exp(dst, src, count)` |
| 平方根倒数 | `Rsqrt` | Level 2 | `Rsqrt(dst, src, count)` |
| 取反 | `Neg` | Level 2 | `Neg(dst, src, count)` |
| 倒数 | `Reciprocal` | Level 2 | `Reciprocal(dst, src, count)` |
| 块规约求和 | `BlockReduceSum` | Level 0 | 含 repeatTime, mask, stride 参数 |
| 数据搬运（含对齐处理） | `DataCopyPad` | - | Ext 版本，支持非对齐 |
| 数据复制 | `Duplicate` | Level 2 | `Duplicate(dst, src, count)` |

### 4.2 复合操作映射

| 复合操作 | 实现方式 |
|---------|---------|
| `x.square()` | `Mul(x, x)` |
| `sigmoid(x)` | `Neg` -> `Exp` -> `Adds(1.0)` -> `Reciprocal` |
| `softmax(x, dim=-1)` | `Adds(-max)` -> `Exp` -> `BlockReduceSum` -> `Duplicate`(广播) -> `Div` |
| 向量点积 (dot product) | `Mul` -> `BlockReduceSum` |

### 4.3 未使用的 API

| API | 原因 |
|-----|------|
| Sigmoid (内置) | 不存在，需手动组合 |
| Square (内置) | 不存在，用 Mul 代替 |
| Softmax (内置) | 不存在，需手动组合 |
| Cube 相关 API (`MatmulImpl` 等) | 投影矩阵维度太小，Vector 路径更高效 |
| RegBase API | DAV_2201 不支持 |

---

## 5. 架构总览

### 5.1 3-Kernel 流水线架构

实际实现采用**三 Kernel 流水线**架构，将计算按硬件特性分配到不同核心类型（AIC=Cube 核心, AIV=Vector 核心）。

```
+-----------------------------------------------------------------------------+
|                    BigFuse 3-Kernel Pipeline                                |
|                                                                             |
|  +-------------------------+    +---------------------+    +---------------+
|  | K0: bf16->fp32 + Flatten|--->| K1: MatMul (Cube)   |--->| K2: Vector    |
|  | (AIV, 48 cores)        |    | (AIC, 8 cores)       |    | Post-process  |
|  |                         |    |                      |    | (AIV, 43 cores)|
|  | residual [1,512,4,1280] |    | A=[512,5120] fp32    |    | RMS Norm      |
|  |   bf16                  |    | B=[24,5120] fp32     |    | Split/Sigmoid |
|  |       |                 |    |       |              |    | Sinkhorn      |
|  |       v                 |    |       v              |    | Apply Mix     |
|  | residual_flat [512,5120]|    | raw_mixes [512,24]   |    |               |
|  |   fp32 -> GM            |    |   fp32 -> GM         |    |               |
|  +-------------------------+    +---------------------+    +---------------+
```

#### K0: 数据格式转换 + 压平 (AIV, 48 cores)

- **输入**: `residual` [1, 512, 4, 1280] bf16 (GM)
- **输出**: `residual_flat` [512, 5120] fp32 (GM)
- **操作**: DataCopyPad 搬运 bf16 数据 -> Cast(bf16->fp32) -> DataCopy 写回 GM
- **Tiling**: T=4 tokens/tile, 单缓冲（计算简单，无需双缓冲）
- **内核属性**: `__global__ __vector__`, AIC 核心直接返回

#### K1: 线性投影 (AIC, 8 cores, Cube MatMul)

- **输入**: `residual_flat` [512, 5120] fp32 (GM), `fn` [24, 5120] fp32 (GM)
- **输出**: `raw_mixes` [512, 24] fp32 (GM)
- **MatMul API**: `MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG>` (DAV_2201 高阶 API)
- **MatMulConfig**: `GetMDLConfig(false, false, 0, false, false, false, true)` — enUnitFlag 开启保证 IterateAll 写回
- **2D 块切分**: M 维 `singleCoreM=64` (ceil(512/64)=8 blocks), N 维 `singleCoreN=24` (1 block, N 太小无法再分)
- **内核属性**: `__global__ __cube__`, AIV 核心直接返回
- **实测延迟**: ~55-60us

#### K2: 向量后处理 (AIV, 43 cores)

- **输入**: `residual_flat` [512, 5120] fp32, `raw_mixes` [512, 24] fp32, `residual` [1, 512, 4, 1280] bf16 (全部 GM)
- **输出**: `post_mix` [512, 4, 1] fp32, `comb_mix` [512, 4, 4] fp32, `layer_input` [512, 1280] bf16 (全部 GM)
- **操作流程**: RMS Norm -> Scale/Bias -> Split/Sigmoid -> Sinkhorn -> Apply Mix（全部在一个 kernel 内完成）
- **Tiling**: T=3 tokens/tile, 5 个单缓冲 Que（qBf16, qFp32, qCalc, qSpl, qOut）
- **M=4 特殊处理**: Sinkhorn 的 4x4 矩阵太小，逐 token 标量循环处理（Vector ReduceSum 收益不足）
- **内核属性**: `__global__ __vector__`, AIC 核心直接返回

### 5.2 设计理由：3-Kernel vs 单 Kernel

| 维度 | 原设计 (单 Kernel 双阶段) | 实际实现 (3-Kernel 流水线) |
|------|--------------------------|---------------------------|
| 线性投影实现 | Vector Mul + BlockReduceSum | Cube MatMul (K1, `MatmulImpl`) |
| bf16->fp32 转换 | Kernel 内 inline Cast | 独立 K0 Kernel |
| 后处理 (RMS/Sigmoid/Sinkhorn/Apply) | Kernel 内 Phase 2 | 独立 K2 Kernel |
| 中间数据 | UB 驻留 (不写 GM) | GM 传递 (residual_flat 10MB, raw_mixes 48KB) |

**选择 3-Kernel 的核心理由**：

1. **Cube MatMul 效率远超 Vector 归约**: 原设计以 N=24 过小为由排除 Cube，但忽略了 K=5120 才是计算密集的规约维度。Cube 单元针对大规约维度优化，在 ~55us 内完成整个 [512,5120]x[5120,24] 矩阵乘。Vector 路径需逐元素迭代 K=5120 次做点积规约，效率差距巨大。

2. **GM 中间结果开销远小于计算节省**: residual_flat (10 MB) 的 GM 写+读开销相对于 Vector 路径额外的计算开销微不足道。K1 的 Cube 利用率虽仅 ~19%（受限于 N=24），但仍比 Vector 方案快数个数量级。raw_mixes (48 KB) 的 GM 开销可忽略不计。

3. **硬件亲和性**: K1 使用 Cube 单元（AIC 核心），K0/K2 使用 Vector 单元（AIV 核心），各自承担擅长的计算类型。AIC/AIV 分离符合 DAV_2201 的硬件设计意图。

4. **实测验证**: 端到端延迟 1657us, PyTorch 加速比 32.78x, 全部精度测试通过。

---

## 6. 多核切分策略

### 6.1 K0: 沿 Token 维均匀切分 (48 AIV cores)

```
Shape: residual [B=1, S=512, M=4, H=1280]

切分目标维度: dim=1 (S)
每核 token 数 = ceil(S / vecCoreNum) = ceil(512 / 48) = 11

Core 0:   tokens 0..10
Core 1:   tokens 11..21
...
Core 46:  tokens 506..511
Core 47:  (空闲)
```

- 每核处理分配到的 token 段，以 T=4 tokens/tile 分 tile 迭代
- 尾核 token 数可能减少 (min 实现)
- AIC 核心直接返回（`__global__ __vector__` 修饰，`ASCEND_IS_AIC` 判断）

### 6.2 K1: 2D 块切分 (8 AIC cores)

MatMul [M=512, K=5120, N=24] 的块切分由 `MatmulImpl` + `MatmulApiTiling` 自动管理：

```
M 维切分: singleCoreM = 64, mBlocks = ceil(512/64) = 8
N 维切分: singleCoreN = 24, nBlocks = ceil(24/24) = 1  (N 太小无法再分)
总块数: 8 * 1 = 8 → 使用 8 个 AIC 核心
```

- 每个 AIC 核心处理一个 [64, 5120] x [5120, 24] 子矩阵乘
- 尾块处理: `mBaseTail` / `nBaseTail` 处理非整除情况
- AIV 核心直接返回（`__global__ __cube__` 修饰，`ASCEND_IS_AIV` 判断）

### 6.3 K2: 沿 Token 维均匀切分 (43 AIV cores)

切分策略与 K0 相同（沿 S 维），但仅使用 43 个核心（`vecCoreNum=43`），为 K1 的 8 个 AIC 核心和 K0 的 48 个 AIV 核心留出调度空间。

### 6.4 Tiling 参数总览

| 参数 | K0 | K1 | K2 |
|------|----|----|-----|
| 核心类型 | AIV (48) | AIC (8) | AIV (43) |
| tokensPerCore | ceil(512/48)=11 | — | ceil(512/43)=12 |
| tokensPerTile | 4 | — | 3 |
| singleCoreM | — | 64 | — |
| singleCoreN | — | 24 | — |
| 切分维度 | S (token) | M (block row) | S (token) |

---

## 7. UB Buffer 规划

### 7.1 K0 Buffer (单缓冲，简单转换)

| Buffer | Que 类型 | 大小 (bytes) | 计算 (T=4) |
|--------|---------|-------------|------------|
| `inQueBf16` | VECIN | 81,920 | T * M * H * sizeof(bf16) = 4 * 4 * 1280 * 2 = 40,960 |
| `outQueFp32` | VECOUT | 81,920 | T * D * sizeof(fp32) = 4 * 5120 * 4 = 81,920 |
| **K0 合计** | | **~123 KB** | |

K0 计算简单（仅 Cast），无需双缓冲。UB 预算安全（<192 KB）。

### 7.2 K1 Buffer (Cube MatMul, 内部管理)

K1 使用 `MatmulImpl` 高阶 API，L0C/L1 Buffer 由 Cube 调度器自动管理。用户无需手动规划 UB。

### 7.3 K2 Buffer (单缓冲，多 Que 复用)

| Buffer | Que 类型 | 大小 (bytes) | 计算 (T=3) | 用途 |
|--------|---------|-------------|------------|------|
| `qBf16` | VECIN | 30,720 | T*M*H*2 = 3*4*1280*2 | residual bf16 输入 |
| `qFp32` | VECIN | 61,440 | T*D*4 = 3*5120*4 | residual_flat fp32 输入, 加权中间 |
| `qCalc` | VECCALC | 61,440 | T*D*4 | 平方计算, mixes, sigmoid, 加权, layer_input fp32 |
| `qSpl` | VECCALC | 1,104 | max(T*K, T*M, T*D) = T*D*4 | sqrsum, scale+bias+split, Sinkhorn comb |
| `qOut` | VECOUT | 7,680 | T*H*2 = 3*1280*2 | layer_input bf16 输出 |
| **K2 合计** | | **~162 KB** | | |

**注意**: qCalc/qSpl 在不同阶段复用（sqrsum -> scale+bias+sinkhorn -> weighted -> layer_input），通过时序解耦避免冲突。PipeBarrier 确保前序操作完成后再复用 buffer。

K2 的 M=4 维度极小，Sinkhorn 4x4 矩阵走逐 token 标量循环，不使用 BlockReduceSum 向量归约。

### 7.4 UB 总额验证

| Kernel | UB 使用 | 限制 | 状态 |
|--------|--------|------|------|
| K0 | ~123 KB | 192 KB | 通过 |
| K1 | 由 Cube SDK 管理 | L0C 128KB, L1 512KB | SDK 保证 |
| K2 | ~162 KB | 192 KB | 通过

---

## 8. 数据流详细设计

### 8.1 K0 数据流 (bf16->fp32 转换 + 压平)

```
For each tile (T=4 tokens):
    // Step 1: CopyIn - Load bf16 residual from GM
    DataCopyPad(bf16Buf, residualGm[tokenStart*M*H : ], T*M*H*sizeof(bf16))

    // Step 2: Compute - Cast bf16 -> fp32 (in-place conversion)
    Cast<float, bfloat16_t>(fp32Out, bf16Buf, T*M*H)

    // Step 3: CopyOut - Write fp32 flat to GM
    DataCopy(residualFlatGm[tokenStart*D : ], fp32Out, T*D)
```

### 8.2 K1 数据流 (Cube MatMul)

```
For each (mBlock, nBlock):
    curM = (mBlock == last) ? mBaseTail : singleCoreM   // 64 or tail
    curN = (nBlock == last) ? nBaseTail : singleCoreN   // 24 or tail

    mm_.SetSingleShape(curM, curN, K)
    mm_.SetTensorA(aGm[mBlock*singleCoreM*K : ], false)  // A = fp32, no transpose
    mm_.SetTensorB(bGm[nBlock*singleCoreN*K : ], true)   // B = fp32, transpose
    mm_.IterateAll(cGm[mBlock*singleCoreM*N + nBlock*singleCoreN : ], 0)
```

### 8.3 K2 数据流 (向量后处理)

K2 在单个 kernel 内完成全部后处理，按 token-tile 迭代：

```
For each tile (T=3 tokens):
    // Phase A1: Load residual_flat fp32 from GM
    DataCopyPad(rFp32, residualFlatGm[tokenStart*D : ], T*D*sizeof(fp32))

    // Phase A2: sqrsum = sum(x^2) per token (scalar, D=5120 elements)
    Mul(sqrT, rIn, rIn, T*D)  // vector square
    // per-token sum via GetValue/SetValue loop  (M=4, D=5120 标量累加)

    // Phase A3-A5: RMS Norm on raw_mixes
    //   rms = rsqrt(sqrsum/D + eps)
    //   mixes = raw_mixes * rms  (per-token, via Mul with BinaryRepeatParams)

    // Phase B1: Scale + Bias + Split
    //   mixes[0:M] -> pre (scale[0:M] + base[0:M])
    //   mixes[M:2M] -> post (scale[M:2M] + base[M:2M])
    //   mixes[2M:K] -> comb (scale[2M:K] + base[2M:K])
    Muls(splBuf, splBuf, scaleVec[...])
    // per-element bias via GetValue/SetValue scalar loop

    // Phase B2: Sigmoid (via AscendC::Sigmoid<float>)
    Sigmoid(pre_mix, splBuf[0:M*T], T*M)
    Sigmoid(post_mix, splBuf[M*T:2M*T], T*M)

    // Phase C: Sinkhorn on comb [T, M, M] (M=4, scalar loop)
    For each token:
        softmax_stable: comb -= max_per_row -> Exp -> row_normalize + eps
        For r in 1..R-1:
            col_normalize: comb /= (col_sum + eps)
            row_normalize: comb /= (row_sum + eps)
    DataCopyPad out: comb_mix GM

    // Phase B3 (deferred): Write post_mix to GM

    // Phase D: Apply Mix
    //   Load residual bf16 -> Cast to fp32
    //   weighted[t,m,h] = residual_fp32[t,m,h] * pre_mix[t,m]
    //   layer_input[t,h] = sum_m weighted[t,m,h]  (Add chain, M=4)
    //   Cast fp32 -> bf16
    DataCopyPad out: layer_input GM
```

### 8.4 流水线同步

K0, K1, K2 是三个独立的内核启动（通过 Host 侧三次 `aivector_launch` / `aicube_launch`），之间通过 GM 传递中间数据。Host 侧确保顺序执行（同步模式），不需要 Device 侧跨内核同步机制。

---

## 9. 特殊场景与边界处理

### 9.1 尾 Chunk 处理

| 边界 | 处理方式 |
|------|---------|
| hidden_size 非 HC 整数倍 | cur_HC = min(HC, H - hc*HC)，mask 控制有效元素数 |
| S 非 tile_T 整数倍 | 尾 batch token 数 = min(tile_T, tile_S - batch_idx * tile_T) |
| 多核非均匀分配 | 尾核 token 数 = min(tile_S_per_core, S - core_idx * tile_S_per_core) |

### 9.2 数据对齐

- GM buffer 按 32B 对齐分配
- 使用 `DataCopyPad` 处理非对齐场景
- M=4、H=1280（1280*4=5120 bytes per token per head），fp32 下 5120*4=20480=32*640，完美 32B 对齐
- H=1280，HC=256，256*4=1024=32*32，完美 32B 对齐

### 9.3 极值保护

| 场景 | 保护措施 |
|------|---------|
| `rsqrt(0)` | `rms_eps` 确保分母 > 0 |
| `exp(大值)` | softmax 中添加 `max` 减法做数值稳定性修正 |
| `sigmoid(大正值)` | fp32 下 sigmoid(89) ~ 1, sigmoid(-89) ~ 0，正常截断 |
| 除零 | `sinkhorn_eps` / `pre_eps` 确保分母 > 0 |

---

## 10. 精度策略

### 10.1 数据类型路径

```
输入: residual bf16
  |
  +- Phase 1: 投影 + RMS + Sigmoid + Sinkhorn
  |   +- Cast bf16 -> fp32 (residual 计算用)
  |   +- fn 权重: fp32 (原生)
  |   +- 所有中间计算: fp32
  |   +- 输出 post_mix, comb_mix: fp32
  |
  +- Phase 2: Apply Mix
      +- Cast bf16 -> fp32 (residual 再读取)
      +- pre_mix: fp32 (Phase 1 产出)
      +- 加权与规约: fp32
      +- 输出 layer_input: Cast fp32 -> bf16
```

### 10.2 精度标准

依据 `ops-precision-standard` 浮点计算社区标准：

| 输出 | 类型 | MERE 阈值 | MARE 阈值 |
|------|------|-----------|-----------|
| post_mix | fp32 | 2^-13 ~ 0.000122 | 10 * 2^-13 |
| comb_mix | fp32 | 2^-13 ~ 0.000122 | 10 * 2^-13 |
| layer_input | bf16 | 2^-7 ~ 0.00781 | 10 * 2^-7 |

### 10.3 精度风险点与缓解

| 风险点 | 缓解措施 |
|--------|---------|
| 线性投影累加 (5120 元素) | fp32 累加，避免 bf16 精度不足 |
| RMS 归一化 rsqrt | 使用硬件 `Rsqrt` 指令 |
| Sinkhorn 迭代 10 轮 | fp32 运算，eps 保护，误差不累积放大 |
| Cast fp32 -> bf16 (layer_input) | 单次舍入，无累积误差 |
| Softmax 指数溢出 | max 减法稳定化 |

---

## 11. Tiling 参数 C++ 定义

### 11.1 TilingHeaderK0 (bf16->fp32 转换)

```cpp
struct TilingHeaderK0 {
    int32_t nTokens;           // 512
    int32_t mhcMult;           // 4
    int32_t hiddenSize;        // 1280
    int32_t rgs;               // 5120 = M * H
    int32_t tokensPerCore;     // ceil(512/48) = 11
    int32_t tokensPerTile;     // 4
    int32_t vecCoreNum;        // 48
};
```

### 11.2 TilingHeaderK1 (Cube MatMul)

```cpp
struct TilingHeaderK1 {
    AscendC::tiling::TCubeTiling cubeTiling;  // MatmulApiTiling::GetTiling() 填充
    int32_t mTotalCnt;                         // ceil(M / singleCoreM)
    int32_t nTotalCnt;                         // ceil(N / singleCoreN)
    int32_t totalBlock;                        // mTotalCnt * nTotalCnt
    int32_t mBaseTail;                         // M - (mTotalCnt - 1) * singleCoreM
    int32_t nBaseTail;                         // N - (nTotalCnt - 1) * singleCoreN
    int32_t convTileK;                         // Phase 0 K tile size (保留)
};
```

### 11.3 TilingHeaderK2 (向量后处理)

```cpp
static constexpr int32_t SCALE_VEC_SIZE = 24;
static constexpr int32_t BASE_VEC_SIZE  = 24;

struct TilingHeaderK2 {
    int32_t nTokens;              // 512
    int32_t mhcMult;              // 4
    int32_t hiddenSize;           // 1280
    int32_t mhcMult3;             // 24 = 2M + M^2
    int32_t rgs;                  // 5120 = M * H
    int32_t tokensPerCore;        // ceil(512 / 43) = 12
    int32_t tokensPerTile;        // 3
    int32_t vecCoreNum;           // 43
    int32_t sinkhornRepeat;       // 10
    float   rmsEps;
    float   mhcPreEps;
    float   mhcSinkhornEps;
    float   mhcPostMultValue;
    float   scaleVec[SCALE_VEC_SIZE];  // Host 展开 scale[3] -> [24]
    float   baseVec[BASE_VEC_SIZE];     // mhc_base[24]
};
```

**关键设计决策**: `scale[3]` 和 `base[24]` 在 Host 侧 Tiling 阶段展开为完整向量（`scaleVec[24]`, `baseVec[24]`），通过 TilingHeader 传入 Device。这避免了 Device 侧重复构造广播向量，属于 Tiling 数据的合理预处理（不违反 C9 约束，因为 Tiling 是元数据而非输入 tensor）。
```

---

## 12. Host 侧接口

### 12.1 整体启动流程

```
Host 侧算子执行顺序:
  1. 解析输入 tensor shape，计算 Tiling 参数
  2. Launch K0 (AIV, 48 cores):  bf16 residual -> fp32 residual_flat
  3. Launch K1 (AIC, 8 cores):   residual_flat @ fn^T = raw_mixes
  4. Launch K2 (AIV, 43 cores):  RMS Norm + Sigmoid + Sinkhorn + Apply Mix
```

每次 Launch 为同步调用，K1 等待 K0 的 GM 输出就绪，K2 等待 K1 的 GM 输出就绪。

### 12.2 Kernel 函数签名

```cpp
// K0: bf16->fp32 转换 + 压平
extern "C" __global__ __vector__ void big_fuse_k0_kernel(
    __gm__ bfloat16_t* residualBf16,     // [1, 512, 4, 1280]
    __gm__ float*       residualFlat,    // [512, 5120]
    __gm__ int32_t*     tilingGm);

// K1: Cube MatMul
extern "C" __global__ __cube__ void big_fuse_k1_kernel(
    __gm__ float*   aFp32,              // [512, 5120] residual_flat
    __gm__ float*   bFp32,              // [24, 5120]  fn weight
    __gm__ float*   cFp32,              // [512, 24]   raw_mixes
    __gm__ int32_t* tilingGm);

// K2: 向量后处理
extern "C" __global__ __vector__ void big_fuse_k2_kernel(
    __gm__ float*       residualFlat,   // [512, 5120]
    __gm__ float*       rawMixes,       // [512, 24]
    __gm__ bfloat16_t*  residualBf16,   // [1, 512, 4, 1280]
    __gm__ float*       postMix,        // [512, 4, 1]
    __gm__ float*       combMix,        // [512, 4, 4]
    __gm__ bfloat16_t*  layerInput,     // [512, 1280]
    __gm__ int32_t*     tilingGm);
```

### 12.3 输入输出 Tensor

| 张量 | Shape | dtype | 角色 | Kernel |
|------|-------|-------|------|--------|
| residual | [B, S, M, H] = [1, 512, 4, 1280] | bf16 | 输入 | K0(in), K2(in) |
| fn_weight | [K, D] = [24, 5120] | fp32 | 输入 | K1(in) |
| scale | [3] | fp32 | 输入 (Host 展开到 Tiling) | — |
| base | [K] = [24] | fp32 | 输入 (Host 展开到 Tiling) | — |
| residual_flat | [B*S, D] = [512, 5120] | fp32 | 中间 (GM) | K0(out), K1(in), K2(in) |
| raw_mixes | [B*S, K] = [512, 24] | fp32 | 中间 (GM) | K1(out), K2(in) |
| post_mix | [B, S, M, 1] = [1, 512, 4, 1] | fp32 | 输出 | K2(out) |
| comb_mix | [B, S, M, M] = [1, 512, 4, 4] | fp32 | 输出 | K2(out) |
| layer_input | [B, S, H] = [1, 512, 1280] | bf16 | 输出 | K2(out) |

设计约束 C9：Host 侧**不对**输入 tensor 做预处理（如转置、reshape）。Scale/Base 展开到 TilingHeader 属于元数据预处理，不违反 C9。

---

## 13. 设计决策汇总

| # | 决策 | 原设计 | 实际实现 | 裁决理由 |
|---|------|--------|---------|---------|
| 1 | 技术路线 | SIMD/MemBase | SIMD/MemBase + Cube | DAV_2201，K1 使用 Cube MatMul 高阶 API |
| 2 | Kernel 数量 | 单 Kernel 双阶段 | **3-Kernel 流水线** (K0/K1/K2) | Cube MatMul 效率远超 Vector 归约；GM 中间结果开销可忽略 |
| 3 | 线性投影实现 | Vector Mul + BlockReduceSum | **Cube MatMul (MatmulImpl)** | K=5120 大规约维度，Cube 55us vs Vector 预估 >500us |
| 4 | 多核切分 | 沿 S 维均匀切 (48 AIV) | K0: S 维 48 AIV; K1: M 维 8 AIC; K2: S 维 43 AIV | AIC/AIV 分离，各自处理擅长任务 |
| 5 | UB batch (T) | 16 (双缓冲) | K0: 4, K2: 3 (单缓冲) | 3-Kernel 下单 kernel 计算量减小，单缓冲足够 |
| 6 | 内部精度 | fp32 | fp32 | 投影累加需精度，Sinkhorn 需稳定 |
| 7 | 数据搬运 | DataCopyPad | DataCopyPad (K0/K2), MatmulImpl DMA (K1) | 自动处理对齐 |
| 8 | bf16->fp32 转换 | Kernel 内 inline | **独立 K0 Kernel** (AIV, 48 cores) | 解耦数据准备与计算 |
| 9 | Sinkhorn softmax | max 减法稳定化 | max 减法稳定化 (scalar loop, M=4) | M=4 太小，标量循环无性能损失 |
| 10 | Scale/Base 处理 | Device 侧广播 | **Host 侧展开到 TilingHeader** | 元数据预处理，不违反 C9 |

---

## 14. 实现演进：从单 Kernel 到 3-Kernel 流水线

### 14.1 演进背景

原始 DESIGN.md 采用单 Kernel 双阶段架构，核心理由是"中间张量极小（mixes 48KB, pre_mix 2KB），UB 驻留可避免 GM 往返"。线性投影使用 Vector Mul + BlockReduceSum 实现，明确排除了 Cube 路线（理由："投影矩阵维度太小，远达不到 Cube 收益阈值"）。

Developer 在实现过程中发现了原设计的性能瓶颈，提出 3-Kernel 替代方案并完成实现验证。

### 14.2 原设计的性能瓶颈分析

原设计假设 `[512, 5120] x [5120, 24]` 矩阵乘"太小不适合 Cube"，这一判断存在两个层面的偏差：

1. **错误聚焦 N=24（输出维度）而非 K=5120（规约维度）**: Cube 单元的核心优势在于快速完成大规约维度的乘累加。K=5120 意味着每个输出元素需要 5120 次 FMA，总共 ~126M FLOPs。Vector Mul + BlockReduceSum 需要迭代 5120 次做标量累加，远慢于 Cube 的硬件矩阵乘阵列。

2. **低估了 Cube 在小 N 场景的绝对速度**: 实测 K1 在 ~55us 内完成整个 matmul。即使 Cube 利用率仅 19%（受限于 N=24），绝对延迟仍极低。Vector 路径即使 100% 利用率也无法匹敌。

### 14.3 3-Kernel 方案的正确性论证

#### 为什么 GM 中间结果不是问题

原设计的核心担忧是 GM 中间结果往返浪费带宽。实际测量表明：

| 中间数据 | 大小 | GM 读写开销 | 相对 K1 计算时间 |
|---------|------|------------|----------------|
| residual_flat | 10 MB | K0 写 + K1/K2 读 | 相比 Vector 路径节省的计算量可忽略 |
| raw_mixes | 48 KB | K1 写 + K2 读 | 可忽略不计 |

K1 的 Cube 路径节省的计算时间（~55us vs Vector 预估 >500us）远大于 GM 中间结果的搬运开销。L2 cache (192 MB) 可以充分缓存 10 MB 的 residual_flat，实际 MTE2 利用率已达 ~95%。

#### 为什么 K0 独立是合理的

bf16->fp32 转换 + 压平是纯内存操作（无计算密集度），独立为 K0 的好处：
- 保持 K1 的输入数据格式纯净（contiguous fp32），简化 MatmulImpl 配置
- K0 在 AIV 上执行，不与 K1 (AIC) 争抢 Cube 资源
- 48 个 AIV 核心并行处理，转换开销分摊后极小

#### 为什么 K2 不能进一步拆分

K2 包含 RMS Norm -> Sigmoid -> Sinkhorn -> Apply Mix 四个阶段。若再拆分为独立 kernel：
- post_mix 和 comb_mix 需要在 kernel 间传递（虽然很小）
- Sinkhorn 的中间矩阵 C 和 RMS Norm 的 rms 标量在 UB 内复用可避免 GM 读写
- M=4 时 Sinkhorn 的 4x4 矩阵全域仅 16 元素，kernel 内标量循环无性能损失

因此 K2 作为融合后处理 kernel 是合理的。

### 14.4 裁决结论

**方案 A（接受 3-Kernel 实现）**，理由：

1. **实证优先**: 32.78x PyTorch 加速比，全部精度测试通过，编译和运行验证通过，是充分的可信证据。
2. **架构正确性**: K1 使用 Cube 处理 [512,5120]x[5120,24] 矩阵乘是架构上的正确选择。N=24 虽限制 Cube 利用率，但绝对延迟仍远超 Vector 路径。
3. **GM 流量可接受**: 中间张量（10 MB + 48 KB）的 GM 开销远小于 Cube 路径节省的计算量。
4. **硬件亲和性**: AIC/AIV 分离利用各自擅长的硬件单元，符合 DAV_2201 架构设计意图。
5. **可用性优先**: 已有可用实现优于重新开发单 Kernel 方案，后者存在未验证的性能风险和更长的开发周期。

### 14.5 后续优化方向

基于 profiling 数据，K2 仍有优化空间：

1. **K2 Vector 化**: 当前 K2 的 sqrsum 累加、Sinkhorn row/col normalization 使用 GetValue/SetValue 标量循环，导致 Vector 单元利用率 <0.1%。将 D=5120 的平方和累加替换为 BlockReduceSum，将 M=4 Sinkhorn 的 row/col sum 替换为 ReduceSum (if applicable)，可显著降低 scalar 占比。

2. **K2 Double Buffer**: 将 residual 搬入与计算流水重叠，隐藏 DMA 延迟。

3. **K2 Tile 调优**: 当前 T=3，尝试 T=4 提升并行粒度（需验证 UB 预算）。
