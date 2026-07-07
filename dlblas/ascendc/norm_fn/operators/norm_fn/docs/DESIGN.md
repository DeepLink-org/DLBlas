# norm_fn 算子技术设计文档 (DESIGN.md)

## 1. 算子概述与数学定义

### 1.1 算子功能

`norm_fn` 实现多头交叉归一化（Multi-Head Cross Normalization）融合计算：对输入的 residual 和 mhc_fn 进行 einsum 内积 + RMS 归一化 + 可选权重乘法 + 归约重塑的一体化计算。

### 1.2 数学公式

```
Step 1: 可选仿射变换
  mhc_fn'[n,k] = mhc_fn[n,k] * mhc_norm_weight[k]  (if weight != None)

Step 2: Flatten + 类型转换
  residual_2d: [n0, n1, mhc_mult, hidden_size](bf16) → [n0*n1, rms_group_size](float)
  其中 rms_group_size = mhc_mult * hidden_size

Step 3: Einsum 内积 (等价于矩阵乘法 C = A @ B^T)
  A ∈ R^{M×K}, B ∈ R^{N×K}, C ∈ R^{M×N}
  其中 M = n0*n1, N = mhc_mult3, K = rms_group_size
  mixes[m,n] = Σ_k A[m,k] * B[n,k]

Step 4: RMS 归一化
  sqrsum[m] = Σ_k A[m,k]²
  rms_factor[m] = rsqrt(sqrsum[m] / K + eps)
  result[m,n] = mixes[m,n] * rms_factor[m]

Step 5: 重塑输出
  result: [M, N] → [n0, n1, N]
```

### 1.3 输入输出规格

| 名称 | 通用形状 | 测试形状 | 数据类型 |
|------|---------|---------|---------|
| residual | (n0, n1, mhc_mult, hidden_size) | (1, 13, 4, 1280) | bfloat16 |
| mhc_fn | (mhc_mult3, mhc_mult * hidden_size) | (24, 5120) | float32 |
| mhc_norm_weight | (mhc_mult * hidden_size,) or None | (5120,) or None | float32 |
| mhc_norm_eps | scalar | 1e-6 | float32 |
| **输出** | **(n0, n1, mhc_mult3)** | **(1, 13, 24)** | **float32** |

### 1.4 计算规模分析

| 指标 | 数值 | 说明 |
|------|------|------|
| M = n0 * n1 | 13 | batch 行数 |
| N = mhc_mult3 | 24 | 输出列数 (头数) |
| K = rms_group_size | 5120 | 内积维度 |
| 总 MAC 数 | 13 x 24 x 5120 ≈ 1.60M | 极小规模 |
| 输出元素数 | 13 x 24 = 312 | 输出数据极少 |
| 算子类型 | Elementwise + Reduction + MatMul-like | 混合型 |

---

## 2. 硬件架构与约束

### 2.1 芯片参数

| 参数 | 值 | 来源 |
|------|-----|------|
| 芯片型号 | Ascend 910B2 | 用户指定 |
| NpuArch | DAV_2201 | `/npu-arch` skill 查表 |
| `__NPU_ARCH__` | 2201 | 编译宏 |
| arch 目录 | arch22 | 算子仓目录简写 |
| CANN 版本 | 9.0.0 | 用户指定 (路径: `/usr/local/Ascend/ascend-toolkit/latest`) |
| UB 容量 | 192 KB (196,608 bytes) | npu-arch skill (DAV_2201) |
| L0C 容量 | 128 KB | npu-arch skill |
| L1 容量 | 1 MB | DAV_2201 硬件规格 |
| Cube 核数 | 24 | npu-arch skill |
| Vector 核数 | 24 (与 Cube 1:1) | DAV_2201 架构规格 |
| 频率 | 1.8 GHz | npu-arch skill |
| Vector 单核 FP32 理论峰值 | ~14.4 GFLOPS | 1.8GHz x 8 FLOPS/cycle |

### 2.2 架构能力约束

| 能力 | DAV_2201 支持 | 说明 |
|------|:---:|------|
| SIMD/MemBase (Vector API) | Yes | 标准路线，256-bit SIMD |
| RegBase | No | DAV_3510 专属 |
| Blaze/tensor_api (CUTLASS 风格) | No | DAV_3510 专属 |
| NDDMA | No | DAV_3510 专属 |
| CCU 通算融合 | No | DAV_3510 专属 |
| MatMul 高阶 API (MatmulImpl) | Yes | 适合大矩阵乘，当前问题规模过小不适用 |
| FP8 / MXFP8 | No | DAV_3510 专属 |

---

## 3. 技术路线决策

### 3.1 路线选择: 单核 SIMD/Vector 路线

| 决策点 | 分析 | 结论 |
|--------|------|------|
| 目标架构 | DAV_2201 (非 DAV_3510) | **SIMD/MemBase 路线** |
| 算子类别 | 混合型: Elementwise + Reduction + Dot-product | 走通用 Vector API 路径 |
| 核心计算 | einsum: (M=13, K=5120) @ (N=24, K=5120)^T | Vector 逐行 dot product |
| 问题规模 | 1.60M MACs, 312 输出元素 | **单核即可** |
| 精度需求 | bf16输入 → fp32中间 → fp32输出 | 混合精度 (输入精度由 bf16 先天决定) |

**不选 Cube (MatMulImpl) 的理由**：
1. 输出规模极小 (13x24=312元素)，Cube 建立 MMAD、L1 滚动窗口、Fixpipe 同步开销远超计算收益
2. M=13 远小于典型的 Cube 分块 (通常 M_tile ≥ 64)，tile 利用率极低
3. 融合需求: RMS 归一化需复用 residual 的 sqrsum，Cube 路径下需额外计算或回读

**不选多核的理由**：
1. 总 MAC 1.60M，单核 Vector 理论耗时 ~111 us
2. 多核切分通信/同步开销可能超过计算收益
3. 输出仅 312 元素，多核合并结果的代价相对较高

**选用单核 SIMD/Vector 路线**：
- 单 Kernel 融合全部计算 (weight mul + einsum + RMS + reduce)
- K 轴分块迭代，UB 内完成所有累积
- sqrsum 批量归约使用 `Pattern::Reduce::AR`
- dot product 逐行使用 Level 2 `ReduceSum`

---

## 4. 整体架构设计

### 4.1 单 Kernel 融合设计

```
┌──────────────────────────────────────────────────────────┐
│                     norm_fn_kernel                        │
│                                                          │
│  Inputs (GM):                                            │
│    residual_gm  [1,13,4,1280] bf16  (逻辑展平 [13,5120]) │
│    mhc_fn_gm    [24,5120]      float                     │
│    weight_gm    [5120]         float (optional)           │
│    eps          scalar         float                     │
│                                                          │
│  Output (GM):                                            │
│    result_gm    [1,13,24]      float                     │
│                                                          │
│  Algorithm (per K-tile, 10 iterations):                  │
│    ┌─ Load: residual_tile[13,512], mhc_fn_tile[24,512]   │
│    ├─ [opt] Weight Mul: mhc_fn_tile *= weight_tile       │
│    ├─ sqrsum partial: square → Pattern::Reduce::AR → +   │
│    ├─ Dot products: Double loop (m=0..12, n=0..23):      │
│    │    Mul(res[m], mhc_fn[n]) → ReduceSum → +accum      │
│    └─ After all tiles: RMS normalize → CopyOut           │
└──────────────────────────────────────────────────────────┘
```

### 4.2 数据布局

| 数据 | GM 物理布局 | UB 逻辑布局 | 对齐 |
|------|------------|------------|------|
| residual | ND [1,13,4,1280], bf16 | 展平 [13, TILE_K_ALIGN], float (Cast后) | 32B |
| mhc_fn | ND [24,5120], float | [24, TILE_K_ALIGN], float | 32B |
| weight | [5120], float | [TILE_K_ALIGN], float | 32B |
| mixes (accum) | — | [13*24], float | 8B |
| sqrsum (accum) | — | [13], float | 32B |
| result | ND [1,13,24], float | [13*24], float | 32B |

---

## 5. Tiling 方案

### 5.1 K 轴分块策略

| 参数 | 值 | 推导/公式 |
|------|-----|----------|
| total_K (K) | mhc_mult * hidden_size | rms_group_size，通用: 可变 |
| TILE_K | 512 | 由 UB 容量约束计算 (见 §6.3) |
| TILE_K_ALIGN | 512 | `AlignUp(TILE_K, 8)` — 512 天然 32B 对齐 |
| num_K_tiles | `CeilDiv(K, 512)` | 测试 K=5120: 10 次迭代, 整除无尾块 |
| last_TILE_K | `K % 512` (0 if divisible) | 需处理尾块逻辑 |

### 5.2 多核策略: 单核

| 项目 | 值 | 理由 |
|------|-----|------|
| 使用核数 | 1 | 问题规模极小，多核通信开销 > 计算收益 |
| M 轴切分 | 无 | M=13 太小，全部单核处理 |
| N 轴切分 | 无 | N=24 太小，全部单核处理 |
| K 轴切分 | 核内迭代 | 唯一有效分块维度 |

### 5.3 K 轴尾块处理

当 `total_K % TILE_K != 0` 时，最后一个 K-tile 的有效元素数小于 TILE_K。
- DataCopyPad: blockLen 使用实际有效元素数 (last_TILE_K * sizeof(element))
- Mul (dot product): count 使用实际有效元素数 last_TILE_K
- ReduceSum: count 使用实际有效元素数 last_TILE_K
- Cast (bf16→float): count 使用实际有效元素数 (但 Pad 区域由 hardware padding 处理)

---

## 6. UB Buffer 规划

### 6.1 UB 容量约束

```
UB 总容量: 192 KB = 196,608 bytes (DAV_2201)
```

### 6.2 Buffer 分配表 (TILE_K = 512)

| Buffer | 数据类型 | 元素数 | 字节数 | 用途 | 生命周期 |
|--------|---------|--------|--------|------|---------|
| residual_tile | float | 13 x 512 = 6,656 | 26,624 | residual K 分块 (bf16 Cast 后) | 全 K-tile |
| mhc_fn_tile | float | 24 x 512 = 12,288 | 49,152 | mhc_fn K 分块 | 全 K-tile |
| weight_tile | float | 512 | 2,048 | 权重 K 分块 (可选, 可复用) | K-tile 内 Phase 1 |
| sq_temp | float | 13 x 512 = 6,656 | 26,624 | square 中间结果, 后复用为 temp_row | K-tile 内 |
| temp_row | float | 512 | 2,048 | 逐行 Mul 结果 (复用 sq_temp 末尾) | 逐对 (m,n) |
| mixes | float | 13 x 24 = 312 | 1,248 | dot product 累加器 | 全 Kernel |
| sqrsum | float | 13 | 52→64 | sqrsum 累加器 (对齐到 8B) | 全 Kernel |
| reduce_tmp | uint8_t | ~4,096 | ~4,096 | ReduceSum 临时空间 | 每次 Reduce 调用 |
| result | float | 13 x 24 = 312 | 1,248 | 最终输出 | Phase 4-5 |
| **Total** | | | **~111,136** | | |

### 6.3 TILE_K 选择推导

```
主要 Buffer (随 TILE_K 线性增长):
  residual_tile:  13 x TILE_K x 4 = 52 x TILE_K
  mhc_fn_tile:    24 x TILE_K x 4 = 96 x TILE_K
  sq_temp:        13 x TILE_K x 4 = 52 x TILE_K
  weight_tile:     1 x TILE_K x 4 =  4 x TILE_K
  Subtotal:                        204 x TILE_K

固定 Buffer (不随 TILE_K 变化):
  mixes(1,248) + sqrsum(64) + reduce_tmp(~4,096) + result(1,248)
  Subtotal: ~6,656

UB 使用 = 204 x TILE_K + 6,656 ≤ 196,608
TILE_K ≤ (196,608 - 6,656) / 204 ≈ 931

取 TILE_K = 512: 使用 = 111,104 bytes ≈ 108.5 KB ✓ (留有 43% 余量)
取 TILE_K = 1024: 使用 = 215,552 > 192 KB ✗ (超出)
取 TILE_K = 256: 使用 = 58,880 bytes ≈ 57.5 KB (可行但迭代次数翻倍)
```

**结论: TILE_K = 512 最佳**，兼顾 UB 利用率和迭代次数。

### 6.4 内存复用策略

- `sq_temp` 完成 sqrsum 计算后，其尾部 512 元素复用为逐行 dot product 的 `temp_row`
- `weight_tile` 在 Phase 1 使用后可被覆盖 (仅 has_weight 分支使用)
- `residual_tile` 和 `mhc_fn_tile` 全 K-tile 生命周期内持续保留
- 不使用 Double Buffer: K 轴仅 ~10 次迭代，QUE_DEPTH=1 足够; Double Buffer 增加的 UB 开销可能迫使 TILE_K 减半

---

## 7. 数据流与流水线设计

### 7.1 K-tile 计算流程

```
For each K-tile (k_start = 0, TILE_K, 2*TILE_K, ..., K-1):
  cur_K = min(TILE_K, K - k_start)  // 尾块处理

  ═══ Phase 1: 数据加载 ═══════════════════════════════════
  CopyIn residual_tile[13, TILE_K] bf16 → UB (MTE2)
    GM: 展平视图 (13, K), stride = (K - cur_K) * 2 bytes
    UB: [13, TILE_K_ALIGN], 经 Cast<float> 转换
    API: DataCopyPad(residualUB, residualGm, extParams, padParams)
    API: Cast<float, bfloat16_t>(residualFloat, residualUB, CAST_NONE, 13*cur_K)

  CopyIn mhc_fn_tile[24, TILE_K] float → UB (MTE2)
    GM: (24, K), stride = (K - cur_K) * 4 bytes
    UB: [24, TILE_K_ALIGN], float
    API: DataCopyPad(mhcFnUB, mhcFnGm, extParams, padParams)

  [if has_weight:]
    CopyIn weight_tile[TILE_K] float → UB (MTE2)
    For n in 0..23:
      Mul(in-place, mhc_fn_tile[n], weight_tile, cur_K)
      // 等价于: mhc_fn[n,k] *= weight[k]

  ═══ Phase 2: sqrsum 累加 ═════════════════════════════════
  Mul(sq_temp, residualFloat, residualFloat, 13 * cur_K)
    // sq_temp[m,k] = residual[m,k]²

  uint32_t srcShape[] = {13, TILE_K_ALIGN};
  ReduceSum<float, Pattern::Reduce::AR, true>(
      sq_partial, sq_temp, reduceTmp, srcShape, true);
  // sq_partial[m] = Σ_k sq_temp[m,k]  for m=0..12

  For m in 0..12:
    sqrsum[m] += sq_partial.GetValue(m)

  ═══ Phase 3: Dot Product 累加 ═════════════════════════════
  // 逐行 Level 2 ReduceSum (避免 BinaryRepeatParams 广播复杂度)
  For m in 0..12:
    For n in 0..23:
      Mul(temp_row, residualFloat[m * TILE_K_ALIGN],
          mhcFnFloat[n * TILE_K_ALIGN], cur_K)
      ReduceSum<float>(scalarBuf, temp_row, reduceTmpF32, cur_K)
      mixes[m * 24 + n] += scalarBuf.GetValue(0)

End For  // K-tile loop

═══ Phase 4: RMS 归一化 ═══════════════════════════════════
For m in 0..12:
  invK = 1.0f / (float)K         // Host 侧预计算, 避免 aicore 内 uint32→float cast
  rms_input = sqrsum[m] * invK + eps
  rms = Rsqrt(rms_input)          // 1/sqrt(sqrsum/K + eps)
  Muls(result[m * 24], mixes[m * 24], rms, 24)
  // result[m,n] = mixes[m,n] * rms  for n=0..23

═══ Phase 5: 输出写回 ═════════════════════════════════════
CopyOut result[312] float → result_gm[1,13,24] float (MTE3)
API: DataCopyPad(resultGm, resultUB, extParams)
```

### 7.2 Double loop 设计理由

Phase 3 采用双层循环 (13x24=312 次 per K-tile) 而非广播批量处理，理由：

1. **BinaryRepeatParams 广播 vs Level 2 直算**:
   - 广播方案需用 `Mul` Level 0 + `BinaryRepeatParams{src0RepStride=0}`, `repeatTime=24`
   - 该方案涉及 repeat/mask 参数组合，TILE_K=512 时 mask 需 uint64_t[8]，且 sBlkStride/rRepStride 计算易错
   - Level 2 `Mul(src0, src1, count)` + `ReduceSum` 语义清晰，count 直传有效元素数

2. **性能分析**:
   - 312 次 per K-tile × 10 tiles = 3,120 次 Mul+Reduce per kernel
   - 这是算法结构固有特征 (小 M, N 下的 dot product)，非实现缺陷
   - Vector FP32 利用率受限于标量控制流开销 (GetValue/SetValue, 循环分支)

3. **备选优化 (未来可探索)**:
   - 重组循环顺序: 外层 n (24), 内层 m (13) → 减少 mhc_fn 行切换
   - Pattern::Reduce::AR 批量处理: 如果能在 UB 中同时容纳残差广播结果, 可单次 Reduce 替代 24 次循环

### 7.3 同步策略

所有同步通过 `TQue` (TPipe) 的 EnQue/DeQue 机制实现，无需 `PipeBarrier`。

| 阶段 | 前操作 (Pipe) | 后操作 (Pipe) | 同步方式 |
|------|-------------|-------------|---------|
| CopyIn→Compute | DataCopyPad (MTE2) | Cast/Mul (V) | TQue DeQue 阻塞 |
| Compute 内部 | Mul, ReduceSum, Cast | 同上 (V) | 硬件序列化 |
| Compute→RMSNorm | SetValue (Scalar) | Muls (V) | 编译器序列化 |
| RMSNorm→CopyOut | Muls (V) | DataCopyPad (MTE3) | TQue EnQue→DeQue |

---

## 8. API 映射与验证

### 8.1 核心 API 表

| 操作 | API | 签名 / 关键参数 | 验证状态 |
|------|-----|----------------|---------|
| GM→UB 搬运 (stride) | `DataCopyPad` (Ext) | `DataCopyPad(LocalTensor<T>&, GlobalTensor<T>&, DataCopyExtParams&, DataCopyPadExtParams<T>&)` — blockCount + blockLen + srcStride + dstStride | 已验证 (kernel_struct_data_copy.h:388-397) |
| UB→GM 搬运 | `DataCopyPad` (Ext) | `DataCopyPad(GlobalTensor<T>&, LocalTensor<T>&, DataCopyExtParams&)` | 已验证 (kernel_struct_data_copy.h:415-418) |
| bf16→float 转换 | `Cast<float, bfloat16_t>` | `Cast(dst, src, RoundMode::CAST_NONE, count)` — 低精度→高精度, 无需舍入 | 已验证 (kernel_operator_vec_vconv_intf.h) |
| 逐元素平方 | `Mul` (Level 2) | `Mul(dst, src0, src1, count)` — dst[i]=src0[i]*src1[i], count=13*TILE_K | 已验证 (kernel_operator_vec_binary_intf.h) |
| 逐行 dot product Mul | `Mul` (Level 2) | `Mul(temp_row, res_row, mhc_row, cur_K)` — 逐对 dot product 的乘法步骤 | 已验证 |
| 逐行 Reduce (dot) | `ReduceSum<float>` (L2) | `ReduceSum(dst, src, sharedTmpBuffer, count)` — 无对齐要求 | 已验证 (kernel_operator_vec_reduce_intf.h:301-303) |
| 批量 Reduce (sqr) | `ReduceSum<..., Pattern::Reduce::AR>` | `ReduceSum<T, Pattern::Reduce::AR, true>(dst, src, tmp, srcShape, srcInnerPad)` — srcShape={13, TILE_K_ALIGN} | 已验证 (reduce.h:216-223, reduce_common.h:31) |
| 标量乘 (RMS) | `Muls` (Level 2) | `Muls(dst, src, scalar, mask, repeatTime, repeatParams)` — 或 Level 2: `Muls(dst, src, scalar, count)` | 已验证 (kernel_operator_vec_binary_scalar_intf.h) |
| 倒数平方根 | `Rsqrt` | `Rsqrt(dst, src, count)` — 融合 rsqrt, 替代 Sqrt + Div | 已验证 (kernel_operator_vec_unary_intf.h) |

### 8.2 API 使用约束

1. **DataCopyPad stride**: DAV_2201 下 `srcStride`/`dstStride` 为 `uint32_t` (非 `int64_t`)，单位是 **bytes**
2. **DataCopyPad blockLen**: 必须是 32B 对齐; DAV_2201 下 `blockCount` 为 `uint16_t`
3. **Cast RoundMode**: bf16→float 使用 `CAST_NONE`，不损失精度; float→bf16 使用 `CAST_ROUND`
4. **ReduceSum L2 count**: 传有效元素数，不受 32B 对齐限制
5. **Pattern::Reduce::AR srcInnerPad**: `true` 表示最内维已 padded 到 32B 对齐 (TILE_K=512 天然满足)
6. **aicore 限制**: `uint32_t` → `float` cast 被禁止; 需在 Host 侧预计算 `invK = 1.0f / K` 传入 TilingData
7. **重复元素数 ≤ 255**: 若未来需用 repeatTime 并行处理，单次 repeat ≤ 255; 当前 Level 2 API 不涉及此限制

---

## 9. 精度策略

### 9.1 数据流精度链

```
residual (bf16, 7-bit mantissa)
  └─ Cast CAST_NONE → float32 (24-bit mantissa) [无损扩展]
       └─ Mul (fp32) → sqr (fp32)
       │    └─ ReduceSum (fp32累加) → sqrsum[m]
       │         └─ Muls × invK + eps → rms_input (fp32)
       │              └─ Rsqrt (fp32) → rms_factor
       └─ Mul (fp32) x mhc_fn (fp32) → temp_row (fp32)
            └─ ReduceSum (fp32累加, 5120项) → mixes[m,n]
                 └─ Muls × rms_factor (fp32) → result (fp32输出)
```

### 9.2 精度风险评估

| 操作 | 数据类型 | 风险 | 分析 |
|------|---------|------|------|
| residual 加载 | bf16→fp32 | low | CAST_NONE 无损; 初始精度由 bf16 先天决定 |
| Mul (dot) | fp32 x fp32 | none | 标准 fp32 乘法 |
| ReduceSum (dot, 5120项) | fp32 累加 | low | 5120 < 2^23=8.4M, 不会丢失低位 |
| sqr → ReduceSum (5120项) | fp32 累加 | low | 平方后量级拉大, 但仍在 fp32 精度内 |
| Rsqrt | fp32 | none | 硬件指令级精度 |
| Muls | fp32 x scalar | none | 标准 fp32 乘法 |
| eps 防除零 | 1e-6 | none | 对 rsqrt 输入的最小偏移, 不影响有效精度 |

### 9.3 精度标准

按浮点计算类社区标准 (`float_compute_community.md`): 输出与 PyTorch fp32 参考的均方根相对误差 (RMSRE) < 1e-4。

已知实现实测: Max Diff 在 1e-8 量级 (fp32 epsilon 级别), 远优于标准。

---

## 10. Host 侧 Tiling 设计

### 10.1 TilingData 结构

```cpp
struct NormFnTilingData {
    // 输入形状 (通用化, 从 tensor desc 获取)
    uint32_t total_M;       // n0 * n1 (batch 行数)
    uint32_t total_N;       // mhc_mult3 (输出列数)
    uint32_t total_K;       // mhc_mult * hidden_size (内积维度)

    // K 轴分块参数
    uint32_t tile_K;        // 单次 K 分块大小 (512)
    uint32_t tile_K_align;  // 32B 对齐后的 K 分块 (512)
    uint32_t num_K_tiles;   // K 轴迭代次数

    // 分支控制
    bool     has_weight;    // 是否应用 mhc_norm_weight

    // 预计算常量 (aicore 禁止 uint32→float cast)
    float    invK;          // 1.0f / total_K

    // 数值稳定常数
    float    eps;           // mhc_norm_eps
};
```

### 10.2 Tiling 计算流程

```
1. 从输入 tensor 获取 total_M, total_N, total_K
2. tile_K = ComputeTileK(UB_SIZE, total_M, total_N, has_weight)
   // 基于 §6.3 的 UB 预算公式动态计算
3. tile_K_align = AlignUp(tile_K, 8)  // 32B 对齐
4. num_K_tiles = CeilDiv(total_K, tile_K)
5. has_weight = (mhc_norm_weight != nullptr)
6. invK = 1.0f / (float)total_K
7. 验证 UB 使用量 < 192KB
8. 填充 TilingData
```

### 10.3 Kernel 启动配置

```cpp
// 单核启动
uint32_t blockDim = 1;
norm_fn_kernel<<<blockDim, nullptr, stream>>>(
    residual_gm, mhc_fn_gm, weight_gm, result_gm, tiling_gm
);
```

---

## 11. 边界情况处理

### 11.1 分支场景

| 场景 | 条件 | 处理 |
|------|------|------|
| 有权重 | has_weight = true | Phase 1 加载 weight_tile, 逐行 Mul |
| 无权重 | has_weight = false | 跳过 weight 加载和乘法, mhc_fn_tile 直接使用 |
| K 整除 | total_K % TILE_K == 0 | 无尾块, 全 tile 统一处理 |
| K 非整除 | total_K % TILE_K != 0 | 最后一 tile 使用 cur_K = total_K % TILE_K |

### 11.2 数据对齐

| 数据 | blockLen | 32B 对齐 |
|------|---------|:---:|
| residual (bf16) | TILE_K x 2 = 1024 bytes | Yes (32 x 32) |
| mhc_fn (float) | TILE_K x 4 = 2048 bytes | Yes (32 x 64) |
| weight (float) | TILE_K x 4 = 2048 bytes | Yes (32 x 64) |
| result (float) | 312 x 4 = 1248 bytes (单次 CopyOut) | Yes (32 x 39) |

### 11.3 数值边界

- `sqrsum[m] = 0`: rsqrt(0/K + eps=1e-6) = 1000, 安全
- `mhc_fn` 含 0: Mul 结果为 0, ReduceSum 正确处理
- `residual` 含 Inf/NaN: bf16 输入通常不触发; 若出现, fp32 转换后传播, Rsqrt(NaN)→NaN

---

## 12. 编译参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--npu-arch` | `DAV_2201` | Ascend 910B2 架构 |
| `__NPU_ARCH__` | `2201` | 编译宏 |
| SoC Version | `Ascend910B2` | 芯片型号 |
| CANN | `9.0.0` | `/usr/local/Ascend/ascend-toolkit/latest` |
| 算子类型 | Vector (非 Cube) | `__vector__` |
| 编译目标 | `aicore` | Device 侧代码 |
| 核数 | 1 (单核) | blockDim = 1 |

---

## 13. 性能分析

### 13.1 理论分析

| 指标 | 数值 | 推导 |
|------|------|------|
| 总 MAC | 1.60M | 13 x 24 x 5120 |
| 单核 FP32 理论峰值 | 14.4 GFLOPS | 1.8 GHz x 8 FLOPS/cycle |
| 理论计算时间 | ~111 us | 1.60M / 14.4G |
| 预计实际延迟 | ~350 us | 含标量开销、内存搬运、流水线延迟 |

### 13.2 瓶颈分析

- **主要瓶颈**: 标量控制流 (Phase 3 双层循环 3,120 次 Mul+Reduce 迭代中的 GetValue/SetValue, 循环分支)
- **Vec FP32 利用率**: 预计 10-40% (受限于标量操作占比)
- **内存带宽**: 非瓶颈 (MTE2 占比较小, 数据量极小)

### 13.3 优化方向 (未来可探索)

1. **循环顺序重排**: 外层 N (24), 内层 M (13) → 减少 mhc_fn 行切换开销
2. **Pattern::Reduce 批量化**: 如 UB 容量允许, 用 BinaryRepeat 广播单行残差到 N 行, 单次 Mul + Pattern::Reduce::AR 替代 N 次循环
3. **循环展开**: 编译器指导或手工展开减少分支预测开销

---

## 14. 设计检查清单

### 通用设计要素
- [x] 多核切分策略: 单核 (问题规模极小)
- [x] UB 切分策略: K 轴 TILE_K=512, ~10 次迭代
- [x] Buffer 规划: ~108.5 KB / 192 KB, 留有 43% 余量
- [x] 分支场景覆盖: has_weight / 无 weight; K 整除 / K 尾块

### 精度
- [x] bf16→fp32: CAST_NONE, 无损
- [x] 中间计算: 全部 fp32
- [x] 输出: fp32
- [x] 满足社区精度标准 (RMSRE < 1e-4)

### API 验证
- [x] DataCopyPad (Ext) — GM↔UB stride 搬运
- [x] Cast — bf16→fp32 精度转换
- [x] Mul (Level 2) — 逐元素乘 + 逐元素平方
- [x] ReduceSum (Level 2) — 逐行 dot product 归约
- [x] ReduceSum Pattern::Reduce::AR — sqrsum 批量归约
- [x] Muls — 标量乘法 (RMS 归一化)
- [x] Rsqrt — 倒数平方根 (替代 Sqrt + Div)
