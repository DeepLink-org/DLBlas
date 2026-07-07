# act_quant_kernel 架构设计文档 (DESIGN.md)

> **芯片**: Ascend910B2 (DAV_2201) | **CANN**: 9.0.0 | **核架构**: CubeCore:VectorCore = 1:2 (24:48)
> **路线**: SIMD/MemBase (DAV_2201 通用 AscendC Pipeline API)

---

## 1. 算子数学定义

### 1.1 输入输出

| 张量 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| `x` | `[..., N]` | bf16 / fp16 | 输入激活张量，末维 N 可被 group_size 整除，必须连续 |
| `x_q` | `[..., N]` | fp8_e4m3fn | 量化输出，与 x 同 shape |
| `x_s` | `[..., N//group_size]` | fp32 | 每组的 scale 因子 |

### 1.2 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `group_size` | int | (必需) | 每组元素数，N % group_size == 0 |
| `eps` | float | 1e-10 | absmax 下限 clamp 值 |
| `dtype` | torch.dtype | float8_e4m3fn | 输出数据类型 |
| `scale_ue8m0` | bool | False | 是否将 scale 转为 UE8M0 格式 |

### 1.3 计算公式

```
设 B = numel(x) // group_size, G = group_size
x_ ∈ R^{B × G}  (逻辑 reshape，内存连续)

for b = 0, ..., B-1:
    amax[b]  = clamp(max_j |x_[b,j]|, min=eps) → fp32
    scale[b] = amax[b] / fp8_max                → fp32
    if scale_ue8m0:
        scale[b] = exp2(ceil(log2(max(|scale[b]|, 1e-10))))
    for j = 0, ..., G-1:
        x_q[b,j] = clamp(x_[b,j] / scale[b], fp8_min, fp8_max) → fp8_e4m3fn

输出: x_q 恢复 x 原 shape, x_s shape = [..., N//group_size]
```

其中 fp8_e4m3fn: min = -448.0, max = 448.0 (规范值)。

---

## 2. 硬件环境

| 参数 | 值 |
|------|-----|
| NpuArch | DAV_2201 |
| `__NPU_ARCH__` | 2201 |
| SocVersion | ASCEND910B |
| UB 容量 | 192 KB (196608 B) |
| L1 容量 | 512 KB (524288 B) |
| L0C 容量 | 128 KB (131072 B) |
| VectorCore 数 | 48 |
| 核心类型 | CubeCore + VectorCore (1:2) |
| 数据搬运引擎 | MTE (Memory Transfer Engine) |

---

## 3. 方案决策

### 3.1 路线选择

| 决策属性 | 判定 | 理由 |
|---------|------|------|
| 是否为 Matmul/Cube 类 | 否 | 不涉及矩阵乘法 |
| 目标架构 | DAV_2201 | Ascend910B2 |
| 是否为 DAV_3510 | 否 | → 不走 RegBase / Blaze 路线 |
| **选定路线** | **SIMD/MemBase** | DAV_2201 通用 AscendC Pipeline API |

### 3.2 算子类型判定

主计算形态: **Reduction (per-group absmax) + Elementwise (scale + quantize)**

- 归约维度: 每组 group_size 元素内部 → max(abs(...))
- 逐元素维度: 每组内 group_size 元素的 scale 广播除 + clamp + 量化

### 3.3 Reduction 模式判定

逻辑 reshape x → [B, G] 后:
- A1 = tile_groups (每次 tile 内的组数)
- R = group_size (归约长度)
- A0 = 1 (归约轴是尾轴)
→ **AR 模式**, 使用 Level 2 ReduceMax API (逐行归约)

对于任意合理的 group_size (≤ 65536), 单行数据量远小于 UB (192KB):
→ **AR-FullLoad 分支**

---

## 4. 多核切分策略

### 4.1 任务划分

- 总任务: `num_groups = numel(x) / group_size` 个独立组的处理
- 切分轴: 沿 groups 维度 (维 0 在 [B, G] 视角下)
- 每核任务: `core_groups = ceil(num_groups / 48)` 组
- 核 i 负责 groups [i * core_groups, min((i+1) * core_groups, num_groups))

### 4.2 负载均衡

- 各组计算量完全相同 (group_size 固定)
- 各核分配的 group 数相差不超过 1
- 尾块核可能处理更少 groups, 影响 < 2%

---

## 5. UB 切分与 Buffer 规划

### 5.1 单次处理量 (tile_groups)

tile_groups 需满足 UB 容量约束 (含 Double Buffer):

```
tile_groups ≤ (UB_total - UB_work) / (2 × group_size × (sizeof(T) + 1) + sizeof(float))
```

其中:
- `UB_total` = 192 KB
- `UB_work` ≈ 36 KB (ReduceMax workBuf + scaleBuf + computeBuf)
- `sizeof(T)` = 2 (bf16/fp16)
- 除 2 是因为 Double Buffer 需要双份 I/O buffer

| group_size | tile_groups_max | 推荐 tile_groups |
|-----------|-----------------|-----------------|
| 128 | 213 | 128 |
| 64 | 426 | 256 |
| 32 | 853 | 512 |
| 16 | 1706 | 1024 |

### 5.2 Buffer 清单

| Buffer | 类型 | 大小 | 说明 |
|--------|------|------|------|
| `inQueueX` | TQue\<VECIN, 1\> ×2 | 2 × tile_groups × group_size × sizeof(T) | 输入双缓冲 |
| `outQueueQ` | TQue\<VECOUT, 1\> ×2 | 2 × tile_groups × group_size × 1 | x_q 输出双缓冲 (fp8=1B) |
| `outQueueS` | TQue\<VECOUT, 1\> ×1 | tile_groups × sizeof(float) | x_s 输出单缓冲 (小数据量) |
| `absBuf` | TBuf\<VECCALC\> | tile_groups × group_size × sizeof(T) | abs 中间结果 (可复用 inQueue) |
| `scaleBuf` | TBuf\<VECCALC\> | tile_groups × sizeof(float) | scale 数组 |
| `reduceBuf` | TBuf\<VECCALC\> | 32 KB | ReduceMax 工作空间 |
| `computeBuf` | TBuf\<VECCALC\> | group_size × sizeof(float) × 2 | fp32 中间计算 (K=2: src + dst) |

### 5.3 UB 总量估算 (tile_groups=128, group_size=128, bf16)

| Buffer 项 | 大小 |
|-----------|------|
| inQueueX (DB) | 2 × 128 × 128 × 2 = 65,536 B |
| outQueueQ (DB) | 2 × 128 × 128 × 1 = 32,768 B |
| outQueueS | 128 × 4 = 512 B |
| scaleBuf | 128 × 4 = 512 B |
| reduceBuf | 32,768 B |
| computeBuf | 128 × 4 × 2 = 1,024 B |
| **总计** | **≈ 133 KB** |
| **利用率** | **133/192 ≈ 69%** |

结论: 有充足余量, 可以容纳更大的 tile_groups 或为后续优化留空间。

---

## 6. 数据流设计

### 6.1 整体流水线

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ CopyIn   │ →  │  Compute │ →  │ CopyOut  │ →  │ CopyOut  │
│ (x GM→UB)│    │  (UB)    │    │ (q UB→GM)│    │ (s UB→GM)│
└──────────┘    └──────────┘    └──────────┘    └──────────┘
     ↑                                                
     └────── Double Buffer (ping-pong) ──────────┘
```

### 6.2 逐 Tile 计算流程

对每个 tile (tile_groups 个组):

```
Step 1 [CopyIn]:   DataCopyPad(xGm[offset] → xLocal, blockCount=tileG, blockLen=group*sizeof(T))
Step 2 [Abs]:      Abs(xLocal, absLocal, tileG*group)                          // 元素级并行
Step 3 [Per-Group Loop] for g = 0..tileG-1:
   3a [ReduceMax]: ReduceMax(absLocal[g*groupAlign], groupSize, &scaleTmp)
   3b [Clamp]:     scaleTmp = max(scaleTmp, eps)
   3c [Scale]:     scaleBuf[g] = float(scaleTmp) / fp8_max
   3d [UE8M0]:     if scale_ue8m0: scaleBuf[g] = exp2(ceil(log2(max(|s|, 1e-10))))
Step 4 [Quantize]: for g = 0..tileG-1:
   4a [CastUp]:    Cast<fp32, T>(x_fp32, xLocal[g*groupAlign], CAST_NONE, group)   // bf16/fp16→fp32
   4b [Div]:       Div<fp32>(x_fp32, x_fp32, scaleBuf[g], group)                    // x / scale
   4c [Clamp]:     Mins<fp32>(x_fp32, x_fp32, fp8_max, group)
                    Maxs<fp32>(x_fp32, x_fp32, fp8_min, group)
   4d [Quantize]:  Fp32ToFp8(x_qLocal[g*group], x_fp32, group)                     // fp32→fp8
Step 5 [CopyOut]:  DataCopyPad(qGm[offset_q] ← qLocal, blockCount=tileG, blockLen=group)
                    DataCopyPad(sGm[offset_s] ← scaleLocal, blockCount=tileG, blockLen=4)
```

### 6.3 关键优化: 广播除

Step 4b 可使用 BinaryRepeatParams 一次 API 调用完成所有组的广播除:

```cpp
// 将 [tileG, group] 的每行除以对应的标量
// src1RepStride=0 实现广播: 每组重复使用同一个 scale
BinaryRepeatParams params = {1, 1, 1, group/8, group/8, 0};
for (g = 0; g < tileG; g += batchSize) {
    Div(x_fp32[g*groupAlign], x_fp32[g*groupAlign], scaleBuf[g], mask, batchSize, params);
}
```

> **注意**: fp32 下 mask 上限 64, 即单次最多处理 64 组。tile_groups > 64 时需要分批。也可逐组循环, 因为除法本身是各组的瓶颈而非 kernel launch 开销。

---

## 7. FP8 量化转换策略 (DAV_2201 特殊处理)

### 7.1 硬件约束

DAV_2201 (`__NPU_ARCH__=2201`) 不原生支持 FP8 数据类型。在 kernel 侧:
- `fp8_e4m3fn_t` = `uint8_t` (位表示)
- 标量 Cast API 不支持 fp8 类型 (仅 DAV_3510 支持)
- 向量 Cast 是否支持 fp8 目标类型需要 Developer 在实现阶段通过编译验证

### 7.2 推荐策略

**优先路径**: 使用 AscendC 向量 Cast API 直接做 `Cast<fp8_e4m3fn_t, float>` 转换。
验证方法: Developer 阶段尝试编译 `Cast<fp8_e4m3fn_t, float>(dst_fp8, src_fp32, RoundMode::CAST_ROUND, count)`, 若编译通过则使用。

**兜底路径**: 若向量 Cast 不支持 DAV_2201 fp8 目标类型, 采用软件位操作转换:

```
fp32 → fp8_e4m3fn 转换步骤:
1. ReinterpretCast 提取 fp32 位表示 (sign、exponent、mantissa)
2. Round mantissa 23bit → 3bit (round-to-nearest-even)
3. 处理溢出/下溢/denormal
4. Pack 为 uint8_t: [sign(1) | exponent(4) | mantissa(3)]
5. 输出 x_q 以 uint8_t 形式存储
```

该转换函数可在 Host 侧预计算部分常量 (fp8_min/fp8_max 位表示), 通过 tiling 参数传入 kernel。

### 7.3 精度约束

- fp8_e4m3fn 只有 3 位尾数, 相对精度约 12.5%
- 量化误差主要来自 fp32→fp8 的舍入, 与参考实现 (PyTorch `.to(dtype)`) 应保持 1-ULP 以内
- 由于 fp8 非常低精度, scale 计算 (尤其是 UE8M0 路径) 对精度影响显著, 需重点验证

---

## 8. API 映射表

| 计算步骤 | AscendC API | 参数 | 验证状态 |
|---------|------------|------|---------|
| GM→UB 搬运 (x) | `DataCopyPad` + `DataCopyExtParams` | blockCount=tileG, blockLen=group×sizeof(T) | ✅ 已验证 (api-datacopy.md) |
| 元素级 Abs | `Abs<T>(dst, src, count)` | count = tileG × group | ✅ AscendC 基础 API |
| 逐行 ReduceMax | `ReduceMax<T>(dst, src, tmpBuf, count, false)` | count = group (有效元素数) | ✅ 已验证 (api-reduce.md Level 2) |
| 标量 Clamp (min) | `Max(dst, src, scalar)` | dst=src in-place | ✅ AscendC 基础 API |
| 标量→向量广播 (scale 除) | `Muls<T>(dst, src, invScale, count)` 或 `Div + BinaryRepeatParams` | 逐组处理 | ✅ 已验证 (api-arithmetic.md) |
| elementwise Clamp | `Mins` + `Maxs` | fp32 精度 | ✅ AscendC 基础 API |
| bf16/fp16→fp32 Cast | `Cast<float, T>(dst, src, CAST_NONE, count)` | count = group | ✅ 已验证 (api-precision.md) |
| fp32→fp8 Cast | `Cast<fp8_e4m3fn_t, float>(dst, src, CAST_ROUND, count)` 或 软件转换 | 见 §7 | ⚠️ 待 Developer 编译验证 |
| UB→GM 搬运 (x_q) | `DataCopyPad` + `DataCopyExtParams` | blockCount=tileG, blockLen=group×1 | ✅ 已验证 (api-datacopy.md) |
| UB→GM 搬运 (x_s) | `DataCopyPad` + `DataCopyExtParams` | blockCount=tileG, blockLen=4 | ✅ 已验证 (api-datacopy.md) |
| UE8M0 scale 转换 | `Exp` + `Ln` (或手写位操作) | exp2(ceil(log2(...))) | ✅ AscendC 基础 API |

---

## 9. Double Buffer 流水线

### 9.1 配置

```cpp
// 输入双缓冲
TQue<QuePosition::VECIN, 1> inQueueX;
pipe->InitBuffer(inQueueX, 2, tileGroups * groupSizeAlign * sizeof(T));

// 输出双缓冲 (x_q)
TQue<QuePosition::VECOUT, 1> outQueueQ;
pipe->InitBuffer(outQueueQ, 2, tileGroups * groupSize * sizeof(uint8_t));
```

### 9.2 流水线阶段

```
时间 →
Core 0: [CopyIn(tile0)][CopyIn(tile1)][CopyIn(tile2)]...
Core 1:        [Compute(tile0)][Compute(tile1)][Compute(tile2)]...
Core 2:               [CopyOut(tile0)][CopyOut(tile1)][CopyOut(tile2)]...
```

MTE 搬运与 Vector 计算重叠执行, 预期隐藏大部分数据搬运延迟。

---

## 10. Tiling 参数设计

### 10.1 Host 侧 Tiling 计算

```cpp
struct ActQuantTiling {
    uint32_t numGroups;        // B = numel / group_size
    uint32_t groupSize;        // G = group_size
    uint32_t groupSizeAlign;   // 对齐后的 G (32B 对齐)
    uint32_t tileGroups;       // 每 tile 处理组数
    uint32_t tileNum;          // 每核 tile 数
    uint32_t totalTileNum;     // 全局 tile 数
    float fp8Max;              // fp8_e4m3fn max = 448.0
    float fp8Min;              // fp8_e4m3fn min = -448.0
    float eps;                 // clamp 下限
    bool scaleUe8m0;           // 是否 UE8M0
    DataType inputDtype;       // bf16 或 fp16
};
```

### 10.2 分组对齐

```
groupSizeAlign = ceil(groupSize / (32 / sizeof(T))) × (32 / sizeof(T))
例如: groupSize=128, bf16 → 已对齐 (128×2=256B, 32B 的倍数)
      groupSize=127, bf16 → 对齐到 128 (128×2=256B, 32B 的倍数)
```

### 10.3 Tile 计算

```
coreGroups = ceil(numGroups / 48)
tileGroups = min(recommended_tile_groups, coreGroups)
coreTileNum = ceil(coreGroups / tileGroups)
```

---

## 11. 分支场景覆盖

### 11.1 数据类型分支

| 输入 dtype | 差异点 | 处理 |
|-----------|--------|------|
| bf16 | sizeof=2, 7bit mantissa | 默认路径 |
| fp16 | sizeof=2, 10bit mantissa | 模板参数切换 (无架构差异) |

### 11.2 Scale 格式分支

| scale_ue8m0 | 差异点 | 处理 |
|-------------|--------|------|
| false | scale = amax / fp8_max | 常规路径 |
| true | scale = exp2(ceil(log2(max(|s|, 1e-10)))) | 额外 Exp + Ln 计算 |

### 11.3 Shape 大小分支

| 场景 | 判定 | 处理 |
|------|------|------|
| num_groups < 48 | 部分 core 空闲 | 正常, 无特殊处理 |
| num_groups 巨大 | 多 tile 迭代 | 正常循环 |
| group_size 非 32B 对齐 | 需要 padding | DataCopyPad 自动处理 |

---

## 12. 性能预估与瓶颈分析

### 12.1 计算量分析

每个 group 的计算量:
- Abs: G 次
- ReduceMax: G 次比较
- Scale: 1 次除法 + 可选 Ln/Exp
- Cast (up): G 次
- Div: G 次
- Clamp (Mins+Maxs): 2G 次
- Cast (down): G 次
- **总计约 6G 次操作**

### 12.2 带宽分析

每 group 的数据搬运:
- 读: G × sizeof(T) = 2G 字节 (bf16/fp16)
- 写 (x_q): G × 1 = G 字节 (fp8)
- 写 (x_s): 4 字节 (fp32)
- **总计约 3G+4 字节/group**

Vector Core 理论算力: ~1.8 GHz × 48 cores × 256 MAC/cycle ≈ 22 TFLOPS (fp16)
GM 带宽: ~200 GB/s (A2 系列典型值)

对于 bf16: compute intensity ≈ 6G ops / (3G+4) bytes ≈ 2 ops/byte (极低)
→ **瓶颈在内存带宽, 不在计算**

### 12.3 优化方向

1. **Double Buffer**: 隐藏数据搬运延迟 (已采用)
2. **增大 tile_groups**: 减少 loop 开销, 但受 UB 限制
3. **融合 Abs+ReduceMax**: 若硬件支持, 减少一次数据遍历
4. **Block 量化格式**: 若 group_size=32, 可考虑 MX 格式加速

---

## 13. 数值精度策略

### 13.1 精度风险点

| 风险 | 说明 | 缓解措施 |
|------|------|---------|
| absmax 下溢 | group_size 很大且所有值都很小 | eps clamp 保护 |
| scale 除零 | amax=0 → scale=0 | eps clamp 保证 amax ≥ eps |
| fp8 舍入 | fp32→fp8 有 3bit mantissa 截断 | 使用 CAST_ROUND 就近舍入 |
| UE8M0 精度 | exp2(ceil(log2(...))) 是离散化 | 这是 UE8M0 的预期行为 |

### 13.2 精度标准

参考量化计算类标准 (`quantization.md`):
- 输出 fp8 (浮点) → 双标杆比对
- 精度等级 L1: MARE ratio ≤ 5, MERE ratio ≤ 1.5, RMSE ratio ≤ 1.5
- 或更严格: fp8 结果 1-ULP 一致性 (因 fp8 精度极低, 应允许 1-ULP)

---

## 14. 工程模板

推荐从 AscendC Elementwise 高性能模板出发改造:

```
$ASCEND_TOOLKIT_HOME/tools/op_project_templates/ascendc/elemwise/
```

改造要点:
1. 将 1-in 1-out 改为 1-in 2-out (x_q + x_s)
2. 替换 compute 逻辑为 per-group quantize
3. 添加 ReduceMax workBuf
4. 调整 tiling 参数

---
