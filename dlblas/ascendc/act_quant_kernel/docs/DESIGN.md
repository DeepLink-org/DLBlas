# act_quant_kernel 架构设计方案

## 1. 算子概述

| 属性 | 值 |
|------|-----|
| 算子名称 | act_quant_kernel |
| 算子类型 | Elementwise + Reduction 混合（per-group 量化） |
| 输入 | `x`: [\*shape] in bf16/fp16; `group_size`: int scalar; `eps`: float scalar; `dtype`: fp8 类型; `scale_ue8m0`: bool |
| 输出 | `x_q`: [\*shape] in fp8_e4m3fn; `x_s`: [\*shape[:-1], shape[-1]//group_size] in fp32 |
| 核心语义 | 将浮点激活值沿最后一维按 group_size 分组，每组独立计算 abs_max 得到 scale，进行 FP8 量化 |
| 目标架构 | Ascend910B2, DAV_2201 |
| CANN 版本 | 9.0.0 |

## 2. 方案决策

### 2.1 技术路线

| 决策维度 | 选择 | 理由 |
|---------|------|------|
| 架构 | DAV_2201 | 目标芯片 Ascend910B2 |
| NpuArch | `DAV_2201`, `__NPU_ARCH__=2201` | 通过 `/npu-arch` skill 查询确认 |
| `--npu-arch` 编译参数 | `dav_2201` (vec 变体) | 纯向量运算，不涉及 Cube |
| 编程路线 | **SIMD/MemBase（通用路线）** | DAV_2201 非 DAV_3510，不适用 RegBase/Blaze；算子为 vector 类 |
| 算子分类 | Reduction + Elementwise 混合 | 归约部分走 AR 模式，量化部分走逐元素计算 |
| Tiling 方法 | AR-FullLoad | group_size ≤ 512，单组数据远小于 UB 192KB，可全载 |

**路由决策树**：
```
算子类型判断: per-group abs_max + scale + quantize
  └─ 归约部分: 沿 group_size 维做 ReduceMax → AR 模式
  └─ 量化部分: 逐元素 scale/div/clamp/cast → Elementwise
架构判断: DAV_2201 → SIMD/MemBase 路线
  └─ 不适用 RegBase（非 DAV_3510）
  └─ 不适用 Blaze（非 MatMul/Cube 类）
```

### 2.2 设计方法论来源

- **Tiling 设计**: `/ascendc-tiling-design` → Reduction AR 模式 (ar-fullload.md) + Elementwise 逐元素模式
- **API 最佳实践**: `/ascendc-api-best-practices` → 标量广播 (Adds/Muls)、Buffer 复用
- **精度标准**: `/ops-precision-standard` → 浮点计算类社区标准（bf16/fp16 输入，fp32 中间计算，fp8 输出）

## 3. 数学定义与算法流程

### 3.1 数学公式

```
给定: x ∈ R^{... × D}, group_size | D

1. 分组重塑:
   x_ = x.reshape(B, G)  其中 B = numel(x) / G, G = group_size

2. 逐组计算 scale:
   amax[b] = clamp(max_j(|x_[b, j]|), min=eps)     # abs → reduce max → clamp min
   scale[b] = amax[b] / fp8_max                      # fp8_max = 448.0 for e4m3fn

   可选 (scale_ue8m0):
   scale[b] = exp2(ceil(log2(max(|scale[b]|, 1e-10))))

3. 量化:
   x_q[b, j] = clamp(x_[b, j] / scale[b], fp8_min, fp8_max)  # fp8_min = -448.0
   x_q_fp8[b, j] = float_to_fp8_e4m3(x_q[b, j])

4. 恢复形状:
   x_q = x_q_fp8.reshape(原始shape)
   x_s = scale.reshape(原始shape[:-1] + (D // G,))
```

### 3.2 fp8_e4m3fn 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| fp8_max | 448.0 | 最大可表示正值 (0b01111110 = 1.75 × 2^8) |
| fp8_min | -448.0 | 最小可表示负值 |
| eps 默认 | 1e-10 | amax 下限，防止除零 |
| scale_ue8m0_min | 1e-10 | ue8m0 模式下 scale 绝对值下限 |

## 4. API 映射表

以下 API 均通过 DAV_2201 头文件验证（路径: `/usr/local/Ascend/cann-9.0.0/aarch64-linux/ascendc/include/basic_api/impl/`）。

### 4.1 数据搬运

| 操作 | API | 验证状态 | 备注 |
|------|-----|:---:|------|
| GM → UB 加载 | `DataCopy` / `DataCopyPad` | ✅ | 32B 对齐用 DataCopy，非对齐用 DataCopyPad |
| UB → GM 存储 | `DataCopy` / `DataCopyPad` | ✅ | 同上 |

### 4.2 归约操作

| 操作 | API | 签名 | 验证状态 |
|------|-----|------|:---:|
| 逐组求最大值 | `ReduceMax` (Level 2) | `ReduceMax(dst, src, sharedTmpBuffer, count)` | ✅ |
| 逐元素绝对值 | `Abs` (Level 2) | `Abs(dst, src, count)` | ✅ |

**类型限制**: `ReduceMax` 和 `Abs` 的 Level 2 版本均支持 `half` / `float` 类型。

### 4.3 逐元素计算

| 操作 | API | 签名 | 验证状态 |
|------|-----|------|:---:|
| 标量乘法 | `Muls` (Level 2) | `Muls(dst, src, scalar, count)` | ✅ |
| 标量加法 | `Adds` (Level 2) | `Adds(dst, src, scalar, count)` | ✅ |
| 张量除法 | `Div` (Level 0/2) | `Div(dst, src0, src1, mask/count, ...)` | ✅ |
| 张量乘法 | `Mul` (Level 2) | `Mul(dst, src0, src1, count)` | ✅ |
| 自然对数 | `Ln` (Level 2) | `Ln(dst, src, count)` | ✅ |
| 指数函数 | `Exp` (Level 2) | `Exp(dst, src, count)` | ✅ |

### 4.4 比较与选择

| 操作 | API | 签名 | 验证状态 |
|------|-----|------|:---:|
| 张量比较 | `Compare` (Level 0) | `Compare(dst, src0, src1, cmpMode, ...)` | ✅ |
| 按掩码选择 | `Select` (Level 0/2) | `Select(dst, selMask, src0, src1, ...)` | ✅ |

**Clamp 实现**: 使用 `Compare` + `Select` 组合实现 min/max 截断。

### 4.5 标量广播

| 操作 | API | 签名 | 验证状态 |
|------|-----|------|:---:|
| 广播填充 | `Duplicate` (Level 2) | `Duplicate(dst, scalar, count)` | ✅ |

**类型限制**: `Duplicate` 在 DAV_2201 上支持 `half` / `float` / `int8_t` 等。不支持运行时 `float → int8_t` 类型转换，需单独使用 Cast。

### 4.6 精度转换

| 操作 | API | 签名 | 验证状态 |
|------|-----|------|:---:|
| 类型转换 | `Cast` (Level 2) | `Cast(dst, src, roundMode, count, ...)` | ✅ |

**DAV_2201 Cast 支持的类型转换**（已验证）:
- `half ↔ int8_t` / `half ↔ uint8_t`
- `float → bfloat16_t`
- **不支持 fp8** 类型——DAV_2201 硬件无原生 FP8 Cast 指令

### 4.7 FP8 输出策略（架构适配）

由于 DAV_2201 不支持原生 FP8 Cast，FP8 输出采用以下策略：

1. **UB 内计算**: 所有量化计算在 FP32 精度下完成
2. **float32 → fp8_e4m3 转换**: 通过软件位操作实现（提取符号/指数/尾数，按 e4m3 格式编码）
3. **存储**: 将 8-bit 编码结果写入 `int8_t` 类型的 UB buffer，通过 `DataCopy` 搬出到 GM
4. **Host 侧类型解释**: Host 侧将 int8 GM buffer 解读为 fp8_e4m3fn 类型

> **补充说明**: 若未来 CANN 版本在 DAV_2201 上通过软件模拟支持 FP8 Cast，可替换为此处的手动位转换逻辑。DAV_3510 上直接使用 `Cast<float, fp8_e4m3fn_t>` 即可。

## 5. Tiling 策略

### 5.1 维度分析

输入 x 的形状重塑后:

```
x_reshape: [B, G]
  B = numel(x) / group_size    # 总组数（归约的外层维度 A）
  G = group_size               # 每组大小（归约轴 R, A0=1）
```

合轴判定: **AR 模式**（A0=1，单轴归约，归约轴为尾轴），每个 `[1, G]` 行独立归约。

### 5.2 AR 分支选择: FullLoad

| 条件 | 判断 | 结论 |
|------|------|------|
| 可在 UB 中至少处理 1 整行数据？ | G ∈ {128, 512}, 最大 512 × 4B = 2KB << 192KB UB | ✅ YES |
| 选择分支 | AR-FullLoad | 整行数据驻留 UB，一次 CopyIn 完成归约 |

### 5.3 Tiling 参数

| 参数 | 含义 | 计算公式 |
|------|------|---------|
| `totalGroups` | 总组数 | `x.numel() / group_size` |
| `groupsPerCore` | 每核处理组数 | `ceil(totalGroups / coreNum)` |
| `groupsPerBatch` | UB 内单次批处理组数 | 由 UB 预算决定（见 §6.2） |
| `rLength` | 归约轴有效长度 | `group_size`（如 128 或 512） |
| `rLengthAlign` | 归约轴 32B 对齐长度 | `AlignUp(group_size * sizeof(fp32), 32) / sizeof(fp32)` |

### 5.4 Tiling 循环结构

```
for batch_idx in range(0, groupsPerCore, groupsPerBatch):
    actual_batch = min(groupsPerBatch, groupsPerCore - batch_idx)
    
    // Step 1: 加载数据 (bf16/fp16 → fp32)
    DataCopy: GM[offset] → UB[input_buf], size = actual_batch × G × sizeof(input_dtype)
    Cast: input_fp32 = cast(input_bf16, fp32)
    
    // Step 2: 逐组归约求 amax
    for g in range(actual_batch):
        Abs(group_data_fp32, G)
        ReduceMax(amax[g], group_data_fp32, tmpBuf, G)
        
    // Step 3: 逐组计算 scale
    for g in range(actual_batch):
        amax[g] = max(amax[g], eps)
        scale[g] = amax[g] / fp8_max
        if scale_ue8m0:
            scale[g] = round_pow2_ceil(scale[g])
    
    // Step 4: 逐组量化
    for g in range(actual_batch):
        Duplicate: broadcast scale[g] → scale_broadcast[G]
        Div: x_scaled = group_data / scale_broadcast
        Clamp: x_clamped = clamp(x_scaled, fp8_min, fp8_max)
        Float2FP8: x_fp8 = float_to_fp8_e4m3(x_clamped)
    
    // Step 5: 写回
    DataCopy: UB[x_fp8] → GM[x_q_out + offset], size = actual_batch × G × 1
    DataCopy: UB[scale] → GM[x_s_out + scale_offset], size = actual_batch × 4
```

## 6. 多核切分策略

### 6.1 切分方式

- **切分维度**: 沿 B 轴（组维度）均匀切分
- **每核任务**: `ceil(totalGroups / coreNum)` 个 group
- **核数获取**: 运行时通过 `platform_ascendc::PlatformAscendC::GetCoreNumAiv()` 获取
- **负载均衡**: 除最后一个 core 外，每个 core 处理固定 `groupsPerCore` 个 group; 最后一个 core 处理剩余

### 6.2 并行特性

- **无跨核通信**: 每个 group 独立处理，无需 SyncAll / workspace 通信
- **无原子操作**: 每个 group 输出独立的 scale，不存在写竞争
- **最优并行度**: group 数远大于核数时（典型场景 B = num_tokens × D / 128，如 7 × 512/128 = 28），负载均衡良好

## 7. UB Buffer 规划

### 7.1 DAV_2201 UB 容量

| Buffer | 容量 |
|--------|------|
| UB 总量 | 192 KB |
| 可用 UB (预留对齐) | ~184 KB |
| L0C | 128 KB |
| L1 | 取决于具体配置 |

### 7.2 Buffer 列表

| Buffer 名称 | 数据类型 | 大小 (元素数) | 大小 (字节) | 用途 |
|-------------|---------|--------------|------------|------|
| `input_buf` | fp32 | `groupsPerBatch × rLengthAlign` | `groupsPerBatch × rLengthAlign × 4` | 输入数据 (bf16→fp32 转换后) + 中间计算 |
| `tmp_buf` | fp32 | `rLengthAlign × 2` | `rLengthAlign × 2 × 4` | ReduceMax 临时缓冲 (1024B 对齐) |
| `scale_buf` | fp32 | `groupsPerBatch` | `groupsPerBatch × 4` | 每组 scale 值 |
| `scale_broadcast_buf` | fp32 | `rLengthAlign` | `rLengthAlign × 4` | 单组 scale 广播结果 (可复用) |
| `x_q_buf` | int8_t | `groupsPerBatch × rLengthAlign` | `groupsPerBatch × rLengthAlign` | FP8 量化输出 |

### 7.3 groupsPerBatch 计算

以 group_size = 512, fp32 中间计算为例:

```
单组 UB 占用:
  input (fp32) + output (int8) = 512×4 + 512×1 = 2560 B
scale_buf 分摊到每组: 4 B
tmp_buf 分摊: 小到可忽略

总需求 ≈ groupsPerBatch × 2564 B + 常数开销

在 184 KB UB 约束下:
  groupsPerBatch ≈ 184 × 1024 / 2564 ≈ 73 组
```

实际取 **groupsPerBatch = 64**，留出安全边界用于 tmpBuf、scale_broadcast 和 32B 对齐。

对于 group_size = 128:
```
单组 UB ≈ 128×4 + 128×1 = 640 B
groupsPerBatch ≈ 184 × 1024 / 644 ≈ 293 → 取 256
```

### 7.4 Double Buffer 策略

对 `input_buf` 和 `x_q_buf` 采用 **Double Buffer** 流水线：

```
流水线阶段:
  Stage 1 (DMA CopyIn batch N):  MTE 搬运 GM→UB
  Stage 2 (Compute batch N-1):   Vector 引擎计算
  Stage 3 (DMA CopyOut batch N-2): MTE 搬运 UB→GM
```

使用 `EnQue` / `DeQue` 同步，3 级流水掩盖 MTE 搬运延迟。

Double Buffer 下 UB 预算需乘 2:
- `groupsPerBatch` 减半 → 实际取 32（group_size=512）或 128（group_size=128）

### 7.5 总 UB 预算 (group_size=512, Double Buffer)

| Buffer | DataType | 大小 | 字节 |
|--------|----------|------|------|
| `input_buf` × 2 | fp32 | 2 × 32 × 512 = 32768 | 128 KB |
| `tmp_buf` | fp32 | 1024 | 4 KB |
| `scale_buf` | fp32 | 32 | 128 B |
| `scale_broadcast_buf` | fp32 | 512 | 2 KB |
| `x_q_buf` × 2 | int8_t | 2 × 32 × 512 = 32768 | 32 KB |
| **总计** | | | **~166 KB** |

166 KB < 192 KB，满足 UB 约束。

## 8. 精度策略

### 8.1 精度标准

根据 `/ops-precision-standard`，本算子属于浮点计算类社区标准：
- 输入 bf16/fp16，中间计算 fp32，输出 fp8
- 精度评估：与 PyTorch 参考实现比对

### 8.2 混合精度设计

| 阶段 | 精度 | 理由 |
|------|------|------|
| 输入加载 | bf16/fp16 → fp32 | 提升中间计算精度 |
| abs + ReduceMax | fp32 | 避免半精度舍入误差累积 |
| scale 计算 | fp32 | 包含除法和条件分支 (scale_ue8m0)，需高精度 |
| 量化 (div + clamp) | fp32 | 确保截断边界精确 |
| fp8 转换 | fp32 → fp8 (bitmanip) | 软件模拟 float→e4m3 |
| 输出存储 | fp8 (int8) / fp32 | x_q: 8-bit; x_s: fp32 |

### 8.3 数值稳定性

| 保护项 | 措施 |
|--------|------|
| 除零保护 | amax clamp(eps=1e-10) 确保 scale > 0 |
| fp8 溢出 | clamp(fp8_min=-448.0, fp8_max=448.0) 防溢出 |
| scale_ue8m0 下溢 | max(\|scale\|, 1e-10) 防 log2(-inf) |
| bf16→fp32 转换 | 使用 Cast(roundMode=CAST_NONE) 精确转换 |

## 9. 特殊场景处理

### 9.1 scale_ue8m0 分支

当 `scale_ue8m0=true` 时，scale 需舍入到最接近的 2 的幂次（向正无穷取 ceil）。

**推荐实现: 位操作直接舍入**（比 Ln+Ceil+Exp 快一个数量级）:
```
float32 round_to_pow2_ceil(float32 val):
    val = |val|
    if val == 0 or is_nan: return 0
    if is_inf: return +inf
    
    uint32 bits = reinterpret_as_uint32(val)
    int32 exp = (bits >> 23) & 0xFF
    uint32 mantissa = bits & 0x7FFFFF
    
    if mantissa != 0:
        exp += 1  // ceil: 有小数部分则指数 +1
    
    if exp >= 255: return +inf  // 溢出
    if exp <= 0: return 2^-126   // 次正规数→最小正规数
    
    return reinterpret_as_float32((exp << 23))
```

> **备选方案**: 若 AscendC 提供 `Ceil` / `Log2` / `Exp2` 可直接使用，但 DAV_2201 下这些 API 可能不存在——需在实现阶段通过头文件确认。

### 9.2 FP8 float→e4m3 转换

软件实现的 float32 → fp8_e4m3fn 转换:
- 提取 float32 的符号位 (1 bit)、指数 (8 bits)、尾数 (23 bits)
- 映射到 e4m3 格式: 符号(1) + 指数(4) + 尾数(3)
- 处理特殊值: NaN → NaN, Inf → 饱和到 fp8_max, 次正规数 → 0 或最小正规数
- 舍入模式: 就近舍入偶数 (round-to-nearest-even)

> 此模块为纯位操作，不依赖 AscendC 硬件 FP8 指令。

### 9.3 非对齐边界处理

- **输入**: bf16/fp16 输入，32B 对齐使用 `DataCopy`，非对齐使用 `DataCopyPad`
- **输出 fp8**: 每个元素 1 字节，天然字节对齐，32B 对齐取决于 group_size
- **尾块处理**: 最后一轮 batch 中 actual_batch ≤ groupsPerBatch，使用 mask 控制有效元素

## 10. 分支场景覆盖

| 分支维度 | 选项 | 策略 |
|---------|------|------|
| **输入 dtype** | bf16 | 使用 Cast(bf16→fp32, CAST_NONE) |
| | fp16 | 使用 Cast(fp16→fp32，同路径) |
| **group_size** | 128 | groupsPerBatch=128 (double buf) / 256 (single buf) |
| | 512 | groupsPerBatch=32 (double buf) / 64 (single buf) |
| **scale_ue8m0** | true | 走 round_to_pow2 位操作分支 |
| | false | 跳过 scale 舍入 |
| **尾块** | actual_batch < groupsPerBatch | mask 控制有效 group 数 |
| **dtype** | fp8_e4m3fn (默认) | e4m3 编码参数: exp_bits=4, mantissa_bits=3 |
| | fp8_e5m2 (可扩展) | e5m2 编码参数: exp_bits=5, mantissa_bits=2 |

## 11. 数据流总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        Host (CPU)                                │
│  Tiling 计算:                                                     │
│    totalGroups = x.numel / group_size                             │
│    groupsPerCore = ceil(totalGroups / coreNum)                    │
│    groupsPerBatch = f(UB, group_size, doubleBuffer)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Device (AI Core, DAV_2201)                       │
│                                                                   │
│  for each batch in [0, groupsPerCore):                            │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │ Pipe: CopyIn (DMA)                                        │ │
│    │  GM[x_bf16] → UB[input_buf_fp32] (含 bf16→fp32 Cast)     │ │
│    └──────────────────────────────────────────────────────────┘ │
│                              │                                    │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │ Pipe: Reduce (Vector)                                     │ │
│    │  for g in batch: Abs → ReduceMax → Clamp(eps) → Scale    │ │
│    └──────────────────────────────────────────────────────────┘ │
│                              │                                    │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │ Pipe: Quantize (Vector)                                   │ │
│    │  Broadcast(scale) → Div → Clamp → Float2FP8              │ │
│    └──────────────────────────────────────────────────────────┘ │
│                              │                                    │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │ Pipe: CopyOut (DMA)                                       │ │
│    │  UB[x_q_int8] → GM[x_q_out]; UB[scale_fp32] → GM[x_s]   │ │
│    └──────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Double Buffer + 3 级流水:                                        │
│    V: CopyIn(N) | Reduce(N-1) | Quantize(N-2) | CopyOut(N-3)    │
│    M: CopyIn(N)                    | CopyOut(N-3)                 │
└─────────────────────────────────────────────────────────────────┘
```

## 12. 硬件约束汇总

| 约束 | 值 | 影响 |
|------|-----|------|
| UB 大小 | 192 KB | groupsPerBatch 上限 |
| repeatTimes 上限 | 255 | 不影响（group_size ≤ 512 < 255 的 32B 块数） |
| 32B 对齐 | DataCopy 要求 | rLengthAlign 取整，尾块用 DataCopyPad |
| Mask 上限 | 65535 (16-bit) | 不影响 |
| 无 FP8 Cast | DAV_2201 硬件限制 | 软件 float→fp8 转换 |
| 无 RegBase 支持 | DAV_2201 架构限制 | 使用 MemBase/SIMD API |
