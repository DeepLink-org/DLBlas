# sparse_attn 算子技术设计文档 (DESIGN.md)

## 1. 算子定义与数学描述

### 1.1 算子名称

`sparse_attn` — 稀疏注意力算子，来自 DeepSeek-V4-Pro 推理内核。

### 1.2 数学公式

```
输入: q [b, m, h, d] bf16      — query
      kv [b, n, d] bf16         — shared key-value per position
      attn_sink [h] fp32        — learnable per-head sink bias
      topk_idxs [b, m, topk] i32 — sparse attention indices, -1=padding
      softmax_scale float        — typically d ** -0.5

Step 1: Gather KV
  valid_mask = topk_idxs >= 0                          # [b, m, topk] bool
  safe_idxs  = clamp(topk_idxs, min=0)                 # [b, m, topk] i32
  gathered_kv[b,m,t,d] = kv[b, safe_idxs[b,m,t], d]   # [b, m, topk, d] bf16
  gathered_kv *= valid_mask.unsqueeze(-1)              # zero invalid

Step 2: Attention Scores
  scores[b,m,h,t] = sum_d( q[b,m,h,d] * gathered_kv[b,m,t,d] ) * softmax_scale
  scores[b,m,h,t] = -inf  where !valid_mask[b,m,t]     # mask invalid

Step 3: Softmax with Attention Sink
  sink[h] = attn_sink[h]                               # broadcast [h] -> [b,m,h,1]
  max_scores[b,m,h] = max( max_t(scores[b,m,h,:]), sink[h] )
  exp_scores[b,m,h,t] = exp( scores[b,m,h,t] - max_scores[b,m,h] )
  exp_scores[b,m,h,t] = 0  where !valid_mask[b,m,t]    # re-zero invalid
  exp_sink[b,m,h] = exp( sink[h] - max_scores[b,m,h] )
  sum_exp[b,m,h] = sum_t( exp_scores[b,m,h,:] ) + exp_sink[b,m,h]
  attn_weights[b,m,h,t] = exp_scores[b,m,h,t] / sum_exp[b,m,h]

Step 4: Weighted Sum
  output[b,m,h,d] = sum_t( attn_weights[b,m,h,t] * gathered_kv[b,m,t,d] )
  output → bf16
```

### 1.3 输入输出规格

| 张量 | Shape | dtype | 说明 |
|------|-------|-------|------|
| q | [b, m, h, d] | bfloat16 | query, BSND layout |
| kv | [b, n, d] | bfloat16 | shared KV, BND layout |
| attn_sink | [h] | float32 | 模型参数，Host 侧传入 |
| topk_idxs | [b, m, topk] | int32 | -1 表示无效/padding 位置 |
| softmax_scale | scalar | float | d ** -0.5，Host 侧传入 |
| **output** | [b, m, h, d] | bfloat16 | 注意力输出 |

### 1.4 默认配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| b (batch) | 2 | batch size |
| m (seq_len) | 16 | query sequence length |
| n (kv_len) | 32 | KV sequence length |
| h (n_heads) | 8 | number of heads |
| d (head_dim) | 64 | head dimension |
| topk | 16 | sparse attention window size |

### 1.5 算子类型判定

本算子属于**稀疏注意力 (Sparse Attention)** 类。与标准 FlashAttention 的关键差异：

| 特性 | FlashAttention | sparse_attn |
|------|---------------|-------------|
| KV 访问模式 | 全量，沿 Sk 分块顺序 | 稀疏，基于 topk_idxs 随机索引 |
| Score 矩阵 | 不显式实例化，分块流式 | 完整实例化 [h, topk]，topk 可控 |
| 跨 KV 分块状态 | 需要 online rescale | 无（单次全量计算） |
| Attention Sink | 无 | 有（仅参与分母） |

**结论**：不走 FA 类方法论（无 KV chunking、无 online softmax、无 AIC/AIV 协同需求）。归类为 **混合算子（Gather + Reduction + Elementwise + Matmul-like）**，采用通用 SIMD/MemBase 路线。

---

## 2. 方案决策

### 2.1 架构信息

| 项目 | 值 | 来源 |
|------|-----|------|
| 芯片型号 | Ascend910B2 | 用户需求 |
| NpuArch | DAV_2201 | npu-arch skill 查表 |
| UB 容量 | 192 KB (196608 B) | npu-arch skill |
| CANN 版本 | 9.0.0 | 用户需求 |
| Vector 核数 | 20 (AIV, 运行时查询) | PlatformAscendC::GetCoreNumAiv() |

### 2.2 技术路线决策

```
算子类型：稀疏注意力（Gather + Reduction + Elementwise）
    ↓
架构 = DAV_2201？
    ├─ 是 → 排除 RegBase（仅 DAV_3510）
    │       排除 Blaze/tensor_api（仅 DAV_3510）
    │       进入通用 SIMD/MemBase 路线
    └─ 否 → DAV_3510 分支（当前不适用）

进一步决策：Cube vs 纯 Vector？
    关键问题：两个 einsum 操作使用 Cube(MatmulImpl) 还是 Vector(逐元素计算)
    
    Cube 方案成本：
    - 需要 AIC+AIV 混合 kernel（__mix__），两次独立 MatmulImpl 调用
    - 中间数据 (gathered_kv, scores, attn_weights) → GM workspace 多轮搬运
    - Gather 本身在 AIV 上，与其他 stage 的 AIC/AIV 握手同步复杂
    - 小 shape (h=8, d=64, topk=16) 下 Cube 发射开销 >> 计算收益
    
    Vector 方案收益：
    - 纯 Vector kernel，单核类型，无 AIC/AIV 协调
    - 全量中间数据常驻 UB（topk 可控确保可行）
    - 零 GM workspace 需求
    - 实现路径清晰：Gather → Matmul-like → Softmax → Matmul-like → Output
    
    决策：纯 Vector SIMD 路线（通用 MemBase）
```

### 2.3 最终决策表

| 决策维度 | 选择 | 理由 |
|----------|------|------|
| **路线** | SIMD/MemBase (通用) | DAV_2201，无 RegBase/Blaze |
| **Kernel 类型** | 纯 Vector (AIV only) | 避免 AIC/AIV 协调复杂度；小 shape 下 Cube 开销 > 收益 |
| **算子调用方式** | Kernel 直调 | 自定义算子，手工 Tiling |
| **并行策略** | 沿 (b, m) 切分，无 cross-core 状态 | 各 query position 独立计算，天然并行 |
| **精度路线** | BF16 输入/输出，FP32 内部计算 | 社区标准：MERE < 2^-7 (bf16 threshold) |
| **Gather 策略** | 逐元素 DataCopy，基于 computed offset | 简单正确；topk 小故开销可控 |

---

## 3. 多核切分策略

### 3.1 任务维度构造

```
totalTasks = b × m                    # 总任务数 = batch × seq_len
usedCoreNum = min(aivNum, totalTasks)  # 运行时查询，禁止硬编码
```

单个 task 对应一个 (batch_idx, seq_idx) 对，处理该位置的 q[m, h, d]、topk_idxs[m, topk]，产生 output[m, h, d]。

**任务分配**：
```
tasksPerCore = CeilDiv(totalTasks, usedCoreNum)
core_i 处理 taskIdx ∈ [i * tasksPerCore, min((i+1) * tasksPerCore, totalTasks))
```

每个 core 处理的 task 数量之差 ≤ 1，负载均衡。

### 3.2 不变量校核

| 不变量 | 适用性 | 状态 |
|--------|--------|------|
| I1 (数据流方向) | FA 类特有 | N/A — 不走 FA 路线 |
| I2 (GQA 同 kvHead) | FA 类特有 | N/A — 本算子无 GQA 概念 |
| I3 (跨 task 隔离) | **适用** | 满足 — 各 task 独立，无共享状态 |
| I4 (s2 状态累积) | FA 类特有 | N/A — 无 KV 分块，无 online softmax |

### 3.3 核数获取

```cpp
// Host 侧 Tiling 函数
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
uint32_t usedCoreNum = min(aivNum, totalTasks);  // totalTasks = b * m
```

空闲核 (coreIdx >= usedCoreNum) 在 kernel 入口直接 return。

---

## 4. UB 切分与 Buffer 规划

### 4.1 UB 容量约束

| 参数 | 值 |
|------|-----|
| UB 总容量 | 192 KB = 196608 B |
| 安全系数 | 0.85 (预留 pipe overhead, 临时空间) |
| 可用 UB | ~163 KB = 167116 B |

### 4.2 tile_m 计算

单 tile 内处理 `tile_m` 个 query position，所有 h 个 head 和所有 topk 个 KV position 一并处理。

**Buffer 清单**（按生命周期阶段峰值）：

**Stage A（加载阶段峰值）**：
| Buffer | 元素数 | dtype | 大小 (B) | 说明 |
|--------|--------|-------|----------|------|
| q_buf | tile_m × h × d | bf16 | tile_m × h × d × 2 | Q 输入 |
| idx_buf | tile_m × topk | int32 | tile_m × topk × 4 | topk_idxs |
| gkv_buf | tile_m × topk × d | bf16 | tile_m × topk × d × 2 | gathered KV |

Stage A 总计 = tile_m × (2hd + 4×topk + 2×topk×d)

**Stage B（计算阶段峰值，fp32 中间量）**：
| Buffer | 元素数 | dtype | 大小 (B) | 说明 |
|--------|--------|-------|----------|------|
| q_fp32 | tile_m × h × d | fp32 | tile_m × h × d × 4 | Q cast to fp32 |
| gkv_fp32 | tile_m × topk × d | fp32 | tile_m × topk × d × 4 | KV cast to fp32 |
| score_buf | tile_m × h × topk | fp32 | tile_m × h × topk × 4 | scores/exp/attn 复用 |
| max_buf | tile_m × h | fp32 | tile_m × h × 4 | max scores |
| sum_buf | tile_m × h | fp32 | tile_m × h × 4 | sum exp |
| tmp_buf | (reduction 临时) | — | ~4 KB | ReduceMax/ReduceSum 临时空间 |

Stage B 总计 ≈ tile_m × (4hd + 4×topk×d + 4h×topk + 8h) + 4096

**Buffer 复用策略**：
- q_buf (bf16) → 转换为 q_fp32 后释放 q_buf，复用该空间
- gkv_buf (bf16) → 转换为 gkv_fp32 后释放 gkv_buf
- gkv_fp32 → 计算完 output 后可复用给 output (二者 shape 不同但大小 ≤)
- score_buf → 依次复用为 exp_scores、attn_weights（同 shape）

最终 Stage B 峰值 = tile_m × (4hd + 4h×topk + 4×topk×d + 8h) + 4096

### 4.3 默认配置下的 tile_m 上限

取 h=8, d=64, topk=16：

```
Stage B = tile_m × (4×8×64 + 4×8×16 + 4×16×64 + 8×8) + 4096
        = tile_m × (2048 + 512 + 4096 + 64) + 4096
        = tile_m × 6720 + 4096

求解: tile_m × 6720 + 4096 ≤ 167116
      tile_m ≤ (167116 - 4096) / 6720 ≈ 24.26

取 tile_m_max = 16 (留安全余量)
```

### 4.4 通用 tile_m 公式

```
tile_m = min(totalTasksPerCore, 16)  // 安全上限
```

tile_m ≤ 16 确保在各种 h/d/topk 组合下 UB 不溢出。当 h/d 更小时可适当增大（由开发者根据实际 Buffer 大小公式运行时计算）。

### 4.5 Double Buffer 策略

本算子采用 **Single Buffer + 串行计算** 策略，不使用 Double Buffer / 流水线。理由：
- 计算流严格串行（Gather → Matmul1 → Softmax → Matmul2），无可重叠的独立阶段
- 单 tile 数据量小，计算延迟以 Gather (memory-bound) 为主，流水线收益有限
- 简化实现，降低同步 bug 风险

---

## 5. 数据流与计算流水线

### 5.1 总体数据流

```
GM ──(DataCopy)──> UB: q_tile, idx_tile
                    │
                    ├── Gather KV (逐元素 DataCopy) ──> gkv_tile
                    │
                    ├── Cast bf16→fp32 (q_fp32, gkv_fp32)
                    │
                    ├── Matmul-like: scores[i,h,t] = Σ_d q[i,h,d] * gkv[i,t,d] * softmax_scale
                    │
                    ├── Mask (valid_mask): -inf for invalid positions
                    │       │
                    ├── ReduceMax: max(scores, dim=-1) → max_scores[i,h]
                    │       │       max(max_scores, attn_sink) → max_scores[i,h]
                    ├── Exp: exp(scores - max_scores) → exp_scores; mask invalid → 0
                    ├── Exp: exp(attn_sink - max_scores) → exp_sink
                    ├── ReduceSum: sum(exp_scores, dim=-1) → sum_exp
                    ├── Add: sum_exp + exp_sink → sum_exp
                    ├── Div: exp_scores / sum_exp → attn_weights
                    │
                    ├── Matmul-like: output[i,h,d] = Σ_t attn_weights[i,h,t] * gkv[i,t,d]
                    │
                    ├── Cast fp32→bf16
                    │
GM <──(DataCopy)─── UB: output_tile
```

### 5.2 阶段 1：Gather KV

```
For i in [0, tile_m):
  For k in [0, topk):
    idx = max(idx_buf[i, k], 0)  # safe index
    valid = (idx_buf[i, k] >= 0)
    offset = (batch_idx * n + idx) * d * sizeof(bf16)
    DataCopy(gkv_buf[i, k, :], kv_gm[offset:offset+d*sizeof(bf16)])
    if !valid:
      Zero(gkv_buf[i, k, :])  # mask invalid positions
```

**关键点**：
- 使用 `DataCopyPad` (优先) 或 `DataCopy` (对齐时) 单次搬运 d 个 bf16 元素
- d ≤ 64 时单次 DataCopy ≤ 128 B，硬件可一次完成
- 无效位置 (idx=-1 被 clamp 到 0 后) 在 gather 后显式置零

### 5.3 阶段 2：Attention Scores (Matmul-like)

两个多级循环实现 batched matmul：

**方法**：q [tile_m, h, d] × gkv^T [tile_m, topk, d] → scores [tile_m, h, topk]

```
For i in [0, tile_m):
  For hh in [0, h):
    For t in [0, topk):
      acc = 0
      For dd in [0, d):           # 内积循环
        acc += q_fp32[i, hh, dd] * gkv_fp32[i, t, dd]
      scores[i, hh, t] = acc * softmax_scale
```

**优化**：使用 Ascend C Vector API 的 `Mul` + `ReduceSum` 批量处理：
```
For i in [0, tile_m):
  For hh in [0, h):
    // tmp[i, t, dd] = q_fp32[i, hh, dd] * gkv_fp32[i, t, dd]  (broadcast q over topk)
    Mul(tmp_fp32, q_fp32_broadcast, gkv_fp32)
    // scores[i, hh, :] = ReduceSum(tmp_fp32, dim=-1) * softmax_scale
    ReduceSum(scores[i, hh, :], tmp_fp32, Pattern::Reduce::AR)
```

### 5.4 阶段 3：Softmax with Sink

```
For i in [0, tile_m):
  // Step 3a: Mask invalid scores to -inf
  For hh in [0, h):
    For t in [0, topk):
      if !valid_mask[i, t]:
        scores[i, hh, t] = -inf

  // Step 3b: Channel max
  For hh in [0, h):
    max_scores[i, hh] = ReduceMax(scores[i, hh, :], dim=-1)
    max_scores[i, hh] = max(max_scores[i, hh], attn_sink[hh])

  // Step 3c: exp(scores - max_scores), re-zero invalid
  For hh in [0, h):
    Sub(scores[i, hh, :], max_scores[i, hh])  // broadcast subtract
    Exp(exp_buf[i, hh, :], scores[i, hh, :])
    For t in [0, topk):
      if !valid_mask[i, t]:
        exp_buf[i, hh, t] = 0.0f

  // Step 3d: exp_sink
  For hh in [0, h):
    exp_sink_val[hh] = exp(attn_sink[hh] - max_scores[i, hh])

  // Step 3e: normalize
  For hh in [0, h):
    sum_exp[i, hh] = ReduceSum(exp_buf[i, hh, :], dim=-1)
    sum_exp[i, hh] += exp_sink_val[hh]
    attn_weights[i, hh, :] = exp_buf[i, hh, :] / sum_exp[i, hh]
```

### 5.5 阶段 4：Weighted Sum (Matmul-like)

```
For i in [0, tile_m):
  For hh in [0, h):
    For dd in [0, d):
      acc = 0
      For t in [0, topk):
        acc += attn_weights[i, hh, t] * gkv_fp32[i, t, dd]
      output[i, hh, dd] = acc
```

**优化**：使用 `Mul` + `ReduceSum` 批量处理：
```
For i in [0, tile_m):
  For hh in [0, h):
    // tmp[i, t, dd] = attn_weights[i, hh, t] * gkv_fp32[i, t, dd]
    Mul(tmp_fp32, attn_broadcast, gkv_fp32)
    // output[i, hh, :] = ReduceSum(tmp_fp32, dim=1)  (沿 topk 维度归约)
    ReduceSum(output[i, hh, :], tmp_fp32, Pattern::Reduce::RA)
```

### 5.6 阶段 5：写出

```
Cast output bf16
DataCopyPad(output_gm[dest_offset], output_bf16)
```

---

## 6. API 映射表

### 6.1 数据搬运

| 用途 | API | 参数/模式 |
|------|-----|----------|
| 加载 Q (连续) | `DataCopyPad` / `DataCopy` | blockLen = h×d×sizeof(bf16)，1 个 block |
| 加载 topk_idxs | `DataCopyPad` | blockLen = topk×sizeof(int32)，1 个 block |
| 加载 attn_sink | `DataCopyPad` | blockLen = h×sizeof(float)，Host→GM→UB |
| Gather 单行 KV | `DataCopyPad` | 逐 (i,k): blockLen = d×sizeof(bf16), offset = computed |
| 写出 output | `DataCopyPad` | blockLen = h×d×sizeof(bf16) |

### 6.2 精度转换

| 用途 | API | 参数 |
|------|-----|------|
| bf16 → fp32 | `Cast<float, half>` (bf16 在 Ascend C 中映射为 half/uint16_t) | 整张 tensor 转换 |
| fp32 → bf16 | `Cast<half, float>` | 写出前转换 |

> 注意：Ascend C 中 bfloat16 使用 `half` / `uint16_t` 类型表示（同为 2 字节），Cast 需配合 `uint16_t` ↔ `float` 或使用 bf16 专用 intrinsic。具体 API 签名以 CANN 9.0.0 文档中 `Cast` 的 bf16 支持确认。

### 6.3 算术运算

| 用途 | API | 说明 |
|------|-----|------|
| 内积 (dot product) | `Mul` + `ReduceSum` | 分两步：逐元素乘 + 沿 d 维归约 |
| 加权求和 | `Mul` + `ReduceSum` | 分两步：逐元素乘 + 沿 topk 维归约 |
| 广播减法 (scores - max) | `Sub` + `Broadcast` | 使用 `Broadcast` 模式广播标量到向量 |
| 除法归一化 | `Div` | exp_scores / sum_exp, broadcast sum_exp |
| Mask 零值填充 | `Mul` (乘以 valid_mask) 或 `Select` | Select 方式更高效 |
| Softmax scale 乘法 | `Muls` | scores * softmax_scale, 标量乘 |

### 6.4 归约操作

| 用途 | API | Pattern |
|------|-----|---------|
| 沿 topk 归约 max | `ReduceMax` | `Pattern::Reduce::AR` |
| 沿 topk 归约 sum | `ReduceSum` | `Pattern::Reduce::AR` |
| 沿 topk 归约 sum (加权求和) | `ReduceSum` | `Pattern::Reduce::RA` |

**Reduce 临时空间**：使用 `GetReduceMaxMaxMinTmpSize` / `GetReduceSumMaxMinTmpSize` 查询，选择 maxSize。

### 6.5 特殊函数

| 用途 | API | 说明 |
|------|-----|------|
| 指数函数 | `Exp` | fp32 精度，对 scores 逐元素 |
| 条件选择 | `Select` | 根据 valid_mask 选择填充值 |
| 最大值 | `Max` | max(max_scores, attn_sink) |
| 绝对值 (mask 构建) | `Abs` | 可选：topk_idxs >= 0 判断 |

### 6.6 禁止使用的 API

| API | 原因 | 替代 |
|-----|------|------|
| `GlobalTensor::SetValue()` | 效率极低 | `DataCopyPad` |
| `GlobalTensor::GetValue()` | 效率极低 | `DataCopyPad` |

---

## 7. 精度策略

### 7.1 精度标准

按 `/ops-precision-standard` 判定路径：
- 输入输出均为浮点 → 浮点计算类社区标准
- BF16 输出 → **MERE < 2^-7 ≈ 0.00781**, **MARE < 10 × 2^-7 ≈ 0.0781**

### 7.2 混合精度方案

| 阶段 | 精度 | 说明 |
|------|------|------|
| Q, KV 加载 | bf16 → fp32 | 加载后立即 Cast 到 fp32 |
| Gather KV | bf16 (加载) → fp32 (计算) | Gather 本身搬运 bf16，用于计算前 Cast |
| Attention Scores | fp32 | 全 fp32 内积累加法 |
| Softmax (Exp, Reduce) | fp32 | fp32 确保数值稳定性 |
| Weighted Sum | fp32 | 全 fp32 累积 |
| Output 写出 | fp32 → bf16 | 最终 Cast 后写出 |

### 7.3 数值稳定性保护

| 保护措施 | 场景 | 说明 |
|----------|------|------|
| max subtraction | Softmax 指数前 | 标准 softmax 稳定化：`exp(x - max)` |
| fp32 中间精度 | 全流程 | 避免 bf16 累加精度损失 |
| valid_mask re-zero | exp 后无效位置 | 防止 exp(-inf) 微小非零值污染分母 |
| epsilon in division | div 前 sum_exp | 建议 `max(sum_exp, 1e-10)` 防止除零 |

### 7.4 边界值处理

| 场景 | 行为 | 说明 |
|------|------|------|
| topk_idxs 全为 -1 | output 全零 | valid_mask 全 false，gathered_kv 被置零，softmax 后 attn_weights 全零 |
| attn_sink 极大 | 所有权重趋近 0 | exp_sink 主导分母，符合数学定义 |
| softmax_scale 极大 | 可能溢出 | fp32 提供足够动态范围 |
| d=1 | 退化为一维点积 | 正常处理 |

---

## 8. 特殊场景与边界处理

### 8.1 Shape 边界

| 场景 | 处理 |
|------|------|
| b × m < usedCoreNum | 部分核空闲（入口跳过），task 负载均衡 |
| tile_m 尾块 < tile_m | 最后一个 iteration 用实际剩余 task 数，DataCopy 按实际大小 |
| h 或 d 非对齐 | DataCopyPad 自动处理非对齐（优先使用） |
| topk=0 | 退化：所有位置无效，输出全零 |

### 8.2 Gather 优化建议（路标）

当前方案采用逐元素 DataCopy 实现 KV gather。对于 topk 较大、重复索引较多的场景，可行优化：
1. **排序聚合**：对 tile 内的 topk_idxs 排序，相同 KV row 合并为一次 DataCopy → scatter 到多个 target
2. **预取**：利用 L2 cache 特性，相邻的 KV rows 被 Cache 加速
3. 此优化不在 MVP 范围内，标记为 Phase 4（性能优化）候选

### 8.3 分支决策

| 条件 | 策略 |
|------|------|
| topk ≤ 0 | 无有效 attention，直接输出零张量 |
| b × m == 1 | 单 task，usedCoreNum = 1 |
| d % 8 != 0 | DataCopyPad 处理非对齐 |
| 通用 | 标准 tile_m=16 路径，尾块自适应 |

---

## 9. Host Tiling 数据结构

### 9.1 Tiling 参数

```cpp
struct SparseAttnTiling {
    // 输入 shape
    uint32_t b, m, n, h, d, topk;

    // 多核切分
    uint32_t totalTasks;     // = b * m
    uint32_t usedCoreNum;    // = min(aivNum, totalTasks)

    // UB 切分
    uint32_t tile_m;         // 单次处理 query position 数

    // 超参数
    float    softmax_scale;

    // attn_sink 已在 GM 中, kernel 自行加载
};
```

### 9.2 Tiling 计算函数

```cpp
SparseAttnTiling ComputeTiling(const SparseAttnParams& params,
                                platform_ascendc::PlatformAscendC& platform) {
    SparseAttnTiling tiling;
    tiling.b = params.b; tiling.m = params.m;
    tiling.n = params.n; tiling.h = params.h;
    tiling.d = params.d; tiling.topk = params.topk;

    tiling.totalTasks = params.b * params.m;

    uint32_t aivNum = platform.GetCoreNumAiv();
    tiling.usedCoreNum = std::min(aivNum, tiling.totalTasks);

    // tile_m 上限计算（根据 §4.3 公式）
    uint32_t ub_avail = platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB) * 85 / 100;
    uint32_t per_task_ub = 4 * params.h * params.d + 4 * params.h * params.topk
                         + 4 * params.topk * params.d + 8 * params.h;
    tiling.tile_m = std::min(
        (ub_avail - 4096) / per_task_ub,
        static_cast<uint32_t>(16)  // 安全上限
    );
    if (tiling.tile_m < 1) tiling.tile_m = 1;

    tiling.softmax_scale = params.softmax_scale;
    return tiling;
}
```

---

## 10. 开发约束

| # | 约束 | 说明 |
|---|------|------|
| C1 | 纯 Vector kernel | 不涉及 AIC/Cube，不用 __mix__ |
| C2 | 禁止硬编码核数 | 必须通过 PlatformAscendC 运行时查询 |
| C3 | 禁止 SetValue/GetValue | 生产代码禁止使用，调试除外 |
| C4 | DataCopyPad 优先 | 非对齐数据优先使用 DataCopyPad |
| C5 | 无 Host 侧预处理 | 不转置、不重排输入 tensor |
| C6 | 单 kernel 文件 | 所有逻辑在一个 .cpp kernel 文件中 |
| C7 | FP32 中间精度 | 关键计算路径（Matmul-like, Softmax）使用 fp32 |

---

## 11. 参考资料

| 资料 | 用途 |
|------|------|
| `/npu-arch` skill | 架构参数 (UB=192KB, DAV_2201, aivNum=20) |
| `/ascendc-tiling-design` skill | FA 类设计方法论（参考不变量、切分策略） |
| `/ascendc-api-best-practices` skill | API 用法 (DataCopyPad, ReduceMax/Sum, Cast, Exp) |
| `/ops-precision-standard` skill | BF16 社区标准 (MERE < 2^-7) |
| `$ASC_DEVKIT_DIR/examples/` | 参考实现 (vector_add, sub, addn 等) |
