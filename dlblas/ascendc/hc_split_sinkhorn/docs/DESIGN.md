# hc_split_sinkhorn 算子设计方案

> **状态**: Final（实现已验证，精度达标）
> **设计者**: Architect Agent
> **目标架构**: Ascend910B2, NpuArch=DAV_2201, __NPU_ARCH__=2201, --npu-arch=dav-2201
> **CANN 版本**: 9.0.0
> **算子类别**: 混合型 Vector 算子（Elementwise + Reduction）
> **技术路线**: SIMD/MemBase（标准 AscendC Vector API）

---

## 1. 算子概述

### 1.1 功能描述

`hc_split_sinkhorn` 算子将输入张量 `mixes` 沿 `mix_hc` 维度拆分为三个分量，分别执行：

1. **Pre 分量**: sigmoid 变换 `sigmoid(x * s0 + bias_pre) + eps`
2. **Post 分量**: sigmoid 变换 `2 * sigmoid(x * s1 + bias_post)`
3. **Comb 分量**: Sinkhorn 迭代双随机归一化（exp 稳定化 + 行列交替归一化）

### 1.2 输入输出规格

| 参数 | 形状 | 数据类型 | 说明 |
|------|------|---------|------|
| `mixes` | `(B, S, mix_hc)` | float32 | 混合输入，`mix_hc = (2 + hc) * hc` |
| `hc_scale` | `(3,)` | float32 | 缩放因子 `[s0, s1, s2]` |
| `hc_base` | `(mix_hc,)` | float32 | 偏置向量 |
| `hc_mult` | scalar int | — | hc 维度大小，默认 4 |
| `sinkhorn_iters` | scalar int | — | Sinkhorn 迭代次数，默认 20 |
| `eps` | scalar float | — | 数值稳定常数，默认 1e-6 |

| 输出 | 形状 | 数据类型 | 说明 |
|------|------|---------|------|
| `pre` | `(B, S, hc)` | float32 | sigmoid 变换后的 pre 分量 |
| `post` | `(B, S, hc)` | float32 | sigmoid 变换后的 post 分量 |
| `comb` | `(B, S, hc, hc)` | float32 | Sinkhorn 归一化后的双随机矩阵 |

### 1.3 关键约束

| 约束 | 值 | 说明 |
|------|-----|------|
| hc 范围 | `hc ≤ 32` | 编译期硬限制（Tiling 结构体数组大小） |
| mix_hc 公式 | `mix_hc = (2 + hc) * hc` | 由 hc 导出 |
| 数据类型 | float32 全链路 | 无精度降级 |
| 对齐 | 内部对齐 (hcAlign/mixHcAlign)，输出紧凑 | Reduce API 约束 |

---

## 2. 数学定义

### 2.1 数据展平与拆分

```
输入: mixes(b, s, mix_hc), hc_scale(3), hc_base(mix_hc)
B = b * s
x = mixes.reshape(B, mix_hc)              // (B, mix_hc), mix_hc = (2+hc) * hc

pre_raw  = x[:, 0:hc]                     // (B, hc)
post_raw = x[:, hc:2*hc]                  // (B, hc)
comb_raw = x[:, 2*hc:2*hc+hc*hc]          // (B, hc*hc)
```

### 2.2 Pre 分量

```
pre = sigmoid(pre_raw * s0 + base[0:hc]) + eps
    = 1.0 / (1.0 + exp(-(pre_raw * s0 + base[0:hc]))) + eps
```

### 2.3 Post 分量

```
post = 2 * sigmoid(post_raw * s1 + base[hc:2*hc])
     = 2.0 / (1.0 + exp(-(post_raw * s1 + base[hc:2*hc])))
```

### 2.4 Comb 分量 -- Sinkhorn 双随机归一化

```
comb = comb_raw.reshape(B, hc, hc) * s2 + base[2*hc:2*hc+hc*hc].reshape(1, hc, hc)

# 第 0 次迭代 (含 exp 数值稳定化)
for each row r in comb[r, :]:
    max_val = max(comb[r, :])
    comb[r, :] = exp(comb[r, :] - max_val)          # 数值稳定
for each row r in comb[r, :]:
    row_sum = sum(comb[r, :])
    comb[r, :] = comb[r, :] / row_sum + eps          # 行归一化 (+eps 初值)
for each col c in comb[:, c]:
    col_sum = sum(comb[:, c])
    comb[:, c] = comb[:, c] / (col_sum + eps)        # 列归一化

# 第 1..sinkhorn_iters-1 次迭代 (无 exp)
for iter in range(1, sinkhorn_iters):
    for each row r:
        row_sum = sum(comb[r, :])
        comb[r, :] = comb[r, :] / (row_sum + eps)
    for each col c:
        col_sum = sum(comb[:, c])
        comb[:, c] = comb[:, c] / (col_sum + eps)
```

### 2.5 输出整形

```
pre  → reshape(B, hc)    → (b, s, hc)
post → reshape(B, hc)    → (b, s, hc)
comb → reshape(B, hc, hc) → (b, s, hc, hc)
```

> 注: reshape 是无副本的视图变换。Kernel 输入已将 mixes 展平为 (B, mix_hc)；输出 pre/post/comb 在 GM 中按 (B, hc) / (B, hc*hc) 布局。宿主侧 torch 扩展负责 reshape 回目标形状。

---

## 3. 方案决策

### 3.1 架构信息

| 项目 | 值 | 获取方式 |
|------|-----|---------|
| 芯片型号 | Ascend910B2 | 环境给定 |
| NpuArch | `DAV_2201` | `/npu-arch` skill 查表 |
| `__NPU_ARCH__` | `2201` | `/npu-arch` skill |
| `--npu-arch` 编译参数 | `dav-2201` (vec) | 环境指定 |
| CANN 版本 | 9.0.0 | 环境给定 |
| UB 容量 | 192 KB | DAV_2201 规格 |
| L0C 容量 | 128 KB | DAV_2201 规格 |
| Vector Core 数 | 24 | Ascend910B2 典型值 |

### 3.2 算子类型判定

```
算子计算特征分析:
  - sigmoid:        Exp + Adds + Div 逐元素运算 → Elementwise
  - ReduceMax(row): 沿最后一维 (hc) 归约 → Reduction (AR)
  - ReduceSum(row): 沿最后一维 (hc) 归约 → Reduction (AR)
  - ReduceSum(col): 沿 hc×hc 矩阵列归约 → Reduction (RA/手动)
  - 无 Cube/MatMul 计算 → 纯 Vector 算子
  - 输入到输出不改变 Shape → 非 Conversion

判定: 混合型 Vector 算子（Elementwise + Reduction）
```

### 3.3 技术路线决策

```
架构判断:
  - NpuArch = DAV_2201（非 DAV_3510）
  - RegBase 路线仅限 DAV_3510 → 不适用
  - Blaze 路线仅限 DAV_3510 MatMul → 不适用

结论: 通用 SIMD/MemBase 路线
  - 使用标准 AscendC Vector API（Level 0/1/2）
  - 数据搬运使用 DataCopyPad（非对齐安全）
  - 归约使用 Level 2 Reduce 接口（逐行处理）
  - 无跨核依赖 → 无需 Group Reduce
```

### 3.4 设计方法论来源

| 设计要素 | 来源 | 关键参考 |
|---------|------|---------|
| 多核切分 + UB 切分框架 | `/ascendc-tiling-design` 通用设计要素 | §4, §5 |
| 归约 AR 模式（行内 max/sum） | `/ascendc-tiling-design` → Reduction patterns | AR-FullLoad 分支 |
| 逐元素 tiling 策略 | `/ascendc-tiling-design` → Elewise patterns | 1D 线性展平 |
| 标量广播优化（Adds/Muls） | `/ascendc-api-best-practices` → api-arithmetic.md | 场景1: 标量操作 |
| DataCopyPad 非对齐安全搬运 | `/ascendc-api-best-practices` → api-datacopy.md | 输入/输出搬运 |
| 归约 API 类型约束 | `/ascendc-api-best-practices` → api-reduce.md | tmpBuffer 类型匹配 |

---

## 4. 多核切分策略

### 4.1 切分维度与算法

沿 batch 维度 `B = b * s` 切分。每个 AI Core 处理连续的若干行，各行之间完全独立，无跨核依赖。

```
每核行数（上取整）: rowsPerCore = ceil(B / coreNum)
实际用核数:         usedCoreNum = ceil(B / rowsPerCore)
尾核行数:           tailCoreRows = B - (usedCoreNum - 1) * rowsPerCore
```

### 4.2 切分示意

```
输入: mixes[B, mix_hc] 展平为 2D
切分: 每核处理 rowsPerCore 行

Core 0: rows [0, rowsPerCore)
Core 1: rows [rowsPerCore, 2*rowsPerCore)
...
Core k: rows [k*rowsPerCore, min((k+1)*rowsPerCore, B))
```

### 4.3 负载均衡分析

| 考量 | 分析 |
|------|------|
| **负载均衡** | B 较大时各核负载接近相等；尾核最多差 rowsPerCore-1 行 |
| **数据局部性** | 每行 mix_hc 个连续 float（最多 24*4=96B），GM 读取模式友好 |
| **无跨核依赖** | 每行计算完全独立，无需 Group Reduce 或跨核同步 |
| **粒度控制** | 最小粒度为 1 行，上取整除法的 wasted cores 最多 coreNum-1 |

---

## 5. UB 切分策略

### 5.1 切分方式

沿 batch 维度在单核内二次切分：每次处理 tileRows (T) 行，循环处理直到该核的所有行处理完毕。

计算分两阶段：
- **阶段 A**: Pre/Post sigmoid 计算
- **阶段 B**: Comb Sinkhorn 迭代

两阶段串行执行，单次 tile 处理中两阶段共用输入缓冲。UB 峰值需求取 max(阶段A 并发, 阶段B 并发)。

### 5.2 Buffer 清单与容量分析 (hc=4, mix_hc=24 示例)

| Buffer Name | 阶段 | 每行元素数 | 元素类型 | 公式 (T=512) | 字节 |
|-------------|------|-----------|---------|--------------|------|
| `inQueueMixes_` | A+B | mixHcAlign | float | T * 24 = 12288 | 48 KB |
| `outQueuePre_` | A | hcAlign | float | T * 8 = 4096 | 16 KB |
| `outQueuePost_` | A | hcAlign | float | T * 8 = 4096 | 16 KB |
| `workBufComb_` | B | hc * hcAlign | float | T * 32 = 16384 | 64 KB |
| `workBufFlatPre_` | A | hc | float | T * 4 = 2048 | 8 KB |
| `workBufFlatPost_` | A | hc | float | T * 4 = 2048 | 8 KB |
| `workBufFlatComb_` | B | hc * hc | float | T * 16 = 8192 | 32 KB |
| `tmpBufRowCol_` | A+B | hcAlign | float | T * 8 = 4096 | 16 KB |
| `tmpBufCompute_` | A+B | — | float | 32 | 128 B |
| `tmpBufParams_` | A+B | — | float | ~300 | ~1.2 KB |
| `tmpBufReduce_` | A+B | — | float | 1024 | 4 KB |

### 5.3 UB 容量约束 (hc=4, T=512)

```
阶段 A 并发峰值:
  inQueueMixes_(48) + outQueuePre_(16) + outQueuePost_(16)
  + workBufFlatPre_(8) + workBufFlatPost_(8) + tmpBufRowCol_(16)
  + tmpBufCompute_(0.125) + tmpBufParams_(1.2) + tmpBufReduce_(4)
  = 117.3 KB < 192 KB  OK

阶段 B 并发峰值:
  inQueueMixes_(48) + workBufComb_(64) + workBufFlatComb_(32)
  + tmpBufRowCol_(16) + tmpBufCompute_(0.125) + tmpBufParams_(1.2)
  + tmpBufReduce_(4)
  = 165.3 KB < 192 KB  OK
```

### 5.4 动态 tileRows 计算公式

```cpp
constexpr uint64_t UB_SIZE = 192 * 1024;         // DAV_2201
constexpr uint64_t REDUCE_TMP_BUF_SIZE = 4096;   // Reduce API 最小空间
constexpr uint64_t BASE_START = 8;               // param 布局: 8-float 边界对齐

uint64_t postBaseOff = ((BASE_START + hc + 7) / 8) * 8;
uint64_t combBaseOff = ((postBaseOff + hc + 7) / 8) * 8;
uint64_t paramFloats = combBaseOff + hc * hc;
uint64_t paramBufSize = align32_up(paramFloats * sizeof(float));

// 每行 float 开销
uint64_t perRowFloats = mixHcAlign     // mixesIn
                      + 3 * hcAlign    // outQueuePre + outQueuePost + rowColTmp
                      + hc * hcAlign   // combBuf
                      + 2 * hc         // flatPre + flatPost
                      + hc * hc;       // flatComb

uint64_t perRowBytes = perRowFloats * sizeof(float);
uint64_t fixedBytes = paramBufSize + REDUCE_TMP_BUF_SIZE + 128;
uint64_t availBytes = (UB_SIZE > fixedBytes) ? (UB_SIZE - fixedBytes) : 0;
uint64_t tileRows = availBytes / perRowBytes;

if (tileRows < 1) tileRows = 1;
if (tileRows > 255) tileRows = 255;  // repeatTimes 限制
```

对于 `hc=4` 典型值: `mixHcAlign=24, hcAlign=8, T_max ≈ 512`.

### 5.5 对齐约束

| 参数 | 公式 | hc=4 示例 | 说明 |
|------|------|----------|------|
| `mixHcAlign` | `align32(mixHc * 4) / 4` | 24 | 96B 已是 32B 倍数 |
| `hcAlign` | `align32(hc * 4) / 4` | 8 | 16B → 32B padding |
| `hc * hcAlign` | `hc * hcAlign` | 32 | comb 矩阵每行 stride |

- **Reduce API count 参数**: 使用有效元素数 `hc`（不是 hcAlign）
- **UB rowOffset 计算**: 使用对齐后的 `hcAlign` stride
- **对齐策略**: 计算时内部用 hcAlign stride（满足 Reduce 对齐要求），写回 GM 时用紧凑格式

---

## 6. 数据流设计

### 6.1 整体流程（Per Core）

```
每核 for tileIdx in [0, tileNum):
    │
    ├─ CopyIn:  GM→UB 加载 mixes[tile:T, mixHcAlign]
    │           - 对齐时: DataCopyPad 一次搬运 T 行
    │           - 非对齐时: 逐行 DataCopyPad
    │
    ├─ LoadParams: 展开 hcScale[3] + hcBase[mixHc] 到 tmpBufParams_
    │
    ├─ Compute:  for row t in [0, T):
    │   ├─ 阶段 A — Pre sigmoid
    │   │   Muls(mixes[t], s0, hc) → Add(base_pre, hc) → Muls(-1.0, hc)
    │   │   → Exp(hc) → Adds(1.0, hc) → Div(1.0, tmp, hc) → Adds(eps, hc)
    │   │   → compact flatPre[t, :hc]
    │   │
    │   ├─ 阶段 A — Post sigmoid
    │   │   Muls(mixes[t], s1, hc) → Add(base_post, hc) → Muls(-1.0, hc)
    │   │   → Exp(hc) → Adds(1.0, hc) → Div(1.0, tmp, hc) → Muls(2.0, hc)
    │   │   → compact flatPost[t, :hc]
    │   │
    │   └─ 阶段 B — Comb Sinkhorn
    │       combRaw * s2 + base_comb → [hc, hc] matrix
    │       iter 0: ReduceMax(row) → Adds(-max) → Exp
    │               ReduceSum(row) → Muls(1/sum) → Adds(eps)
    │               ManualReduceSum(col) → Muls(1/(sum+eps)) // 逐元素
    │       iter 1..N-1:
    │               ReduceSum(row) → Muls(1/(sum+eps))
    │               ManualReduceSum(col) → Muls(1/(sum+eps)) // 逐元素
    │       → compact flatComb[t, :hc*hc]
    │
    └─ CopyOut: UB→GM 写回 compact pre/post/comb
```

### 6.2 Sigmoid 实现细节

sigmoid(x) = 1/(1+exp(-x))。使用 Adds/Muls 标量广播优化，避免 Duplicate 开销：

```
Step 1: Muls(raw, s, hc)        // scale
Step 2: Add(tmp, base, hc)      // bias
Step 3: Muls(tmp, -1.0, hc)     // negate
Step 4: Exp(tmp, tmp, hc)       // exp(-x)
Step 5: Adds(tmp, 1.0, hc)      // 1 + exp(-x)
Step 6: Div(ones, ones, tmp, hc) // 1.0 / (1+exp(-x))
Step 7: Adds(ones, eps, hc)     // + eps (pre only)
Step 8: Muls(tmp, 2.0, hc)      // * 2 (post only)
```

全程 FP32，无精度降级。`rowColTmp` 临时缓冲复用为常数 1.0 缓冲区和 Div 的 dst/src。

### 6.3 Sinkhorn 迭代数据流

对于每样本的 `[hc, hc]` 矩阵（UB 中存储为 `[hc, hcAlign]` 布局，填充列 = 0）：

**第 0 次迭代** (含 Exp 稳定化):

1. 行最大值稳定化: `ReduceMax<float>` (Level 2) 逐行 → `Adds(-maxVal, hc)` → `Exp(hc)`
2. 行归一化: `ReduceSum<float, true>` (Level 2) 逐行 → `Muls(1.0/sumVal, hc)`, 第 0 次后 `+ eps`
3. 列归一化: 手动逐元素列求和 → 逐元素 `/(colSum + eps)`

**第 1..sinkhorn_iters-1 次迭代**:

1. 行归一化: `ReduceSum<float, true>` (Level 2) 逐行 → `Muls(1.0/(sumVal+eps), hc)`
2. 列归一化: 手动逐元素列求和 → 逐元素 `/(colSum + eps)`

**设计决策: 列归一化采用手动逐元素循环而非 `Pattern::Reduce::RA`**

| 方案 | 优势 | 劣势 |
|------|------|------|
| Pattern::Reduce::RA | 批量处理，SIMD 加速 | 需要 hcAlign 32B 对齐；hc 较小(<=32)时启动开销大于收益；需额外 dst tensor 和 sharedTmpBuffer |
| **手动逐元素循环** (采用) | 无对齐要求；代码简洁；hc<=32 时 loop 开销可忽略 | hc 大时效率下降 (MAX_HC=32 消除此风险) |

基于 `hc <= 32` 约束，手动循环方案更优。

---

## 7. API 映射表

### 7.1 数据搬运

| 操作 | API | 参数要点 |
|------|-----|---------|
| GM→UB (mixes) | `DataCopyPad` | 对齐时一次搬运 T 行；非对齐时逐行搬运 |
| UB→GM (pre) | `DataCopyPad` | `{1, T*hc*4, 0, 0}` |
| UB→GM (post) | `DataCopyPad` | 同上 |
| UB→GM (comb) | `DataCopyPad` | `{1, T*hc*hc*4, 0, 0}` |

### 7.2 逐元素运算

| 操作 | API | 用途 |
|------|-----|------|
| 逐元素乘法 | `Mul<float>` | pre/post scale |
| 逐元素加法 | `Add<float>` | 加 bias |
| 逐元素除法 | `Div<float>` | sigmoid 倒数 |
| 指数函数 | `Exp<float>` | sigmoid + Sinkhorn exp |
| 标量加法 | `Adds<float>` | +eps, +1.0, -maxVal |
| 标量乘法 | `Muls<float>` | *(-1.0), *(2.0), *(1/sum) |

### 7.3 归约运算

| 操作 | API | 说明 |
|------|-----|------|
| 行最大值 | `ReduceMax<float>` (Level 2) | `count=hc`，有效元素数 |
| 行求和 | `ReduceSum<float, true>` (Level 2) | `count=hc` |

### 7.4 API 验证状态

| API | 验证来源 | 状态 |
|-----|---------|------|
| `ReduceMax<float>` (Level 2) | `/ascendc-api-best-practices` → api-reduce.md §2 | 已验证 |
| `ReduceSum<float, true>` (Level 2) | 同上 | 已验证 |
| `Exp<float>` | Vector Unary API | 已验证 |
| `Mul`, `Add`, `Div` | Vector Binary API | 已验证 |
| `Adds`, `Muls` | Vector Binary Scalar API | 已验证 |
| `DataCopyPad` | Data Copy API | 已验证 |
| `SetValue` / `GetValue` | Common API | hc<=32 逐元素访问（已确认可用） |

---

## 8. Tiling 数据结构

### 8.1 Struct 定义

```cpp
#define HC_SPLIT_SINKHORN_MAX_HC 32
#define HC_SPLIT_SINKHORN_MAX_MIX_HC ((2 + HC_SPLIT_SINKHORN_MAX_HC) * HC_SPLIT_SINKHORN_MAX_HC)

struct HcSplitSinkhornTiling {
    // 基础 shape 信息
    uint64_t totalBatch;         // B = b * s
    uint64_t hc;                 // hc 维度
    uint64_t mixHc;              // mix_hc = (2+hc)*hc
    uint32_t sinkhornIters;      // Sinkhorn 迭代次数 (>= 1)
    float    eps;                // 数值稳定常数

    // 对齐信息
    uint64_t mixHcAlign;         // 32B 对齐后的 mix_hc（每行元素数）
    uint64_t hcAlign;            // 32B 对齐后的 hc

    // 多核切分
    uint64_t rowsPerCore;        // 每核处理的行数
    uint32_t tailCoreRows;       // 尾核行数
    uint32_t usedCoreNum;        // 实际使用的核数

    // UB 切分
    uint32_t tileRows;           // 每 tile 行数 (T)
    uint32_t tilesPerCore;       // 每核 tile 数

    // 参数
    float hcScale[3];
    float hcBase[HC_SPLIT_SINKHORN_MAX_MIX_HC];
};
```

### 8.2 Host 侧 Tiling 计算流程

```
1. 获取 AI Core 数量: coreNum = GetCoreNumAiv() 或 aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)
2. 计算 totalBatch = b * s
3. 计算 mixHc = (2 + hc) * hc
4. 算对齐:
     mixHcAlign = align32_elements(mixHc, sizeof(float))
     hcAlign = align32_elements(hc, sizeof(float))
5. 多核切分:
     rowsPerCore = ceil(totalBatch / coreNum)
     usedCoreNum = ceil(totalBatch / rowsPerCore)
     tailCoreRows = totalBatch - (usedCoreNum-1) * rowsPerCore
6. UB 切分:
     tileRows = calcTileRows(hc, mixHcAlign, hcAlign)
     tilesPerCore = ceil(rowsPerCore / tileRows)
```

### 8.3 Kernel 侧执行流程

```
Kernel Entry:
  1. blockIdx = GetBlockIdx()
  2. rowOffset = blockIdx * rowsPerCore
  3. totalRows = (blockIdx < usedCoreNum-1) ? rowsPerCore : tailCoreRows
  4. tileNum = ceil(totalRows / tileRows)
  5. InitBuffer 分配所有 UB Buffer
  6. LoadParams: 将 hcScale + hcBase 加载到 UB

Process Loop (for tileIdx in [0, tileNum)):
  1. rowsThisTile = (tileIdx == tileNum-1) ? (totalRows - rowsDone) : tileRows
  2. CopyIn(rowsThisTile, rowsDone): GM→UB 加载 mixes
  3. Compute(rowsThisTile): 阶段 A (pre/post sigmoid) + 阶段 B (comb sinkhorn)
  4. CopyOut(rowsThisTile, rowsDone): UB→GM 写回 compact pre, post, comb
  5. rowsDone += rowsThisTile
```

---

## 9. 精度策略

### 9.1 精度标准

采用浮点计算类社区标准 (`/ops-precision-standard` → `float_compute_community.md`):

| 指标 | 阈值 | 说明 |
|------|------|------|
| MERE | < 2^-13 ≈ 0.000122 | 平均相对误差 |
| MARE | < 10 * 2^-13 ≈ 0.00122 | 最大相对误差 |

### 9.2 数值稳定性保护

| 保护机制 | 实现方式 | 覆盖场景 |
|---------|---------|---------|
| FP32 全链路 | 所有中间计算使用 `float` | 无精度降级 |
| Exp 稳定化 | `exp(x - row_max)` 防上溢 | Comb 首轮迭代 |
| 除法保护 | 分母 `+ eps` (默认 1e-6) | 所有归一化步骤 |

### 9.3 迭代误差累积分析

Sinkhorn 迭代 `sinkhorn_iters` 次 (默认 20)，每次含两次归约+归一化:

- 单次 ReduceSum (hc=4 元素): FP32 相对误差 ≈ `4 * 2^-23 ≈ 4.77e-7`
- 20 次迭代累积: ≈ `20 * 4.77e-7 ≈ 9.54e-6`，远小于 MERE 阈值
- 实测结果: MERE ≈ 8.8e-9, MARE ≈ 1.5e-7, 约为阈值的 1/10000

对于 `hc <= 32` 场景，FP32 精度充足，无需二分累加。

---

## 10. 边界情况处理

| 边界情况 | 处理策略 | 位置 |
|---------|---------|------|
| `B = 1` | 单核处理，无多核并行开销 | Host Tiling |
| `hc = 1` | hcAlign=8 (16B→32B padding)，仅 1 个有效元素 | calcTileRows |
| `mixHc` 非 32B 对齐 | DataCopyPad 逐行搬运 (isPad=false) | CopyIn |
| `eps = 0` | 不特殊处理 (调用方保证 eps > 0) | — |
| `sinkhorn_iters = 0` | Tiling 中 clamp 为 1 (至少 1 次含 Exp) | ComputeTiling |
| `sinkhorn_iters` 很大 | Tiling 不受影响 (循环在 UB 内) | Kernel Compute |
| 尾核行不足 | totalRows = tailCoreRows | Kernel Init |
| 尾 tile 行不足 | rowsThisTile = totalRows - rowsDone | Process loop |
| `hc > 32` | Tiling 数组越界风险: MAX_HC=32 编译期限定 | 编译期 |
| 除零保护 | `+ eps` 保证分母 > 0 | 数学公式 |

---

## 11. 性能考量

### 11.1 计算特征

| 阶段 | 主要运算 | 每样本计算量 (hc=4) | 瓶颈 |
|------|---------|---------------------|------|
| Pre sigmoid | Exp, Add, Mul, Div, Adds, Muls | ~24 FLOP | Vector |
| Post sigmoid | Exp, Add, Mul, Div, Adds, Muls | ~24 FLOP | Vector |
| Comb Sinkhorn (20 iters) | ReduceMax*4 + ReduceSum*160 + Exp*4 + 逐元素 | ~2000+ FLOP | Scalar-bound |

Sinkhorn 迭代占总计算量 > 95%，全程在 UB 内完成。

### 11.2 优化措施

1. **UB 内 Sinkhorn 循环**: 全部迭代在 UB 内完成，避免 GM<->UB 反复搬运
2. **标量广播优化 (Adds/Muls)**: 替代 Duplicate+Sub/Div，每样本每行节省 1 次 Duplicate + 1 条算术指令
3. **单 Buffer**: 避免双缓冲的 2x UB 开销
4. **手动列归一化**: hc<=32 时手动循环优于 Pattern::Reduce::RA

### 11.3 潜在瓶颈

- **逐行 ReduceMax/ReduceSum**: 每样本每行调用 Level 2 Reduce，`iters * 2 * hc = 160` 次/hc=4。由于 hc 仅 4 元素，API 调用开销占比高（Scalar-bound ~74.5%）
- **SetValue/GetValue**: hc<=32 时每样本 ~100 次，总数可控

### 11.4 实测性能 (C1: b=2, s=8, hc=4, iters=20)

| 指标 | 值 |
|------|-----|
| Kernel 执行时间 | ~39 us |
| AI Vector 时间 | ~37 us |
| AIV Scalar 占比 | ~74.5% |
| Block 数量 | 16 |
| 精度 (MERE) | 8.77e-09 |
| 精度 (MARE) | 1.50e-07 |

---

## 12. 设计决策总结

| 决策 | 选择 | 理由 |
|------|------|------|
| 技术路线 | SIMD/MemBase | DAV_2201, 无法使用 RegBase/Blaze |
| 多核切分 | Batch 维度 (B=b*s) | 各行独立，天然并行，无跨核通信 |
| 归约接口 | Level 2 ReduceMax/ReduceSum | hc 较小时 API 开销可接受 |
| 列归一化 | 手动逐元素循环 | hc <= 32 时优于 Pattern::Reduce::RA |
| Sigmoid 实现 | Adds/Muls 标量广播 | 节省 Duplicate+Sub/Div |
| Sinkhorn 循环位置 | 全部在 UB 内 | 避免 GM 反复读写 |
| 精度 | FP32 全链路 | 输入输出均为 FP32，无降精度必要 |
| UB Buffer | 单缓冲 (count=1) | 数据量适中，简化同步 |
| 对齐策略 | 内部对齐 + 紧凑写出 | 满足 Reduce API 对齐约束，输出无填充 |

---

## 参考资料

- `/ascendc-tiling-design` -- 通用设计要素（多核/UB 切分、Buffer 规划、分支覆盖）
- `/ascendc-tiling-design/references/reduction/patterns.md` -- AR 模式归约设计
- `/ascendc-api-best-practices/references/api-arithmetic.md` -- Adds/Muls 标量广播优化
- `/ascendc-api-best-practices/references/api-reduce.md` -- Reduce API (Level 2) 使用指南
- `/npu-arch` -- DAV_2201 硬件参数 (UB 192KB, L0C 128KB, Vector Core 24)
- `/ops-precision-standard` -- FP32 浮点社区精度标准
