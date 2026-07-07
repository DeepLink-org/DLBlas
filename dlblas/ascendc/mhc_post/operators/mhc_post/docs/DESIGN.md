# MHC Post 算子架构设计文档 (DESIGN.md)

> **算子名称**: mhc_post (Multi-Head Combining Post-processing)
> **芯片型号**: Ascend910B2
> **NpuArch**: DAV_2201
> **CANN 版本**: 9.0.0
> **文档版本**: v2.0
> **编写日期**: 2026-07-02

---

## 1. 算子概述

### 1.1 数学定义

```
输入:
  x              ∈ R^{n0 × n1 × h}           (bf16)
  residual       ∈ R^{n0 × n1 × M × h}        (bf16)
  post_layer_mix ∈ R^{n0 × n1 × M × 1}        (fp32)
  comb_res_mix   ∈ R^{n0 × n1 × M × M}        (fp32)

计算 (fp32 中间精度):
  term2_{a,b,m,n} = Σ_k comb_res_mix_{a,b,m,k} × residual_float_{a,b,k,n}
                  = [M×M] @ [M×H] → [M×H] 矩阵乘法

  out_{a,b,m,n}   = x_float_{a,b,n} × post_layer_mix_{a,b,m,1} + term2_{a,b,m,n}

输出:
  output = cast_to_bf16(out) ∈ R^{n0 × n1 × M × h}   (bf16)
```

其中:
- M = mhc_mult = 4 (固定常量)
- h = 1280 (默认值)
- 默认 shape: n0=2, n1=4096, h=1280, M=4

### 1.2 einsum 展开为向量点积

M=4 时:

```
for m in 0..3:
    term2[m,:] = cmb[m,0] × residual[0,:]
               + cmb[m,1] × residual[1,:]
               + cmb[m,2] × residual[2,:]
               + cmb[m,3] × residual[3,:]
```

每条输出行需要: 4 次标量-向量乘 (Muls) + 3 次向量累加 (Add) = 7 条向量指令。

### 1.3 算子分类

| 分类维度 | 判定 |
|---------|------|
| 主类别 | **MatMul** (批量小矩阵乘 M=4, K=4, N=1280) + **Elementwise** 融合 |
| 子类别 | Broadcast MulAdd + 精度转换 (bf16↔fp32) |
| 计算特征 | K=4 极短向量点积，Memory-Bound |

### 1.4 输入输出规格

| 张量 | Shape | dtype | 默认大小 | 说明 |
|------|-------|-------|---------|------|
| x | (n0, n1, h) | bfloat16 | 20 MB | 主路径输入 |
| residual | (n0, n1, M, h) | bfloat16 | 80 MB | 残差输入 (4 行) |
| post_layer_mix | (n0, n1, M) | float32 | 128 KB | 广播乘系数 [n0,n1,M,1] |
| comb_res_mix | (n0, n1, M, M) | float32 | 512 KB | 4×4 组合矩阵 |
| **output** | (n0, n1, M, h) | bfloat16 | 80 MB | 融合输出 |

---

## 2. 硬件环境

### 2.1 目标平台

| 参数 | 值 | 来源 |
|------|-----|------|
| 芯片型号 | Ascend910B2 | 用户需求输入 |
| NpuArch | **DAV_2201** | `/npu-arch` skill 查询 |
| SocVersion | Ascend910B2 | 用户需求输入 |
| CANN 版本 | 9.0.0 | 用户需求输入 |
| `__NPU_ARCH__` | 2201 | 编译宏 |
| `--npu-arch` 编译选项 | `dav-2201` | CMake 参数 |
| CPU 架构 | aarch64-linux | 头文件搜索路径 |

### 2.2 硬件资源 (DAV_2201, Ascend910B2)

| 资源 | 规格 | 设计约束 |
|------|------|---------|
| **UB** | 192 KB (196608 bytes) | 单 tile 数据上限 |
| **L0C** | 128 KB | 本方案不涉及 (Vector 路线) |
| **L1** | 可配置分区 | 本方案不涉及 |
| **BT** | 1 KB | 系数暂存充足 |
| **AI Vector Core 数** | 24 | 运行时通过 aclrtGetDeviceInfo 获取 |
| **频率** | 1.8 GHz | 性能建模参考 |
| **HBM 带宽** | ~1.2 TB/s (经验值) | Memory-Bound 分析上限 |
| **SIMD 宽度** | 256 bit | 单指令 64×bf16 或 8×fp32 |

---

## 3. 方案决策

### 3.1 决策流程

```
Step 0: 平台路由 (platform routing)
  NpuArch = DAV_2201 (Ascend910B2)
  → 非 DAV_3510，排除 RegBase 路线和 Blaze/tensor_api 路线
  → 走通用 SIMD/MemBase 路线

Step 0.5: 算子类型匹配
  加载 ascendc-tiling-design → matmul/patterns.md:
  - DAV_2201 上的 MatMul → MatMul 高阶 API (MatmulImpl + MatmulApiTiling)
  - 但需专项评估 Cube vs Vector

Step 1: Cube vs Vector 专项评估
  Cube 单元约束:
  - 基本粒度: 16×16 MAC/cycle
  - K=4: MAC 利用率仅 25% (16 个乘法器仅使用 4 个)
  - M=4: 需 padding 到 16，浪费 75% 的 L0C 空间和 MMA 带宽
  - 8192 个独立 batch: 8192 次 Cube launch/teardown 开销累积严重

  Vector 单元:
  - 无粒度限制: 每指令处理 256-bit 向量数据
  - K=4 展开: 7 条向量指令/输出行 (4 Muls + 3 Add)
  - 无 padding 浪费
  - bf16↔fp32 Cast 为单指令操作，几乎零开销

Step 2: 结论
  → 选择 Vector API (SIMD/MemBase) 路线
```

### 3.2 决策结论

| 决策项 | 选择 | 理由 |
|--------|------|------|
| **编程模型** | SIMD/MemBase (TPipe + TQue) | DAV_2201 标准路径 |
| **计算单元** | Vector (AIV) | K=4 点积 Muls+Add 优于 Cube 16×16 MAC |
| **API 体系** | Ascend C Vector API | Cast, Muls, Add, DataCopyPad |
| **不使用 MatmulImpl** | K=4 过小 | Cube 固定粒度 16×16 导致 75% 浪费 |
| **不使用 RegBase** | DAV_2201 不支持 | RegBase 为 DAV_3510 专有能力 |
| **不使用 Blaze** | DAV_2201 不支持 | tensor_api 为 DAV_3510 专有能力 |

### 3.3 与 MatMul 高阶 API 的详细性能建模对比

| 指标 | Cube (MatmulImpl) | Vector (本方案) | 说明 |
|------|-------------------|----------------|------|
| M/K 利用率 | 25% (4/16) | 100% | 无 padding |
| Batch launch 开销 | 8192 次 | 0 (循环内) | Vector 路径无 launch 开销 |
| L1/L0 传输开销 | 每 tile 多次 | 0 | Vector 仅需 UB |
| L0C→UB Fixpipe | 需要 | 不需要 | 减少数据搬运路径 |
| 总操作数 | 高 (含 padding 计算) | 低 (仅有效计算) | |

---

## 4. 多核切分策略

### 4.1 切分维度分析

| 维度 | 大小 | 可切分? | 理由 |
|------|------|---------|------|
| n0 | 2 | 否 | 仅 2 个 batch，切分无意义 |
| **n1** | 4096 | **是** | 8192 batch，均匀性好 |
| M | 4 | 否 | 固定常量，不可切 |
| h | 1280 | 否 | 核内 tile 切分，不参与核间切分 |

**决策**: 沿 **n1 维度** 均匀切分，每个 AI Core 处理连续的 n1 子范围。

### 4.2 核数动态计算

```cpp
// Host 侧: 运行时动态获取核数
int64_t availableCoreNum = 0;
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);

// 限制到 MAX_CORE_NUM (20) 或 n1 的最小值，避免过度切分
uint32_t blockNum = min(availableCoreNum, MAX_CORE_NUM, n1);
```

| 参数 | 值 | 说明 |
|------|-----|------|
| availableCoreNum | 24 (910B2) | 运行时获取，禁止硬编码 |
| MAX_CORE_NUM | 20 | 防止单核粒度过小 |
| 实际 blockNum | min(24, 20, 4096) = 20 | |
| n1PerCore | ceil(4096/20) = 205 | 每核处理 205 个 n1 位置 |

### 4.3 核内 n1 范围计算

```cpp
uint32_t n1PerCore = (n1 + blockNum - 1) / blockNum;  // 向上取整
uint32_t n1Start = blockIdx * n1PerCore;
uint32_t n1End = min(n1Start + n1PerCore, n1);

if (n1Start >= n1) { n1Start = n1; n1End = n1; }  // 空闲核跳过
```

- 尾核自动适应较小的 n1 范围
- 空闲核 (blockIdx * n1PerCore >= n1) 直接跳过
- n1PerCore = 205, 最末核处理 4096 - 19×205 = 4096 - 3895 = 201 个 n1 位置

---

## 5. UB Tiling 策略

### 5.1 两层 Tiling 结构

```
for a in 0..n0-1:                              [外循环: n0=2]
  for b in n1Start..n1End-1:                   [多核: 每核 ~205 个 n1]
    LoadCoefficients(a, b)                     [一次性加载 80B 系数到 TBuf]
    for ci in 0..cTiles-1:                     [UB tile: 沿 h 维度切分]
      CopyIn(tile ci+1)  ─┐
      Compute(tile ci)     ├── 三阶段流水线重叠
      CopyOut(tile ci-1)  ─┘
```

### 5.2 C_TILE 选择

| C_TILE | tiles/row (H=1280) | I/O buffer (双缓冲) | 总 UB | 结论 |
|--------|-------------------|---------------------|-------|------|
| 32 | 40 | ~2.5 KB | ~3 KB | 保守，循环开销大 |
| **64** | **20** | **~5.0 KB** | **~5 KB** | **推荐默认** |
| 128 | 10 | ~10 KB | ~10 KB | 可选优化 |
| 256 | 5 | ~20 KB | ~20 KB | UB 仍安全 |

**推荐 C_TILE = 64**:
- 128 字节 (64×2B) = 4×32B，自然对齐
- 20 次 tile 迭代，repeatTimes ≤ 255 约束满足
- UB 占用仅 2.5%
- C_TILE=128 作为性能调优候选 (减少 tile 迭代 50%)

### 5.3 尾块处理

```cpp
inline uint32_t TileSize(uint32_t ci) const {
    uint32_t remaining = h_ - ci * cTile_;
    return (remaining >= cTile_) ? cTile_ : remaining;
}
```

- h=1280, C_TILE=64 → 1280/64=20 整除，无尾块
- 通用性: 非整除 h 下尾块自动截断

### 5.4 系数加载策略 (B_TILE=1)

**关键设计决策**: 系数 (comb_res_mix 64B + post_layer_mix 16B = 80B) 逐 batch 加载到 TBuf，所有列 tile 共享。

| 方案 | 加载次数 | 总搬运量 | 复杂度 |
|------|---------|---------|--------|
| **每 batch 一次 (B_TILE=1)** | 8192 | 655 KB | 低 |
| 每 tile 一次 | 8192 × 20 = 163840 | 13.1 MB | 低 |
| B_TILE=N 批量 | 8192/N | 655 KB | 高 (buffer 管理/对齐/尾块) |

逐 batch 加载仅增加 655 KB 搬运量 (占总数据 ~180 MB 的 0.34%)，远低于批量方案的实现复杂度。**选择 B_TILE=1**。

---

## 6. Buffer 规划

### 6.1 UB Buffer 布局

```
┌──────────────────────────────────────────────────────┐
│ UB (192 KB)                                           │
│                                                       │
│  ┌─────────────────────┐  inQueRes_[0..3]:            │
│  │ residual[4][64] bf16│  TQue<VECIN, 2>, 4×128×2    │
│  │ ×2 double-buffered  │  = 1024 B                    │
│  ├─────────────────────┤                               │
│  │ x[64] bf16          │  inQueX_:                    │
│  │ ×2 double-buffered  │  TQue<VECIN, 2>, 128×2      │
│  │                     │  = 256 B                     │
│  ├─────────────────────┤                               │
│  │ coeff[20] fp32      │  coeffBuf_:                  │
│  │ (80B, single-buffer)│  TBuf<VECCALC> = 80 B       │
│  ├─────────────────────┤                               │
│  │ out[4][64] bf16     │  outQue_[0..3]:              │
│  │ ×2 double-buffered  │  TQue<VECOUT, 2>, 4×128×2   │
│  │                     │  = 1024 B                    │
│  ├─────────────────────┤                               │
│  │ resFp32_[0..3][64]  │  TBuf<VECCALC>, 4×256       │
│  │ = 1024 B            │                               │
│  │ term2_[0..3][64]    │  TBuf<VECCALC>, 4×256       │
│  │ = 1024 B            │                               │
│  │ xFp32_[64]          │  TBuf<VECCALC>, 256 B       │
│  │ tmpFp32_[64]        │  TBuf<VECCALC>, 256 B       │
│  └─────────────────────┘                               │
│                                                       │
│  总计: ~4944 B (2.5% of 192 KB)                       │
└──────────────────────────────────────────────────────┘
```

### 6.2 详细 Buffer 清单

| Buffer | 类型 | dtype | 元素数 | 份数 | 大小 (字节) | 用途 |
|--------|------|-------|--------|------|-------------|------|
| inQueRes_[0..3] | TQue<VECIN, 2> | bf16 | 64 | 4×2 | **1024** | 残差输入 4 行 (双缓冲) |
| inQueX_ | TQue<VECIN, 2> | bf16 | 64 | 1×2 | **256** | x 输入 (双缓冲) |
| coeffBuf_ | TBuf<VECCALC> | fp32 | 20 | 1 | **80** | 系数: cmb[4×4] + pm[4] |
| outQue_[0..3] | TQue<VECOUT, 2> | bf16 | 64 | 4×2 | **1024** | 输出 4 行 (双缓冲) |
| resFp32_[0..3] | TBuf<VECCALC> | fp32 | 64 | 4 | **1024** | bf16→fp32 转换暂存 |
| term2_[0..3] | TBuf<VECCALC> | fp32 | 64 | 4 | **1024** | 点积累加 / 最终结果 |
| xFp32_ | TBuf<VECCALC> | fp32 | 64 | 1 | **256** | x bf16→fp32 转换暂存 |
| tmpFp32_ | TBuf<VECCALC> | fp32 | 64 | 1 | **256** | 临时 Muls 结果 |
| **总计** | | | | | **~4944** | UB 利用率 2.5% |

### 6.3 系数 TBuf 布局

```
coeffBuf_ (20 × fp32 = 80 bytes):

Offset  内容
0..15   comb_res_mix[0,0], [0,1], [0,2], [0,3]   (第 0 行)
16..31  comb_res_mix[1,0], [1,1], [1,2], [1,3]   (第 1 行)
32..47  comb_res_mix[2,0], [2,1], [2,2], [2,3]   (第 2 行)
48..63  comb_res_mix[3,0], [3,1], [3,2], [3,3]   (第 3 行)
64..79  post_layer_mix[0], [1], [2], [3]          (4 个广播标量)
```

### 6.4 为什么系数用 TBuf 而非 TQue

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| **TBuf (当前)** | 全列 tile 共享，零同步开销 | 无法与计算流水重叠 | **推荐** |
| TQue | EnQue/DeQue 流水同步 | 每 tile 需重新加载 (80B×20=1.6KB/batch) | 过度设计 |

系数仅 80B，每 batch 加载一次后通过 `LocalTensor::GetValue()` 读取标量值，延迟可忽略。

### 6.5 双缓冲流水线

```
时间轴 →
CopyIn:   [tile 0─────────][tile 1─────────][tile 2─────────]...
Compute:          [tile 0─────────][tile 1─────────][tile 2──]...
CopyOut:                 [tile 0─────────][tile 1─────────]...
```

实现要点:
1. **预加载**: 进入循环前提前 `CopyIn(tile 0)`
2. **重叠**: 循环内 `CopyIn(tile N+1)` 与 `Compute(tile N)` 与 `CopyOut(tile N-1)` 重叠
3. **收尾**: 循环后单独 `CopyOut(last tile)`

---

## 7. 数据流设计

### 7.1 完整数据流

```
┌──────────────────────────────────────────────────────────────┐
│ GM → UB (CopyIn)                                              │
│                                                               │
│  residual[a,b,:,ci*64:(ci+1)*64] → DataCopyPad → inQueRes_   │
│  x[a,b,ci*64:(ci+1)*64]          → DataCopyPad → inQueX_     │
│  comb_res_mix[a,b,:,:]           → DataCopyPad → coeffBuf_   │
│  post_layer_mix[a,b,:]           → DataCopyPad → coeffBuf_   │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│ UB 计算 (Compute, fp32)                                       │
│                                                               │
│  1. Cast bf16→fp32 (CAST_NONE):                               │
│     resFp32[m] ← resBf16[m]                                   │
│     xFp32      ← xBf16                                        │
│                                                               │
│  2. K=4 向量点积 (Muls + Add):                                │
│     for m in 0..3:                                            │
│       term2[m] ← cmb[m,0] × resFp32[0]                       │
│       for k in 1..3:                                          │
│         tmp ← cmb[m,k] × resFp32[k]                           │
│         term2[m] ← term2[m] + tmp                             │
│                                                               │
│  3. Broadcast MulAdd:                                         │
│     for m in 0..3:                                            │
│       tmp ← xFp32 × pm[m]                                     │
│       outFp32[m] ← tmp + term2[m]                             │
│                                                               │
│  4. Cast fp32→bf16 (CAST_ROUND):                              │
│     outBf16[m] ← outFp32[m]                                   │
│                                                               │
├──────────────────────────────────────────────────────────────┤
│ UB → GM (CopyOut)                                             │
│                                                               │
│  outBf16[m] → DataCopyPad → output[a,b,m,ci*64:(ci+1)*64]    │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 单 Batch 流水线伪代码

```
PipelineBatch(a, b):
    // 1. 一次性加载系数到 TBuf (所有列 tile 共享)
    LoadCoefficients(a, b)

    // 2. 预加载第一个 tile
    CopyInData(a, b, tile_0_start, TileSize(0))

    // 3. 流水线主循环
    for ci in 0..(cTiles-1):
        if ci+1 < cTiles:
            CopyInData(a, b, tile_{ci+1}_start, TileSize(ci+1))   // 异步启动

        Compute(TileSize(ci))                                       // 等待 DeQue + 计算

        if ci > 0:
            CopyOut(a, b, tile_{ci-1}_start, TileSize(ci-1))       // 写出前一个

    // 4. 写出最后一个 tile
    CopyOut(a, b, tile_{cTiles-1}_start, TileSize(cTiles-1))
```

### 7.3 计算核心: K=4 向量点积

每行 m 的指令序列:

```
// m=0: term2[0] = cmb[0,0]*res[0] + cmb[0,1]*res[1] + cmb[0,2]*res[2] + cmb[0,3]*res[3]
Muls(term2[0], resFp32[0], cmb[0,0], count)                    // 初始赋值
Muls(tmp,      resFp32[1], cmb[0,1], count); Add(term2[0], term2[0], tmp, count)
Muls(tmp,      resFp32[2], cmb[0,2], count); Add(term2[0], term2[0], tmp, count)
Muls(tmp,      resFp32[3], cmb[0,3], count); Add(term2[0], term2[0], tmp, count)

// m=1..3: 同上模式, 替换 cmb[m,*]
```

全 M=4 行总计: 16 Muls + 12 Add = **28 条向量指令/tile**.

### 7.4 广播 MulAdd

```
for m in 0..3:
    Muls(tmp, xFp32, pm[m], count)       // x * post_layer_mix[m]
    Add(term2[m], tmp, term2[m], count)  // + term2[m] (in-place)
```

总计: 4 Muls + 4 Add = **8 条向量指令/tile**.

### 7.5 GM 地址偏移公式

```
residual[a,b,m,c]    → offset = (a * n1 + b) * M * h + m * h + c
x[a,b,c]             → offset = (a * n1 + b) * h + c
comb_res_mix[a,b,m,k] → offset = (a * n1 + b) * M * M + m * M + k
post_layer_mix[a,b,m] → offset = (a * n1 + b) * M + m
output[a,b,m,c]      → offset = (a * n1 + b) * M * h + m * h + c
```

实现通过 `GlobalTensor` 的 `operator[]` 切片式寻址:

```cpp
uint64_t elemBase = static_cast<uint64_t>(a) * n1_ + b;
residualGm_[elemBase * MHC_MULT * h_ + m * h_ + cStart];
xGm_[elemBase * h_ + cStart];
combResMixGm_[elemBase * MHC_MULT * MHC_MULT];
postLayerMixGm_[elemBase * MHC_MULT];
outGm_[elemBase * MHC_MULT * h_ + m * h_ + cStart];
```

---

## 8. API 映射

### 8.1 Device 侧 (Kernel)

| 功能 | API | 关键参数 | 验证状态 |
|------|-----|---------|---------|
| GM→UB 搬运 (非对齐) | `DataCopyPad` | `{1, bytes, 0, 0}`, `{false, 0, 0, 0}` | 已验证 (编译通过) |
| bf16→fp32 无损转换 | `Cast(dstFp32, srcBf16, RoundMode::CAST_NONE, count)` | CAST_NONE | 已验证 |
| fp32→bf16 舍入转换 | `Cast(dstBf16, srcFp32, RoundMode::CAST_ROUND, count)` | CAST_ROUND | 已验证 |
| 标量×向量 | `Muls(dst, src, floatScalar, count)` | 第 3 参数为 float | 已验证 |
| 向量+向量 (in-place) | `Add(dst, src0, src1, count)` (dst==src0 支持) | in-place supported | 已验证 |
| TQue 双缓冲入队 | `EnQue(localTensor)` | VECIN/VECOUT, BUFFER_NUM=2 | 已验证 |
| TQue 双缓冲出队 | `DeQue<T>()` | VECIN/VECOUT, BUFFER_NUM=2 | 已验证 |
| TBuf 静态分配 | `InitBuffer(buf, size)` | VECCALC | 已验证 |
| TBuf 数据访问 | `buf.Get<T>()` / `localTensor.GetValue(idx)` | 标量读取 | 已验证 (LocalTensor, 非 GlobalTensor) |
| 核索引 | `GetBlockIdx()` | 多核切分 | 已验证 |

### 8.2 Host 侧

| 功能 | API | 用途 |
|------|-----|------|
| 设备初始化 | `aclInit(nullptr)` + `aclrtSetDevice(0)` | 运行时初始化 |
| 核数获取 | `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` | 动态获取核数 |
| GM 分配 | `aclrtMalloc(ACL_MEM_MALLOC_HUGE_FIRST)` | Device 内存 |
| Host↔Device 搬运 | `aclrtMemcpy(ACL_MEMCPY_HOST_TO_DEVICE/DEVICE_TO_HOST)` | 数据搬运 |
| Kernel 启动 | `kernel<<<blockNum, nullptr, stream>>>` | 多核并行 |
| 同步 | `aclrtSynchronizeStream(stream)` | 等待完成 |

### 8.3 API 黑名单合规

| 禁止 API | 是否使用 | 说明 |
|----------|---------|------|
| `GlobalTensor::SetValue()` | 否 | 未使用 |
| `GlobalTensor::GetValue()` | 否 | 仅对 `LocalTensor` (coeffBuf_) 使用 GetValue |
| `DataCopy` (非对齐 GM↔UB) | 否 | 统一使用 `DataCopyPad` |

---

## 9. 精度策略

### 9.1 精度路由

```
ops-precision-standard skill:
  包含数值计算? → 是
  均为浮点? → 是 (bf16 + fp32)
  商用标准? → 否
  → 浮点计算类社区标准
```

### 9.2 精度标准

| 输出 dtype | MERE 阈值 | MARE 阈值 | 判定 |
|-----------|----------|----------|------|
| bf16 | 2^-7 ≈ 0.00781 | 10 × 2^-7 ≈ 0.0781 | MERE < threshold AND MARE < 10*threshold |

```
MERE = avg(|actual - golden| / (|golden| + 1e-7))
MARE = max(|actual - golden| / (|golden| + 1e-7))
```

### 9.3 混合精度设计

| 阶段 | 操作 | 数据类型 | 舍入 | 精度影响 |
|------|------|---------|------|---------|
| 加载 | residual, x bf16 读取 | bf16 | -- | 输入精度 7 位尾数 |
| 升精度 | Cast bf16→fp32 | fp32 | CAST_NONE | 无损扩展 |
| 加载 | comb_res_mix, post_layer_mix 读取 | fp32 | -- | 原生 23 位尾数 |
| 点积 | Muls + Add, K=4 展开 | fp32 | -- | 4 次累加，fp32 精度充足 |
| 融合 | Broadcast Mul + Add | fp32 | -- | 同为 fp32 |
| 降精度 | Cast fp32→bf16 | bf16 | CAST_ROUND | 1-ULP 舍入误差 |

### 9.4 数值稳定性分析

| 风险 | 评估 | 缓解 |
|------|------|------|
| K=4 累加误差 | **低**. fp32 23 位尾数 >> bf16 7 位尾数 | 无需 Kahan |
| 大数吃小数 | **低**. 输入量级相近 (同网络层) | 无需特殊处理 |
| fp32→bf16 截断 | **中**. 被截断到 bf16 精度 | CAST_ROUND 就近舍入 |
| INF/NAN | **极低**. 无除法/指数等风险操作 | 正常值域内安全 |

### 9.5 实测精度 (Round 0, bf16 输出)

| Shape | 总元素 | 差异 > 1e-2 | 通过率 | Max Diff |
|-------|--------|-------------|--------|----------|
| (2,4096,4,1280) | 41,943,040 | 13 | 99.99997% | 1.5625e-02 |

13 个"超标"元素经 PyTorch vs NumPy 交叉验证确认为 bf16 1-ULP 舍入差异，非 Kernel bug。精度达标。

---

## 10. 性能评估

### 10.1 计算与数据量

| 指标 | 值 | 说明 |
|------|-----|------|
| 总 MFLOPs | ~377.5 | 9 FLOP/elem × 41.9M 元素 |
| 总算术指令/tile | 36 条 (28 点积 + 8 融合) | + 4 Cast |
| 总数据量 | ~180.6 MB | 输入 100.6 MB + 输出 80 MB |
| 计算密度 | ~2.09 FLOP/Byte | Memory-Bound |

### 10.2 理论上限

| 瓶颈 | 上限 | 计算方式 |
|------|------|---------|
| HBM 带宽 | ~150 us | 180.6 MB / 1200 GB/s |
| Vector 计算 | ~1.09 us | 377.5 MFLOPs / 345.6 GFLOPS |

**结论**: 算子属于 **Memory-Bound**，性能上限由 HBM 带宽决定。理论最优延迟约 150 us。

### 10.3 实测性能 (Round 001, v1 单 Buffer)

| 指标 | 值 | 分析 |
|------|-----|------|
| 总延迟 | 7,767 us | |
| AIV Vector | 4,090 us (52.7%) | 有效计算 |
| AIV MTE2 (Load) | 1,563 us (20.1%) | 加载等待 |
| AIV MTE3 (Store) | 1,263 us (16.3%) | 存储等待 |
| 数据吞吐 | ~24.4 GB/s | 带宽利用率 ~2% |

**v1 瓶颈**: 单 Buffer 无流水线 → Load/Compute/Store 完全串行。

### 10.4 v2 (双缓冲) 预期提升

| | v1 (单 Buffer) | v2 (双缓冲) | 加速比 |
|---|---------------|------------|--------|
| 流水线 | 串行 (1×) | 三阶段重叠 (≤ 3×) | ~2-3× |
| 预期延迟 | 7,767 us | ~2,500-4,000 us | |
| 预期吞吐 | ~24.4 GB/s | ~50-80 GB/s | |
| 带宽利用率 | ~2% | ~4-7% | |

### 10.5 进一步优化空间

| 优化项 | 预估收益 | 复杂度 | 优先级 |
|--------|---------|--------|--------|
| C_TILE=128 | ~5-10% | 低 | P1 |
| B_TILE>1 系数复用 | ~0.3% | 高 | P3 |
| DataCopyExtParams 合并残差行 | ~1-2% | 中 | P2 |
| 多 Stream | 有限 (已达 memory-wall) | 高 | P4 |

---

## 11. 分支场景覆盖

### 11.1 Shape 变体

| 场景 | 条件 | 策略 |
|------|------|------|
| 标准 | n0=2, n1=4096, h=1280 | 常规路径 |
| n1 非整除 | n1 % blockNum != 0 | 尾核自动缩小范围: `n1End = min(n1Start + n1PerCore, n1)` |
| h 非整除 | h % C_TILE != 0 | 尾 tile: `TileSize(ci) = min(cTile_, h_ - ci*cTile_)` |
| 小 n1 (< 核数) | n1 < blockNum | blockNum = n1 |
| 空闲核 | blockIdx * n1PerCore >= n1 | n1Start = n1End = n1, 全跳过 |
| 单 n0 | n0 = 1 | 外循环 1 次, 逻辑不变 |

### 11.2 数据类型

本算子仅需支持需求指定的 bf16 I/O + fp32 计算。若未来扩展:

| 场景 | 修改点 |
|------|--------|
| fp16 I/O | `bfloat16_t` → `half`, 其余逻辑不变 |
| 纯 fp32 | 去除 Cast, UB 占用翻倍但仍安全 |

---

## 12. Tiling 数据结构

```cpp
// 编译期常量 (mhc_post_tiling.h)
constexpr uint32_t B_TILE = 1;      // 逐 batch 处理
constexpr uint32_t C_TILE = 64;     // 列 tile 大小
constexpr uint32_t MHC_MULT = 4;    // 多头压缩倍数 (固定)
constexpr uint32_t MAX_CORE_NUM = 20;  // 最大核数
constexpr uint32_t N0_DEFAULT = 2;
constexpr uint32_t N1_DEFAULT = 4096;
constexpr uint32_t H_DEFAULT = 1280;

// 运行时 Tiling (Host→Kernel 传递)
struct MhcPostTiling {
    uint32_t n0;          // dim 0 大小
    uint32_t blockNum;    // 实际使用核数
    uint32_t bTile;       // = 1
    uint32_t cTile;       // = 64
    uint32_t h;           // h 维度大小
    uint32_t n1;          // dim 1 大小
};
// 总计 24 字节，一次 aclrtMemcpy 搬运
```

**设计理由**: Kernel 侧通过 `GetBlockIdx()` 和 `blockNum` 自行计算 `n1Start_/n1End_`，避免 Host 侧为每核预计算不同的 tiling 数据。

---

## 13. 架构合规性

| 检查项 | 状态 | 说明 |
|--------|------|------|
| NpuArch = DAV_2201 | 符合 | Ascend910B2 |
| 不使用 RegBase API | 符合 | 仅 DAV_3510 支持 |
| 不使用 Blaze API | 符合 | 仅 DAV_3510 支持 |
| 不使用 Cube (L0A/L0B/L0C) | 符合 | Vector 路线 |
| 核数动态获取 | 符合 | `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` |
| 禁止硬编码 blockDim | 符合 | blockNum 动态计算 |
| 禁止硬编码 UB 大小 | 符合 | 通过 TQue/TBuf 自动管理 |
| 无跨核通信 | 符合 | 各核独立处理 n1 子范围 |
| API 黑名单合规 | 符合 | 无禁止 API 使用 |
| TPipe/TQue 模式 | 符合 | 标准 SIMD/MemBase 模式 |
| `__global__ __vector__` 入口 | 符合 | Kernel 声明正确 |

---

## 14. 文件结构

```
operators/mhc_post/
├── CMakeLists.txt                       # 构建配置 (LANGUAGES ASC CXX)
├── run.sh                               # 编译+运行脚本
├── op_kernel/
│   ├── mhc_post_tiling.h                # Tiling 常量 + 结构体
│   ├── mhc_post_kernel_decl.h           # Kernel 声明 (extern "C" __global__ __vector__)
│   └── mhc_post_kernel.asc              # Kernel 实现 (TPipe + TQue + Vector)
├── op_host/
│   ├── mhc_post.asc                     # Host 侧: aclInit → KernelCall → main
│   └── data_utils.h                     # 文件读写工具
├── scripts/
│   ├── gen_data.py                      # 测试数据生成
│   ├── golden.py                        # PyTorch 参考实现
│   ├── verify_result.py                 # MERE/MARE 精度验证
│   └── test_torch.py                    # PyTorch 功能测试
└── docs/
    ├── DESIGN.md                        # 本文档
    ├── PLAN.md                          # 开发计划
    ├── precision/summary.txt            # 精度验收
    └── perf/round_NNN/                  # 性能数据归档
```
