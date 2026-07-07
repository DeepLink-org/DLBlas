# head_compute_mix_bwd 算子架构设计

## 1. 环境与架构信息

| 项目 | 值 |
|------|-----|
| 芯片型号 | Ascend910B2 |
| NpuArch | `DAV_2201` (`__NPU_ARCH__=2201`, `--npu-arch=dav-2201`) |
| CANN 版本 | 9.0.0 |
| UB 容量 | 192 KB (196608 bytes) |
| L1 容量 | 512 KB |
| L0C 容量 | 128 KB |
| 技术路线 | **SIMD/MemBase（通用路线）** |

### 1.1 路线决策

| 决策因子 | 值 | 结论 |
|---------|-----|------|
| 目标架构 | DAV_2201 | 非 DAV_3510，不适用 RegBase / Blaze 路线 |
| 算子类型 | Broadcast + Elementwise + Reduction 融合 | 无 Cube/MatMul 计算 |
| 路由结果 | **SIMD/MemBase 通用路线** | 使用 AscendC Vector API (`Mul`/`Muls`/`Add`/`Adds`/`Sigmoid`) |

---

## 2. 算子数学定义

### 2.1 输入输出规格

| 张量 | Shape | dtype | 角色 |
|------|-------|-------|------|
| `input_mix` | (n0, n1, mhc_mult) | float32 | 输入（前向激活值） |
| `mhc_scale` | (1,) | float32 | 输入（标量缩放因子） |
| `mhc_base` | (mhc_mult,) | float32 | 输入（per-channel 偏置） |
| `grad_out` | (n0, n1, mhc_mult) | float32 | 输入（上游梯度） |
| `grad_input_mix` | (n0, n1, mhc_mult) | float32 | 输出（input_mix 梯度） |
| `grad_mhc_scale` | (1,) | float32 | 输出（mhc_scale 梯度） |
| `grad_mhc_base` | (mhc_mult,) | float32 | 输出（mhc_base 梯度） |

**标准配置**：n0=2, n1=1024, mhc_mult=4（固定）。

### 2.2 计算分解

```
① z            = input_mix * mhc_scale + mhc_base          # Broadcast + Elementwise
② sigmoid      = σ(z) = 1 / (1 + e^{-z})                   # Elementwise
③ sigmoid_grad = sigmoid * (1 - sigmoid)                   # Elementwise（sigmoid 导数）
④ grad_z       = grad_out * sigmoid_grad                    # Elementwise
⑤ grad_input_mix = grad_z * mhc_scale                      # Elementwise（标量广播）
⑥ temp         = grad_z * input_mix                         # Elementwise
⑦ grad_mhc_base  = sum(grad_z,  dims=(0,1))                # Reduction → (mhc_mult,)
⑧ grad_mhc_scale = sum(temp,     dims=(0,1,2))             # Global Reduction → (1,)
```

### 2.3 算子类型分解

| 阶段 | 类型 | 输入广播模式 | 适用 Tiling 模式 |
|------|------|-------------|-----------------|
| ① | Broadcast + Elewise | mhc_scale 标量广播, mhc_base per-row 广播 | Elewise OneDim |
| ②③④⑤⑥ | Elementwise | 无（同 shape 计算） | Elewise OneDim |
| ⑦ | Reduction (dim 0,1) | — | **ARA-FullLoad** [合轴后 R=2048, A=4] |
| ⑧ | Global Reduction | — | **Group Reduce**（全轴归约） |

---

## 3. 架构设计

### 3.1 总体方案：单 Kernel 融合 + 多核 Group Reduce

**核心策略**：将所有计算融合在一个 Kernel 内完成，避免中间结果落入 GM。归约部分采用 Group Reduce 模式：

```
                       ┌─────────────────────────────────┐
                       │       Host: Tiling 配置           │
                       │  (行切分 / UB 分块 / workspace)    │
                       └──────────────┬──────────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            │                         │                         │
       ┌────▼────┐              ┌────▼────┐              ┌────▼────┐
       │ Core 0  │              │ Core 1  │              │ Core N-1│
       │         │              │         │              │         │
       │ Load rows[0..K)       │ Load rows[K..2K)      │ Load ... │
       │ Elewise 全链路        │ Elewise 全链路         │          │
       │ Partial Reduce ⑦⑧    │ Partial Reduce ⑦⑧     │          │
       │ Store grad_input_mix  │ Store grad_input_mix   │          │
       │ Write partials → ws   │ Write partials → ws    │          │
       └────┬────┘              └────┬────┘              └────┬────┘
            │                        │                        │
            └────────────────────────┼────────────────────────┘
                                     │
                              SyncAll() 屏障
                                     │
                             ┌───────▼───────┐
                             │ Core 0: Merge  │
                             │ workspace →    │
                             │ final outputs  │
                             └───────────────┘
```

### 3.2 多核切分策略

将输入 (n0, n1, mhc_mult) 展平为 (total_rows, inner_dim) = (n0*n1, mhc_mult)，按行维度均分到各核。

| 参数 | 含义 | 计算公式 |
|------|------|---------|
| `total_rows` | 总行数 | `n0 * n1` (= 2048) |
| `inner_dim` | 每行元素数 | `mhc_mult` (= 4) |
| `total_elems` | 总元素数 | `total_rows * inner_dim` |
| `core_num` | 使用核数 | `min(ceil(total_elems * 4B / 4096B), avail_cores)` |
| `rows_per_core` | 每核基础行数 | `ceil(total_rows / core_num)` |
| `block_num` | 虚拟 block 数 | `ceil(total_rows / rows_per_core)` |
| `tail_rows` | 尾 block 行数 | `total_rows - (block_num - 1) * rows_per_core` |

**切分粒度约束**：每核至少 4KB 数据，确保值得开核。对于当前 shape (2048*4*4B=32KB)，推荐 `core_num=8`（每核 256 行 = 4KB）。

### 3.3 UB 切分与 Buffer 规划

#### 3.3.1 合轴（Reduction 侧）

按 reduction tiling 模式合轴：

| 步骤 | 操作 | 结果 |
|------|------|------|
| 标记 A/R | dim0→R, dim1→R, dim2→A | (R, R, A) |
| 合并相邻同类型 | R+R → R | (R=2048, A=4) |

合轴后为 **(R=2048, A=4)**，归约轴 R 在前，保留轴 A 在后。在单核视角下，这等效于 **ARA 模式**（A1=1, R=rows_per_core, A0=inner_dim）。

由于 inner_dim=4 极小且 `rows_per_core * inner_dim` 不超过 UB 容量，属于 **ARA-FullLoad**（全载）场景。

#### 3.3.2 UB Buffer 规划

由于计算链需要同时保有 `input_mix`（用于步骤⑥ `temp = grad_z * input_mix`）和 `grad_z`（用于步骤⑤），需至少 4 个数据级 buffer。

**Buffer 分配表**：

| Buffer 名 | 类型 | 元素数 | 字节数 | 用途 |
|-----------|------|--------|--------|------|
| `inQIm` | TQue (VECIN, DB=2) | `tile_rows * 4` | `tile_rows * 16` | 搬运 `input_mix` tile → 保留用于步骤⑥ |
| `inQGo` | TQue (VECIN, DB=2) | `tile_rows * 4` | `tile_rows * 16` | 搬运 `grad_out` tile → 步骤④相乘 |
| `bufZ` | TBuf | `tile_rows * 4` | `tile_rows * 16` | 主计算工作区：z→sigmoid→sigmoid_grad→grad_z |
| `outQOut` | TQue (VECOUT, DB=2) | `tile_rows * 4` | `tile_rows * 16` | 搬运 `grad_input_mix` 输出 + 中间临时空间 |
| `scaleBuf` | TBuf | 1 | 4 | 缓存 `mhc_scale` 标量 |
| `baseBuf` | TBuf | 8 | 32 | 缓存 `mhc_base` 广播模式（4+4 重复，对齐需要） |
| `sigmoidTmp` | TBuf | 动态 | Tiling 计算 | `Sigmoid` API 临时缓冲区 |
| `reduceTmp` | TBuf | 256 | 1024 | `ReduceSum` API 临时缓冲区 |
| `accBase` | TBuf | 4 | 16 | `grad_mhc_base` 跨 tile 累加器 (4 float32) |
| `accScale` | TBuf | 1 | 4 | `grad_mhc_scale` 跨 tile 累加器 (1 float32) |

**UB 总占用公式**：

```
ub_total = 2 * (2 * tile_rows * 16)    # inQIm/inQGo Double Buffer (per-side)
         + 1 * tile_rows * 16          # bufZ (TBuf)
         + 2 * tile_rows * 16          # outQOut Double Buffer (per-side)
         + sigmoid_tmp_size
         + 1024 + 32 + 4 + 16 + 4      # reduceTmp + baseBuf + scaleBuf + accBase + accScale
       = 80 * tile_rows + sigmoid_tmp_size + 1080
```

**约束**：`80 * tile_rows + sigmoid_tmp_size + 1080 <= 196608`

#### 3.3.3 Buffer 生命周期与复用

每个 UB tile 内的数据流：

```
时间线 (每个 UB tile 内):
  Load:   inQIm ← input_mix[tile]        (GM→UB, 保留用于步骤⑥)
          inQGo ← grad_out[tile]         (GM→UB, 保留用于步骤④)
          scaleBuf ← mhc_scale           (GM→UB, 一次性)
          baseBuf  ← mhc_base (8 elem)   (GM→UB, 一次性)
  ── Compute ──
  ① Muls(bufZ, bufIm, scaleVal, N)      → bufZ = input_mix * scale
  ② Add(broadcast: bufZ += baseBuf)      → bufZ = z
  ③ Sigmoid(bufZ, bufZ, sigmoidTmp, N)   → bufZ = sigmoid
  ④ Muls(outBuf, bufZ, -1.0f)           → outBuf = -sigmoid
  ⑤ Adds(outBuf, outBuf, 1.0f)           → outBuf = 1 - sigmoid
  ⑥ Mul(bufZ, bufZ, outBuf)              → bufZ = sigmoid * (1-sigmoid)  (sigmoid_grad)
  ⑦ Mul(bufZ, bufGo, bufZ)               → bufZ = grad_z
  ⑧ Muls(outBuf, bufZ, scaleVal)         → outBuf = grad_input_mix
  ⑨ Mul(bufIm, bufZ, bufIm)              → bufIm = grad_z * input_mix (= temp)
  ⑩ Accumulate:
      - 列归约 bufZ  → accBase += sum_by_col(grad_z)
      - 全归约 bufIm → accScale += sum(temp)
  ── Store ──
  outQOut ← outBuf                        (UB→GM grad_input_mix[tile])
```

**关键设计点**：
- `bufIm`（inQIm DeQue 后的 tensor）从 Load 保留到步骤⑨，不被覆盖，确保 `input_mix` 可用于最终的 `temp` 计算。
- `bufZ` 沿途覆写：`input_mix_copy → z → sigmoid → sigmoid_grad → grad_z`，无冲突。
- `outBuf` 复用：先用于 `1-sigmoid` 中间值，再用于 `grad_input_mix` 输出。

### 3.4 归约设计

#### 3.4.1 grad_mhc_base 归约：ARA-FullLoad + 手动列累加

合轴后 shape (rows_per_core, 4)，对行轴归约保留 4 列。

由于 `inner_dim=4` 极小，使用**手动逐行跨步累加**（比 `ReduceSum` API 更直接）：

```
partial_base[0..3] = {0}
for row in tile:
    partial_base[0] += grad_z[row * 4 + 0]
    partial_base[1] += grad_z[row * 4 + 1]
    partial_base[2] += grad_z[row * 4 + 2]
    partial_base[3] += grad_z[row * 4 + 3]
```

对于更大的 inner_dim（未来扩展），可改用 `ReduceSum<T, Pattern::Reduce::RA>`。

#### 3.4.2 grad_mhc_scale 归约：全局归约

全元素归约 (rows_per_core * 4) → 1 标量。使用 `ReduceSum` API：

```
ReduceSum(oneVal, bufIm, reduceTmp, count)  → oneVal[0] = sum(temp)
accScale += oneVal[0]
```

#### 3.4.3 跨核合并：Group Reduce

**Workspace 布局**：

```
per_core_slot = ALIGN_UP(5 * sizeof(float), 256) = 256 bytes
total_workspace = core_num * 256 bytes

每个 slot 内容:
  offset 0:  partial_base[0]   (float32, 4 bytes)
  offset 4:  partial_base[1]   (float32, 4 bytes)
  offset 8:  partial_base[2]   (float32, 4 bytes)
  offset 12: partial_base[3]   (float32, 4 bytes)
  offset 16: partial_scale     (float32, 4 bytes)
  offset 20..255: padding (0)
```

**合并流程**：

```
Phase 1（各核独立）:
  for each tile:
      elementwise 全链路
      累加 partial_base / partial_scale
  WritePartialsToWorkspace(blockIdx)

Phase 2:
  SyncAll()  ← 跨核内存屏障，确保所有核的 workspace 写入完成
  if blockIdx == 0:
      for b = 0..block_num-1:
          read workspace[b * 256 .. b * 256 + 20]
          sum into final_base[4], final_scale
      write final_base → GM grad_mhc_base
      write final_scale → GM grad_mhc_scale
```

**同步说明**：
- `SyncAll()` 在 Ascend C 中包含跨核内存屏障语义，确保所有核的 DMA 写入在 Core 0 读取前完成。
- 建议在 `WritePartialsToWorkspace()` 末尾额外调用 `PipeBarrier<PIPE_V>()` 以确保 UB→GM 的 DMA 在进入 SyncAll 前已提交。

---

## 4. API 映射表

| 计算步骤 | 数学表达 | AscendC API | 约束条件 |
|---------|---------|------------|---------|
| 标量乘 | `x * scalar` | `Muls<T>(dst, src, scalar, count)` | count 需 32B 对齐 |
| 标量加 | `x + scalar` | `Adds<T>(dst, src, scalar, count)` | count 需 32B 对齐 |
| 广播加 | `row + base[4]` | `Add<T>(dst, src0, src1, count, mask, repeatTime, BinaryRepeatParams)` | 块大小需 8 元素对齐；repeatTime <= 255 |
| Sigmoid | `1/(1+e^{-x})` | `Sigmoid<T>(dst, src, tmpBuf, count)` | float32 支持；tmpBuf 大小由 Host 侧 `GetSigmoidMaxMinTmpSize` 获取 |
| 逐元素乘 | `x * y` | `Mul<T>(dst, src0, src1, count)` | 同 shape |
| 归约求和 | `sum(x)` | `ReduceSum<T>(dst, src, tmpBuf, count)` | tmpBuf 需 >= 256 float32 |
| 数据搬运 | GM↔UB | `DataCopyPad<T>(dst, src, ...)` | 所有搬运使用 DataCopyPad，无 32B 对齐限制 |
| 管道同步 | — | `EnQue()` / `DeQue()` | TQue 标准模式 |
| Host tmpSize | — | `GetSigmoidMaxMinTmpSize(shape, dtypeBytes, ...)` | Host 侧调用 |
| Host 核数 | — | `PlatformAscendC::GetCoreNumAiv()` | 编译期不可用 |

### 4.1 关键 API 约束说明

| API | 约束 | 本算子适配 |
|-----|------|-----------|
| `DataCopyPad` | 无 32B 对齐要求 | 全程使用，避免非对齐风险 |
| `Sigmoid` | tmpBuf 需 Host 侧动态计算 | Tiling 阶段 `GetSigmoidMaxMinTmpSize` |
| `BinaryRepeatParams` | 广播块大小需 8 元素对齐 | mhc_base(4) 先展开为 8 元素模式 [b0,b1,b2,b3,b0,b1,b2,b3] |
| `Add` repeatTime | 单次调用 <= 255 | 超过时分批处理 |
| `ReduceSum` tmpBuf | 需 >= 256 float32 (=1024B) | 分配 256 元素 reduceTmp |

---

## 5. Tiling 参数结构

```cpp
struct HeadComputeMixBwdTilingData {
    // 多核切分
    uint32_t total_rows;        // n0 * n1
    uint32_t inner_dim;         // mhc_mult (=4)
    uint32_t core_num;          // 使用核数
    uint32_t rows_per_core;     // 每核基础行数
    uint32_t block_num;         // block 数 (= ceil(total_rows / rows_per_core))
    uint32_t tail_rows;         // 尾 block 行数

    // UB 切分
    uint32_t tile_rows;         // UB 单次处理行数
    uint32_t ub_loops;          // 每核 UB 循环次数 (= ceil(rows_per_core / tile_rows))

    // Sigmoid tmpBuf
    uint32_t sigmoid_tmp_size;  // Host 侧 GetSigmoidMaxMinTmpSize

    // Workspace (Group Reduce)
    uint32_t workspace_size;    // block_num * ALIGN_UP(5*4, 256)
    uint32_t ws_offset_stride;  // ALIGN_UP(5*4, 256)
};
```

### 5.1 Tiling 计算流程

```
1. total_rows = n0 * n1
2. total_elems = total_rows * inner_dim
3. core_num = min(max(1, ceil(total_elems * 4 / 4096)), avail_cores)
4. rows_per_core = ceil(total_rows / core_num)
5. block_num = ceil(total_rows / rows_per_core)
6. tail_rows = total_rows - (block_num - 1) * rows_per_core

7. sigmoid_tmp_size:
   GetSigmoidMaxMinTmpSize({total_elems}, sizeof(float), false, &max, &min)
   → use max

8. tile_rows:
   ub_budget = UB_SIZE - sigmoid_tmp_size - reduceTmp(1024B) - baseBuf(32B) - misc(60B)
   // 4 data buffers: inQIm(DB), inQGo(DB), bufZ, outQOut(DB)
   // DB = 2x per side, but only 1 side active. Effective: 2 + 2 + 1 + 2 = 7 x tile_bytes
   // Actually: inQIm has 2 slots (one for DMA, one for compute), same for inQGo and outQOut
   // Total data buffer UB = 2 * tile_bytes (inQIm) + 2 * tile_bytes (inQGo) + tile_bytes (bufZ) + 2 * tile_bytes (outQOut)
   // = 7 * tile_rows * 16
   tile_rows = floor(ub_budget / (7 * 16))
   tile_rows = min(tile_rows, rows_per_core)

9. ub_loops = ceil(rows_per_core / tile_rows)

10. ws_offset_stride = ALIGN_UP(5 * sizeof(float), 256) = 256
11. workspace_size = block_num * ws_offset_stride
```

### 5.2 UB 容量命名常量

```cpp
// 在 tiling 头文件中定义架构相关常量
constexpr uint32_t UB_SIZE_DAV_2201 = 196608;  // DAV_2201 UB: 192 KB
```

> 核数通过 `PlatformAscendC::GetCoreNumAiv()` 动态获取，UB 大小使用命名常量并在不同架构下替换。

---

## 6. 精度策略

### 6.1 精度标准

按 `ops-precision-standard` 判定：
- 输入/输出 dtype 均为 float32（纯浮点计算）
- 用户未声明商用标准
- **采用：浮点计算类社区标准** → rtol=1e-4, atol=1e-6

### 6.2 数值稳定性评估

| 计算 | 精度风险 | 策略 |
|------|---------|------|
| `z = input_mix * scale + base` | 低 | float32 直算 |
| `sigmoid(z)` | 低 | AscendC 内置 Sigmoid API |
| `sigmoid * (1-sigmoid)` | 低 | float32 直算 |
| `grad_z.sum()` 累加 2048 个 float32 | 中等 | 逐元素累加，2048 次加法的舍入误差在 1e-6 级别，可接受 |
| `temp.sum()` 累加 8192 个 float32 | 中等 | 同上，8192 次累加仍在社区标准内 |

> **结论**：float32 直算满足社区标准，无需混合精度或二分累加（Kahan summation）。

---

## 7. 边界条件与分支覆盖

| 场景 | 处理策略 |
|------|---------|
| n0=1, n1=1 | 最小 shape，total_rows=1 → 自动降为单核，单行处理 |
| n0/n1 任意变化 | total_rows 动态计算，tiling 自动适配 |
| mhc_mult=4（固定） | inner_dim 固定为 4，减少 UB 计算复杂度 |
| float32 only | 无 dtype 分支，仅 float32 路径 |
| 尾 block 行数 < rows_per_core | `ub_loops` 最后 tile 使用实际剩余行数 |
| add repeatTime > 255 | 分批循环调用 Add API |
| 非整除行数（core_num 不整除 total_rows） | tail_rows 机制，最后一个 block 处理较少行 |
| 有空闲核（block_num < core_num） | 仅 block_num 个 block 执行，多余核空转 |

---

## 8. 性能预估

### 8.1 计算量与数据量

| 操作 | 元素数 | 每条 FLOPs | 总 FLOPs |
|------|--------|-----------|---------|
| Mul (5次逐元素乘) | 8192 | 1 | 40960 |
| Muls (3次标量乘) | 8192 | 1 | 24576 |
| Add/Adds (3次加) | 8192 | 1 | 24576 |
| Sigmoid | 8192 | ~10 | 81920 |
| 归约累加 | 8192+2048 | 1 | 10240 |

**总 FLOPs ≈ 182K**，属于轻量计算。

**数据搬运**：
- GM→UB 读取：input_mix (32KB) + grad_out (32KB) + scale (4B) + base (32B) = ~64KB
- UB→GM 写入：grad_input_mix (32KB) + workspace (*) + final_scalars (~20B) = ~34KB
- **总搬运 ≈ 98KB**

### 8.2 瓶颈分析

- **计算密度**：182K FLOPs / 98KB = 1.86 FLOPs/Byte，属于**带宽瓶颈型**轻量算子
- **核心优化**：融合设计避免了中间结果的 GM 读写，已将带宽需求降至最低
- **多核扩展**：对于 32KB 级数据，8 核即可充分利用并行度；更多核边际收益递减

---

## 9. 设计决策记录

| 决策 | 选项 A | 选项 B | 选择 | 理由 |
|------|--------|--------|------|------|
| 架构路线 | SIMD/MemBase | RegBase | SIMD/MemBase | DAV_2201 不适用 RegBase |
| 核数 | 8 核 | 4 核 / 1 核 | 8 核 | 每核 256 行 = 4KB，平衡并行度与调度开销 |
| 归约方式 | ReduceSum API | 手动逐元素累加 | 混合：grad_mhc_base 手动（4列极小），grad_mhc_scale 用 API | 手动循环对于 4 列场景更高效 |
| Buffer 数 | 3 buffer | 4 buffer | 4 buffer（双缓冲有效 7 slot） | 需同时保有 input_mix 和 grad_z  |
| mhc_base 广播 | 单次 BinaryRepeatParams | 分步广播 | 8 元素展开 + BinaryRepeatParams | BinaryRepeatParams 要求块大小 8 对齐 |
| 精度 | float32 直算 | 混合精度/二分累加 | float32 直算 | 累加次数 ≤ 8192，社区标准内 |

---

## 10. 参考资料

- Ascend C Tiling Design — Elewise 场景路由: `ascendc-tiling-design/references/elewise/patterns.md`
- Ascend C Tiling Design — Broadcast 场景路由: `ascendc-tiling-design/references/broadcast/patterns.md`
- Ascend C Tiling Design — Reduction 场景路由: `ascendc-tiling-design/references/reduction/patterns.md`
- Ascend C Tiling Design — Group Reduce 算法: `ascendc-tiling-design/references/reduction/alg-group-reduce.md`
- Ascend C API — Sigmoid: `highlevel_api/lib/activation/sigmoid.h`
- Ascend C API — ReduceSum: `basic_api/kernel_operator_vec_reduce_intf.h`
- Ascend C API — Binary/Broadcast: `basic_api/kernel_operator_vec_binary_intf.h`
- NPU 架构 — Ascend910B2: `npu-arch/references/npu-hardware-params.md`
- 精度标准 — 浮点计算类社区标准: `ops-precision-standard/reference/float_compute_community.md`
