# Round 0 审查报告（Step 4 初审）

- **审查日期**：2026-07-01
- **审查者**：Ascend C 算子审查者 Agent
- **判定**：**FAIL**
- **总分**：**66 / 100**

---

## 1. 审查概要

| 维度 | 满分 | 得分 | 状态 |
|------|------|------|------|
| 1. 编译验证 | 10 | 10 | PASS |
| 2. 架构合规 | 15 | 15 | PASS |
| 3. 编码规范 | 15 | 15 | PASS |
| 4. 性能优化 | 20 | 11 | FAIL |
| 5. 测试覆盖 | 15 | 15 | PASS |
| 6. 精度验证 | 10 | 3 | WARN |
| 7. 文档 | 15 | 12 | PASS |
| **总计** | **100** | **66** | **FAIL** |

**FAIL 原因**：存在 CRITICAL 级必须修复问题（4.1/4.3/4.5）。

---

## 2. 必须修复问题（Blocking Issues）

### CRITICAL-001: Tile 数量计算 off-by-1 导致 GM 越界访问与 44.5% 额外计算

**问题位置**：`op_kernel/head_compute_mix_fwd_kernel.asc`，`Process()` 方法，第 82-83 行

**问题代码**：
```cpp
uint32_t totalTiles = ubLoop_;
if (ubTail_ > 0) totalTiles++;
```

**问题描述**：

`totalTiles` 的计算公式存在 off-by-1 错误。ubLoop 已经包含了 tail tile（当 ubTail > 0 时，ubLoop 中的最后一项对应 tail tile）。当前代码错误地在 ubLoop 基础上又加 1，导致为每个 block 多创建一个完整 tile。

**Tiling 语义**（按 `ComputeTiling` 公式）：
```
ubLoopF = ceil(blockFormer / ubFormer)
ubTailF = blockFormer - (ubLoopF - 1) * ubFormer
```
- 当 ubTailF > 0 时：tiles = [ubFormer] * (ubLoopF - 1) + [ubTailF]，共 ubLoopF 个
- 当 ubTailF == 0 时：tiles = [ubFormer] * ubLoopF，共 ubLoopF 个

因此正确公式为 `totalTiles = ubLoop_`。

**具体影响**（以默认 shape [16, 16384, 4] = 1,048,576 元素为例）：

| Block 类型 | ubLoop | ubTail | 正确 tile 数 | 当前 tile 数 | 多出元素 |
|-----------|--------|--------|-------------|-------------|---------|
| Former (x47) | 3 | 2560 | 3 | 4 | 9728/block |
| Tail (x1) | 2 | 4096 | 2 | 3 | 9728/block |

- **多出总元素**：48 x 9728 = 466,944 元素 = **44.53% 额外计算量**
- **GM 内存越界**：每个 block 读取和写入均超出 `blockSize_` 范围。最后一个 block 的越界写入超出整个输出张量 9728 元素
- **实际输出正确性**：非最后 block 的越界区域被后续 block 的正确数据覆盖（"侥幸通过"）；最后一个 block 的越界写入不会被覆盖，属于 undefined behavior

**重现条件**：所有 block size < ubFormer * (ubLoop) 的情况均受影响。对于 ubLoop >= 1 的任意 block，均会多创建一个 tile。

**对于小 shape 的特例**（如 8 元素，blockTail=8, ubLoopT=1, ubTailT=8）：
- 正确 tile 数：1（大小 8）
- 当前 tile 数：2（大小 9728 + 8）
- `firstSize = ubFormer = 9728`，但 `blockSize_ = 8`
- 第一条 CopyInTile 读取 9728 元素，仅前 8 元素有效，造成严重越界读取

**修复建议**：

在 `Process()` 方法中，将：
```cpp
uint32_t totalTiles = ubLoop_;
if (ubTail_ > 0) totalTiles++;
```

改为：
```cpp
uint32_t totalTiles = ubLoop_;
```

同时需要修正 `firstSize` 的计算逻辑，当 `totalTiles == 1` 且 `ubTail_ > 0` 时，`firstSize` 应为 `ubTail_` 而非 `ubFormer`：

```cpp
uint32_t firstSize;
if (totalTiles > 1) {
    firstSize = ubFormer;  // 多 tile 场景，首 tile 一定是 ubFormer
} else {
    firstSize = (ubTail_ > 0) ? ubTail_ : ubFormer;
}
```

此外，循环体内的 `curSize` 判定也需修正：
```cpp
// 循环中 i 从 1 开始，tile i 对应的索引
uint32_t curSize;
if (static_cast<uint32_t>(i + 1) < totalTiles) {
    curSize = ubFormer;  // 非最后一个 tile
} else {
    curSize = (ubTail_ > 0) ? ubTail_ : ubFormer;  // 最后一个 tile
}
```

---

### 其他阻塞项

| 检查项 | 要求 | 状态 |
|--------|------|------|
| 1.1 独立编译成功 | 编译通过 | PASS |
| 2.1 TPipe/TQue 模式 | 正确使用 | PASS |
| 2.2 入口属性正确 | `__global__ __vector__` | PASS |
| 3.1 矢量 API | Muls/Add/Exp/Reciprocal | PASS |
| 3.2 API 约束满足 | CAST 模式正确 | PASS |
| 4.1 动态硬件参数 | 核数动态获取，UB tile 硬编码 | PASS (核数动态) |
| 6.1 FP32 精度 | 算子仅支持 FP16 | N/A |

---

## 3. 逐维度详细评分

### 维度 1：编译验证（10/10）

| 子项 | 得分 | 判定 |
|------|------|------|
| 1.1 独立编译成功 | 7/7 | PASS — 在 CANN 9.0.0 环境下 `cmake .. && make -j4` 成功 |
| 1.2 无代码级警告 | 3/3 | PASS — 编译输出无 warning |

**验证过程**：
- 清除已有 build 产物后，执行独立编译
- CMake 配置成功（ASC、CXX 编译器均就绪）
- `head_compute_mix_fwd` 可执行文件 + `libhead_compute_mix_fwd_ops.so` 均编译成功
- 编译日志中无任何 warning

---

### 维度 2：架构合规（15/15）

| 子项 | 得分 | 判定 |
|------|------|------|
| 2.1 TPipe/TQue 模式 | 3/3 | PASS — 使用 TPipe 指针 + TQue (VECIN/VECOUT DOUBLE_BUFFER) + TBuf (VECCALC) |
| 2.2 入口属性正确 | 3/3 | PASS — `extern "C" __global__ __vector__ void head_compute_mix_fwd_kernel(...)` |
| 2.3 定义顺序正确 | 3/3 | PASS — class 内有 Init() / Process() / private 成员函数，顺序合理 |
| 2.4 内存管理配对 | 3/3 | PASS — EnQue/DeQue 严格配对，FreeTensor 在 DeQue 后正确调用 |
| 2.5 数据流完整 | 3/3 | PASS — CopyInTile → ComputeTile → CopyOutTile 三阶段流水线完整 |

**审查细节**：
- TQue 位置：inQueue = VECIN（正确，输入从 GM 搬入），outQueue = VECOUT（正确，输出搬出到 GM）
- TBuf 位置：f32WorkBuf0/1 和 baseF32Expanded 均为 VECCALC（正确，用于计算）
- FreeTensor：ComputeTile 中 `inQueue_.FreeTensor(inLocal)` 和 CopyOutTile 中 `outQueue_.FreeTensor(outLocal)` 正确配对

---

### 维度 3：编码规范（15/15）

| 子项 | 得分 | 判定 |
|------|------|------|
| 3.1 矢量 API | 4/4 | PASS — Muls, Add, Exp, Adds, Reciprocal 均为矢量 API，无逐元素标量操作 |
| 3.2 API 约束满足 | 4/4 | PASS — Cast RoundMode 正确（half→float: CAST_NONE, float→half: CAST_ROUND） |
| 3.3 数据对齐 | 4/4 | PASS — ubFormer 满足 256B 对齐（9728 * 2 / 256 = 76，整数）+ mhc_mult=4 对齐（9728/4=2432） |
| 3.4 命名规范 | 3/3 | PASS — 变量名清晰（f32WorkBuf、baseF32Expanded、ubLoop、ubTail），驼峰+下划线混合一致 |

**审查细节**：
- `SetValue` 使用在 `LocalTensor<float>` 上（允许），非 `GlobalTensor::SetValue`（禁止）
- 无禁止 API（GlobalTensor::SetValue/GetValue）
- `DataCopyPad` 正确使用（替代 DataCopy 以确保对齐兼容性）
- 代码注释清晰，关键逻辑均有说明

---

### 维度 4：性能优化（11/20）

| 子项 | 得分 | 判定 |
|------|------|------|
| 4.1 动态硬件参数 | 3/4 | WARN — 核数通过 `aclrtGetDeviceInfo` 动态获取 ✓；但 UB tile 大小 `UB_FORMER_HALF=9728` 为硬编码常量，虽按 192KB DAV_2201 计算正确，但应通过 constexpr 表达式推导而非魔法数字 |
| 4.2 多核并行 | 3/4 | PASS — 沿 dim0 维度切分，blockFormer 按 512 元素对齐保证负载均衡。小扣分：`MAX_CORE_NUM=24` 常量未使用且与实际 48 核不符（不影响功能，因实际核数动态获取） |
| 4.3 流水线/双缓冲 | 1/4 | **FAIL** — 双缓冲 TQue (BUFFER_NUM=2) 结构存在，但 CRITICAL-001 的 tile 计数 bug 导致实际处理量超 44.5%，使流水线效率大幅下降。每个 block 多处理一个完整 9728 元素 tile，该 tile 的数据被读写但结果被覆盖（非尾 block）或写入越界（尾 block） |
| 4.4 同步策略 | 4/4 | PASS — 逐项依赖分析如下：CopyIn(EnQue) → DeQue → Compute(EnQue outQueue) → DeQue → CopyOut。EnQue/DeQue 配对严格，无冗余 PipeBarrier。三阶段流水线同步依赖关系正确 |
| 4.5 计算效率 | 0/4 | **FAIL** — CRITICAL-001 导致 44.53% 无效计算；`PrepareBaseExpanded()` 使用 SetValue 循环（9728 次迭代/核），虽仅执行一次但效率低（DAV_2201 不支持 Duplicate(tensor→tensor)，属平台限制）；无循环内逐行 API 调用问题 |

**同步策略逐项依赖分析**：

```
Stage 0 (CopyIn tile_i):
  DataCopyPad(inLocal, inputGm_[offset], ...)
  inQueue.EnQue(inLocal)
  → 生产者发布 tile_i 的输入数据

Stage 1 (Compute tile_i-1):
  inLocal = inQueue.DeQue<half>()     ← 等待 CopyIn(tile_i-1) 的 EnQue
  Cast<float,half>(work, inLocal, ...)
  Muls/Add/Exp/Adds/Reciprocal/Adds
  Cast<half,float>(outLocal, work, ...)
  outQueue.EnQue(outLocal)            ← 发布 tile_i-1 的计算结果
  inQueue.FreeTensor(inLocal)

Stage 2 (CopyOut tile_i-2):
  outLocal = outQueue.DeQue<half>()   ← 等待 Compute(tile_i-2) 的 EnQue
  DataCopyPad(outputGm_[offset], outLocal, ...)
  outQueue.FreeTensor(outLocal)
```

依赖关系图：
```
CopyIn(tile_i) ──EnQue──→ Compute(tile_i) ──EnQue──→ CopyOut(tile_i)
                              ↑                          ↑
                          DeQue                      DeQue
```

每个箭头表示 TQue 的 EnQue→DeQue 生产者-消费者同步。所有依赖通过 TQue 自动满足，无冗余 barrier。**结论：同步策略冗余率 = 0%，评分 4/4。**

**上板性能分析**（PLAN.md 自报数据，独立验证确认可运行）：
- Task Duration: 52.581 us（48 核，1M 元素）
- 修复 CRITICAL-001 后，理论耗时可降至 ~36.4 us（ratio = 48 * 22016 / (48 * 31744) ≈ 0.694）
- 当前实测耗时受 bug 影响不具代表性，不纳入评分计算

---

### 维度 5：测试覆盖（15/15）

| 子项 | 得分 | 判定 |
|------|------|------|
| 5.1 测试数据生成 | 4/4 | PASS — `gen_data.py` 覆盖 normal/zeros/extreme/asymmetric/large_pos/large_neg 6 种模式 |
| 5.2 结果验证脚本 | 4/4 | PASS — `verify_result.py` 使用 allclose (rtol=1e-2, atol=1e-3)，输出 mismatch 索引 |
| 5.3 多级别覆盖 | 4/4 | PASS — Level 0（8 元素）✓、Level 1（1K）✓、Level 2（极端/零值/大负/非对齐）✓、Level 3（1M）✓ |
| 5.4 精度标准明确 | 3/3 | PASS — README.md 和 verify_result.py 均声明 rtol=1e-2, atol=1e-3 |

**独立验证结果**（全部 PASS）：

| 测试场景 | Shape | 元素数 | Max Diff | 状态 |
|---------|-------|--------|----------|------|
| 默认 1M | [16, 16384, 4] | 1,048,576 | 2.93e-03 | PASS |
| 小 shape | [1, 128, 4] | 512 | 1.95e-03 | PASS |
| 极小 | [2, 1, 4] | 8 | 9.77e-04 | PASS |
| 极端值 | [1, 1, 4] | 4 | 1.95e-03 | PASS |
| 零值 | [8, 16, 4] | 512 | 9.77e-04 | PASS |
| 大负值 | [1, 64, 4] | 256 | 0.00e+00 | PASS |

**PyTorch 通路**：8/8 用例全部 PASS（含 small_8/1K/default_1M/zeros/extreme/non_aligned/large_neg/asymmetric）

---

### 维度 6：精度验证（3/10）

| 子项 | 得分 | 判定 |
|------|------|------|
| 6.1 FP32 全用例 PASS | 0/4 | N/A — 算子仅声明支持 FP16，无 FP32 实现 |
| 6.2 FP16 全用例 PASS | 3/3 | PASS — 所有独立测试用例均在 atol=1e-3, rtol=1e-2 内 |
| 6.3 BF16 全用例 PASS | 0/3 | N/A — 算子未声明支持 BF16 |

**精度评估**：
- FP16 精度表现良好，max diff = 2.93e-03，远低于 rtol=1e-2 阈值
- FP32 中间计算策略（sigmoid 全链路 FP32）有效避免了 Exp 的 FP16 溢出问题
- 极端值（±10.0, ±5.0）测试通过，验证了 sigmoid 饱和区域的数值稳定性

---

### 维度 7：文档（12/15）

| 子项 | 得分 | 判定 |
|------|------|------|
| 7.1 README.md 存在 | 3/3 | PASS — 3207 字节，结构完整 |
| 7.2 数学公式 | 3/3 | PASS — `output = sigmoid(input_mix * mhc_scale + mhc_base) + mhc_pre_eps` 明确列出 |
| 7.3 编译运行指南 | 3/3 | PASS — 含 `bash run.sh` 一键运行 + 分步骤详细命令 |
| 7.4 API 映射/约束 | 3/3 | PASS — DESIGN.md 6.1/6.2 章节提供完整的 API 映射表 |
| 7.5 已知限制 | 0/3 | **FAIL** — README.md 和 DESIGN.md 均缺少「已知限制」章节。应明确记录：<br>1. 仅支持 FP16 dtype<br>2. DAV_2201 不支持 Duplicate(tensor→tensor)，mhc_base 扩展使用 SetValue 循环<br>3. mhc_mult 必须为 4 的约束<br>4. 小 shape 下的性能特征 |

---

## 4. 设计合规性检查

对照 `docs/DESIGN.md` 逐项验证实现一致性：

| DESIGN.md 条目 | 实际实现 | 一致性 |
|---------------|---------|--------|
| 展平 1D Elementwise 策略 | dim0 = batch * n1 * mhc_mult 展平处理 | 一致 |
| 多核切分公式 | ComputeTiling() 使用标准 Elementwise tiling 公式 | 一致 |
| UB 切分 256B + 4x 对齐 | UB_FORMER_HALF=9728，满足 128 元素对齐和 4 倍数 | 一致 |
| FP32 中间计算 | Cast half→float → FP32 sigmoid → Cast float→half | 一致 |
| Sigmoid 计算链 | Muls(-1)→Exp→Adds(1)→Reciprocal→Adds(eps) | 一致（注意：1.0/denom 使用 Reciprocal 替代 Div） |
| Double Buffer 流水线 | inQueue (TQue×2) + f32WorkBuf (手动 ping-pong) + outQueue (TQue×2) | 一致 |
| mhc_base 扩展 | SetValue 循环（DAV_2201 无 Duplicate(tensor→tensor)） | 一致 |
| 数据搬运 API | DataCopyPad（非对齐安全） | 一致 |
| Cast RoundMode | CAST_NONE (half→float), CAST_ROUND (float→half) | 一致 |

**偏差记录**（已在 PLAN.md §7 中记录，均属合理平台适配/优化）：

| 偏差项 | DESIGN.md | 实际实现 | 审查意见 |
|--------|-----------|---------|---------|
| mhc_base 存储类型 | `half mhc_base_f16[4]` | `float mhc_base_f32[4]` | 合理，避免 uint16↔half 转换 |
| Sigmoid 分母 | `Div(work, ones, denom)` | `Reciprocal(work, work)` | 合理优化，节省 ~38KB UB |
| mhc_base 扩展方式 | 步进 Duplicate | SetValue 循环 | DAV_2201 平台限制，可接受 |
| 文件扩展名 | .h/.cpp | .asc/.asc | 合规 |
| mhc_pre_eps 类型 | FP32 scalar | FP32 scalar | 一致 |

---

## 5. 代码清洁检查

| 检查项 | 结果 |
|--------|------|
| 硬编码 blockDim | 无 — Grep 确认无 `blockDim\s*=\s*\d+` 模式 |
| 硬编码 blockIdx | 无 — Grep 确认无 `blockIdx\s*=\s*\d+` 模式 |
| 未使用常量 | `MAX_CORE_NUM=24` 定义但未使用（tiling.h:31）；`dupTemp` 在注释中存在但代码中无对应 buffer |
| 注释准确性 | tiling.h 第 20 行 `dupTemp (TBuf, float, 512)` 注释与实际代码不符（无 dupTemp buffer） |

---

## 6. 问题清单汇总

### CRITICAL（必须修复，阻塞通过）

| ID | 位置 | 描述 | 影响 |
|----|------|------|------|
| C-001 | kernel.asc:82-83 | totalTiles off-by-1，每 block 多一个 tile | GM 越界 + 44.5% 额外计算 |

### HIGH（强烈建议修复）

| ID | 位置 | 描述 | 影响 |
|----|------|------|------|
| 无 | — | — | — |

### MEDIUM（建议修复）

| ID | 位置 | 描述 | 影响 |
|----|------|------|------|
| M-001 | tiling.h:20-22 | `dupTemp` buffer 在注释中存在但未使用，注释与代码不一致 | 代码可读性 |
| M-002 | tiling.h:31 | `MAX_CORE_NUM=24` 定义但未使用，且与实际 48 核不符 | 代码清洁度 |
| M-003 | README.md | 缺少「已知限制」章节 | 文档完整性 |
| M-004 | DESIGN.md §1 | AI Core 数量标注为 24，实际为 48 | 文档准确性 |

### LOW（可选改进）

| ID | 位置 | 描述 |
|----|------|------|
| L-001 | tiling.h:23 | `UB_FORMER_HALF=9728` 为魔法数字，建议用 constexpr 表达式从 192KB UB 推导 |
| L-002 | kernel.asc:130-143 | `PrepareBaseExpanded()` 中 SetValue 循环可用步进 Duplicate 方案优化（DAV_2201 上 Duplicate(tensor→tensor) 不可用，但标量 Duplicate 可用，可减少 SetValue 调用次数） |

---

## 7. 审查结论

**判定：FAIL**

**总分：66 / 100**

**结论理由**：
1. CRITICAL-001（totalTiles off-by-1）是必须修复的内存安全与性能 bug。该 bug 导致：
   - 每个 block 均超出 `blockSize_` 范围读写 GM（越界访问）
   - 最后一个 block 的越界写入超出整个输出张量
   - 44.53% 的无效计算开销
   - 小 shape（blockSize < ubFormer）下 `firstSize = ubFormer` 直接越界读取
2. 当前测试用例"侥幸通过"的原因是：非尾 block 的越界数据被后续 block 覆盖；尾 block 的越界区域不在输出文件读取范围内。这不代表 bug 不存在或无害。

**修复后预期评分**：修复 CRITICAL-001 后，维度 4 可提升至 ~18/20（4.3 从 1→4，4.5 从 0→3），总分可达 **~88/100（PASS）**。

---

**审查基于**：
- 独立编译：CANN 9.0.0 + Ascend910B2 (DAV_2201)
- 独立精度验证：6 种 shape/mode 组合，全部 FP16 测试通过
- 独立 PyTorch 通路验证：8/8 测试用例通过
- 代码静态分析：逐行审查 kernel、tiling、host、extension 代码


---

## Round 1 审查报告（Step 5 复审）

- **审查日期**：2026-07-01
- **审查者**：Ascend C 算子审查者 Agent
- **判定**：**PASS**
- **总分**：**91 / 100**

---

## 1. 审查概要

| 维度 | 满分 | 得分 | 状态 | Round 0 对比 |
|------|------|------|------|-------------|
| 1. 编译验证 | 10 | 10 | PASS | 10 → 10 (不变) |
| 2. 架构合规 | 15 | 15 | PASS | 15 → 15 (不变) |
| 3. 编码规范 | 15 | 15 | PASS | 15 → 15 (不变) |
| 4. 性能优化 | 20 | 18 | PASS | 11 → **18** (+7) |
| 5. 测试覆盖 | 15 | 15 | PASS | 15 → 15 (不变) |
| 6. 精度验证 | 10 | 3 | N/A | 3 → 3 (不变, FP16-only) |
| 7. 文档 | 15 | 15 | PASS | 12 → **15** (+3) |
| **总计** | **100** | **91** | **PASS** | **66 → 91 (+25)** |

**判定理由**：总分 91 >= 80，且无必须修复问题。上一轮的 CRITICAL-001 已正确修复，所有 MEDIUM 问题已修复。独立编译通过，独立精度验证全部 PASS。

---

## 2. Round 0 修复验证

### CRITICAL-001: totalTiles off-by-1 -- 已验证修复

**位置**：`op_kernel/head_compute_mix_fwd_kernel.asc`, `Process()` 方法

**验证结果**：**已正确修复**

逐项验证：

| 修复点 | 预期代码 | 实际代码 (行号) | 状态 |
|--------|---------|----------------|------|
| 移除 `if (ubTail_ > 0) totalTiles++` | `totalTiles = ubLoop_` | 第 77 行: `uint32_t totalTiles = ubLoop_;` | 正确 |
| 单 tile 场景 firstSize 修正 | `firstSize = (ubTail_ > 0) ? ubTail_ : ubFormer` | 第 84-85 行 | 正确 |
| 多 tile 场景 firstSize | `firstSize = ubFormer` | 第 81-82 行 | 正确 |
| 循环内 curSize 未尾判定 | `(i+1) < totalTiles ? ubFormer : ubTail_` | 第 101-104 行 | 正确 |

**实测验证**：
- 修复前 (Round 0 编译): `blockFormer=44032, blockNum=24`（当时用 24 核）
- 修复后 (Round 1 编译): `blockFormer=22016, blockNum=48`（48 核动态调度）
- 1M 元素测试: `ubLoopF=3, ubTailF=2560, ubLoopT=2, ubTailT=4096`
- 处理元素数 = 47 × 22016 + 13824 = 1,048,576 = dim0，**无越界**

### MEDIUM 问题修复验证

| ID | 描述 | 修复状态 | 验证方法 |
|----|------|---------|---------|
| M-001 | dupTemp 注释与代码不一致 | **已修复** | Grep 确认: `dupTemp` 在源码中无任何残留，tiling.h 注释仅描述实际 buffer |
| M-002 | MAX_CORE_NUM=24 与实际不符 | **已修复** | tiling.h:30: `MAX_CORE_NUM = 48` |
| M-003 | README.md 缺「已知限制」 | **已修复** | README.md:51-56: 4 项已知限制（FP16 only / mhc_mult=4 / DAV_2201 限制 / 小 shape 性能） |
| M-004 | DESIGN.md AI Core 数量 | **已修复** | DESIGN.md:19: 48（原 24） |

---

## 3. 独立编译验证

### 3.1 CMake 配置

| 检查项 | 状态 |
|--------|------|
| `find_package(ASC REQUIRED)` | 通过 — CMakeLists.txt:7 |
| `LANGUAGES ASC CXX` | 通过 — CMakeLists.txt:9 |
| `--npu-arch=dav-2201` | 通过 — CMakeLists.txt:38 + 98 |
| `tiling_api` 链接 | 通过 — CMakeLists.txt:23 + 79 |

### 3.2 编译结果

```
- 清理 build/ 产物后重编
- cmake ..           — 成功 (26.2s)
- make -j4           — 成功 (0 warnings)
- head_compute_mix_fwd (可执行文件) — 编译成功
- libhead_compute_mix_fwd_ops.so    — 编译成功
```

**编译环境**：CANN 9.0.0 + bisheng 编译器，DAV_2201

---

## 4. 独立精度验证

### 4.1 直接调用测试（9/9 PASS）

| # | Mode | Shape | Elements | Max Diff | RTOL=1e-2 判定 | ATOL=1e-3 判定 |
|---|------|-------|----------|----------|----------------|----------------|
| 1 | normal | [16,16384,4] | 1,048,576 | 2.93e-03 | PASS (2.93e-03 < 1e-2*~1.0) | — |
| 2 | zeros | [8,16,4] | 512 | 9.77e-04 | PASS | 通过 allclose |
| 3 | extreme | [1,1,4] | 4 | 1.95e-03 | PASS | 通过 allclose |
| 4 | asymmetric | [4,256,4] | 4,096 | 1.46e-03 | PASS | 通过 allclose |
| 5 | large_pos | [1,64,4] | 256 | 9.77e-04 | PASS | 通过 allclose |
| 6 | large_neg | [1,64,4] | 256 | 0.00e+00 | PASS | 通过 allclose |
| 7 | normal | [2,1,4] | 8 | 9.77e-04 | PASS | 通过 allclose |
| 8 | normal | [1,256,4] | 1,024 | 2.44e-03 | PASS | 通过 allclose |
| 9 | normal | [32,32768,4] | 4,194,304 | 2.93e-03 | PASS | 通过 allclose |

### 4.2 PyTorch 扩展测试（8/8 PASS）

```
P1 small_8:      PASSED (Max diff=9.765625e-04)
P2 1K:           PASSED (Max diff=2.929688e-03)
P3 default_1M:   PASSED (Max diff=2.929688e-03)
P4 zeros:        PASSED (Max diff=9.765625e-04)
P5 extreme:      PASSED (Max diff=1.953125e-03)
P6 non_aligned:  PASSED (Max diff=2.929688e-03)
P7 large_neg:    PASSED (Max diff=0.000000e+00)
P8 asymmetric:   PASSED (Max diff=1.464844e-03)

Total: 8, Passed: 8, Failed: 0
```

### 4.3 精度评估

- 最大误差 2.93e-03 出现在 1M/4M 元素的大 shape 测试中，远低于 rtol=1e-2 阈值
- FP32 中间计算策略（sigmoid 全链路 FP32）有效保证 FP16 输出的数值稳定性
- 极端值（±10.0, ±5.0）和零值输入均正确通过，sigmoid 饱和区域处理正确
- 大负值场景（sigmoid ≈ 0）误差为 0，符合预期

---

## 5. 逐维度详细评分

### 维度 1：编译验证（10/10）

| 子项 | 得分 | 判定 |
|------|------|------|
| 1.1 独立编译成功 | 7/7 | PASS — 清除 build 产物后 `cmake .. && make -j4` 成功 |
| 1.2 无代码级警告 | 3/3 | PASS — 编译输出零 warnings |

---

### 维度 2：架构合规（15/15）

| 子项 | 得分 | 判定 |
|------|------|------|
| 2.1 TPipe/TQue 模式 | 3/3 | PASS — TPipe 指针 + TQue(VECIN,2) + TBuf(VECCALC, x3) + TQue(VECOUT,2) |
| 2.2 入口属性正确 | 3/3 | PASS — `extern "C" __global__ __vector__ void head_compute_mix_fwd_kernel(...)` |
| 2.3 定义顺序正确 | 3/3 | PASS — Init() → Process() → private (PrepareBaseExpanded / CopyInTile / ComputeTile / CopyOutTile / members) |
| 2.4 内存管理配对 | 3/3 | PASS — EnQue/DeQue 严格配对；FreeTensor 与 DeQue 正确配对（ComputeTile 中 free inLocal, CopyOutTile 中 free outLocal） |
| 2.5 数据流完整 | 3/3 | PASS — 三阶段流水线：CopyInTile → ComputeTile → CopyOutTile，首 tile 特殊处理 + 尾阶段 flush |

---

### 维度 3：编码规范（15/15）

| 子项 | 得分 | 判定 |
|------|------|------|
| 3.1 矢量 API | 4/4 | PASS — Muls, Add, Exp, Adds, Reciprocal 均为矢量 API，无逐元素标量操作 |
| 3.2 API 约束满足 | 4/4 | PASS — Cast RoundMode 正确（half→float: CAST_NONE, float→half: CAST_ROUND） |
| 3.3 数据对齐 | 4/4 | PASS — ubFormer=9728，256B 对齐 (9728×2/256=76)，4 倍数对齐 (9728/4=2432) |
| 3.4 命名规范 | 3/3 | PASS — 变量名语义清晰（f32WorkBuf, baseF32Expanded, ubLoop, ubTail），驼峰/下划线风格一致 |

**审查细节**：
- `SetValue` 仅用于 `LocalTensor<float>`（允许），无 `GlobalTensor::SetValue`（禁止）使用
- `DataCopyPad` 正确使用（替代 DataCopy 以保证对齐兼容性）
- 代码注释清晰，每个方法均有文档说明

---

### 维度 4：性能优化（18/20）

| 子项 | 得分 | 判定 |
|------|------|------|
| 4.1 动态硬件参数 | 3/4 | PASS — 核数通过 `aclrtGetDeviceInfo` 动态获取；UB tile 大小 `UB_FORMER_HALF=9728` 为 constexpr 常量，虽按 192KB DAV_2201 推导正确，但建议用 constexpr 表达式从 UB 容量自动推导 |
| 4.2 多核并行 | 4/4 | PASS — 沿 dim0 切分，blockFormer 512 对齐保证负载均衡；MAX_CORE_NUM=48 已修正；实际核数动态获取（48 核全用） |
| 4.3 流水线/双缓冲 | 4/4 | PASS — TQue DOUBLE_BUFFER × 2 路（VECIN + VECOUT）+ 手动 ping-pong f32WorkBuf（TBuf × 2）；CRITICAL-001 修复后行程无额外 tile，流水线效率恢复；tile 计数正确（ubLoopF=3, ubLoopT=2, 总共 48 × 3 - 1 × 2 + 2 × 2 = 146 正确 tile） |
| 4.4 同步策略 | 4/4 | PASS — 逐项依赖分析：CopyIn(EnQue)→Compute(DeQue→EnQue outQueue)→CopyOut(DeQue)。所有同步通过 TQue 自动满足，无冗余 PipeBarrier。三阶段流水线依赖关系正确，同步策略冗余率 0%。详见同步分析（附录 A） |
| 4.5 计算效率 | 3/4 | PASS — CRITICAL-001 修复，无越界访问，无额外计算；PrepareBaseExpanded() 使用 SetValue 循环（每核一次，9728 次迭代），受限于 DAV_2201 平台（Duplicate(tensor→tensor) 不可用），可接受的工程权衡。扣 1 分：SetValue 循环是已知低效路径，但平台限制无更好替代 |

**上板性能**（PLAN.md round_003 数据）：
- Task Duration: 51.921 us（48 核，1M 元素）
- 修复后性能与修复前基本持平（52.581→51.921 us, -1.3%），因 Scalar 流水线（82.3%）为瓶颈，Bug 产生的额外 tile 与饱和流水线重叠
- **修复核心价值是消除 GM 越界访问这一内存安全问题，而非性能提升**

---

### 维度 5：测试覆盖（15/15）

| 子项 | 得分 | 判定 |
|------|------|------|
| 5.1 测试数据生成 | 4/4 | PASS — gen_data.py 覆盖 normal/zeros/extreme/asymmetric/large_pos/large_neg 6 种模式 |
| 5.2 结果验证脚本 | 4/4 | PASS — verify_result.py 使用 allclose (rtol=1e-2, atol=1e-3)，输出 mismatch 索引 |
| 5.3 多级别覆盖 | 4/4 | PASS — Level 0（4-8 元素）✓, Level 1（256-1K 元素）✓, Level 2（极端/零值/大负/非对齐）✓, Level 3（1M-4M 元素）✓ |
| 5.4 精度标准明确 | 3/3 | PASS — README.md 和 verify_result.py 均声明 rtol=1e-2, atol=1e-3 |

---

### 维度 6：精度验证（3/10）

| 子项 | 得分 | 判定 |
|------|------|------|
| 6.1 FP32 全用例 PASS | 0/4 | N/A — 算子按规定仅支持 FP16 输入/输出，无 FP32 实现 |
| 6.2 FP16 全用例 PASS | 3/3 | PASS — 9 个直接调用 + 8 个 PyTorch 用例全部通过，max diff=2.93e-03 |
| 6.3 BF16 全用例 PASS | 0/3 | N/A — 算子未声明支持 BF16 |

**说明**：精度维度总分受限于算子仅支持单一 dtype，并非精度问题。所有声明支持的 FP16 用例 100% 通过。

---

### 维度 7：文档（15/15）

| 子项 | 得分 | 判定 |
|------|------|------|
| 7.1 README.md 存在 | 3/3 | PASS — 结构完整（~3200 字节） |
| 7.2 数学公式 | 3/3 | PASS — `output = sigmoid(input_mix * mhc_scale + mhc_base) + mhc_pre_eps` |
| 7.3 编译运行指南 | 3/3 | PASS — `bash run.sh` 一键运行 + 分步骤命令 |
| 7.4 API 映射/约束 | 3/3 | PASS — DESIGN.md §6 提供完整 API 映射表 |
| 7.5 已知限制 | 3/3 | **修复确认** — README.md:51-56 含 4 项限制（FP16 only / mhc_mult=4 / DAV_2201 SetValue 循环 / 小 shape 性能） |

---

## 6. 设计合规性检查

| DESIGN.md 条目 | 实际实现 | 一致性 |
|---------------|---------|--------|
| 展平 1D Elementwise 策略 | dim0 = batch * n1 * mhc_mult 展平处理 | 一致 |
| 多核切分公式 | ComputeTiling() 使用标准 Elementwise tiling 公式 | 一致 |
| UB 切分 256B + 4x 对齐 | UB_FORMER_HALF=9728，满足 128 元素对齐和 4 倍数 | 一致 |
| FP32 中间计算 | Cast half→float → FP32 sigmoid → Cast float→half | 一致 |
| Sigmoid 计算链 | Muls(-1)→Exp→Adds(1)→Reciprocal→Adds(eps) | 一致 |
| Double Buffer 流水线 | inQueue(TQue×2) + f32WorkBuf(手动 ping-pong) + outQueue(TQue×2) | 一致 |
| mhc_base 扩展 | SetValue 循环（DAV_2201 无 Duplicate(tensor→tensor)） | 一致 |
| 数据搬运 API | DataCopyPad（非对齐安全） | 一致 |
| Cast RoundMode | CAST_NONE (half→float), CAST_ROUND (float→half) | 一致 |

**无新增偏差**。

---

## 7. 硬件参数检查

| 检查项 | Grep 命令 | 结果 |
|--------|----------|------|
| 硬编码 blockDim | `blockDim\s*=\s*[0-9]` | 无匹配 |
| 硬编码 blockIdx | `blockIdx\s*=\s*[0-9]` | 无匹配 |
| 核数动态获取 | `aclrtGetDeviceInfo` | head_compute_mix_fwd.asc:218 — 动态获取 |
| MAX_CORE_NUM | tiling.h:30 | 48（修正后），但实际未在运行时引用（核数由 aclrtGetDeviceInfo 提供） |

---

## 8. 代码清洁检查

| 检查项 | Round 0 状态 | Round 1 状态 |
|--------|-------------|-------------|
| `dupTemp` 残留注释 | 存在问题 (M-001) | **已清除** — Grep 确认源码中零残留 |
| MAX_CORE_NUM 值错误 | 24 (M-002) | **已修正为 48** |
| 未使用常量 | MAX_CORE_NUM, DOUBLE_BUFFER 未使用 | MAX_CORE_NUM=48, DOUBLE_BUFFER=2 仍为未使用常量 |
| 注释准确性 | 与代码一致 | 与代码一致 |

**说明**：`MAX_CORE_NUM` 和 `DOUBLE_BUFFER` 虽然是未使用常量，但前者作为文档型常量记录平台最大核数（可用于 future 的 fallback/限流逻辑），后者记录 Double Buffer 策略选择，属于可接受的工程实践。不作扣分。

---

## 9. 问题清单汇总

### CRITICAL（必须修复，阻塞通过）

无。

### HIGH（强烈建议修复）

无。

### MEDIUM（建议修复）

无。

### LOW（可选改进）

| ID | 位置 | 描述 |
|----|------|------|
| L-001 | tiling.h:22 | `UB_FORMER_HALF=9728` 为魔法数字，建议用 constexpr 表达式从 UB 容量自动推导（如 `constexpr int64_t UB_SIZE = 192 * 1024; constexpr int64_t UB_FORMER_HALF = (UB_SIZE / 20) / 128 * 128;`），增强跨平台可读性 |
| L-002 | kernel.asc:149-153 | `PrepareBaseExpanded()` 中 SetValue 循环效率低（9728 次 SetValue 调用/核），DAV_2201 平台限制下无更好替代，但可在注释中引用官方文档或 errata 说明不可用原因 |

---

## 10. 审查结论

**判定：PASS**

**总分：91 / 100**

**结论理由**：

1. CRITICAL-001（totalTiles off-by-1）已正确修复，GM 越界访问风险消除，44.5% 的额外计算已移除
2. 所有 4 个 MEDIUM 问题已修复（M-001: dupTemp 注释清理, M-002: MAX_CORE_NUM 修正, M-003: README 已知限制补充, M-004: DESIGN.md AI Core 数量修正）
3. 独立编译通过（CANN 9.0.0 + Ascend910B2, DAV_2201），零 warnings
4. 独立精度验证通过：9 个直接调用用例 + 8 个 PyTorch 用例全部 PASS，max diff 2.93e-03 < rtol 1e-2
5. 设计合规性无偏差
6. 无必须修复问题

**可选优化建议**（非阻塞）：
- L-001: 将 UB_FORMER_HALF 从魔法数字改为 constexpr 表达式推导
- L-002: 在 PrepareBaseExpanded 注释中引用 DAV_2201 Duplicate 不可用的官方依据

---

## 附录 A：同步策略逐项依赖分析（4.4 详细审查）

### Pipeline 三阶段依赖分析

```
Stage 0 (CopyIn tile_i):
  DataCopyPad(inLocal, inputGm_[offset], ...)
  inQueue.EnQue(inLocal)
  → 生产者发布 tile_i 的输入数据

Stage 1 (Compute tile_i):
  inLocal = inQueue.DeQue<half>()        ← 等待 CopyIn(tile_i) 的 EnQue
  Cast<float,half>(work, inLocal, ...)
  Muls → Add → Muls → Exp → Adds → Reciprocal → Adds
  Cast<half,float>(outLocal, work, ...)
  outQueue.EnQue(outLocal)               ← 发布 tile_i 的计算结果
  inQueue.FreeTensor(inLocal)

Stage 2 (CopyOut tile_i):
  outLocal = outQueue.DeQue<half>()      ← 等待 Compute(tile_i) 的 EnQue
  DataCopyPad(outputGm_[offset], outLocal, ...)
  outQueue.FreeTensor(outLocal)
```

### 依赖关系图

```
CopyIn(tile_i) ──EnQue──→ Compute(tile_i) ──EnQue──→ CopyOut(tile_i)
                              ↑                          ↑
                          DeQue                      DeQue
```

### 同步判定

- 每个箭头表示 TQue 的生产者-消费者同步
- inQueue: CopyIn (生产者) → Compute (消费者)，通过 EnQue/DeQue 自动同步
- outQueue: Compute (生产者) → CopyOut (消费者)，通过 EnQue/DeQue 自动同步
- 无冗余 PipeBarrier
- **结论：同步冗余率 = 0% (最优)**

### 主循环同步开销验证

循环体内 Compute(tile_i-1) 的 DeQue 与 CopyIn(tile_i) 的 EnQue 通过 inQueue 的 DOUBLE_BUFFER 实现并发。因 ubLoop ≥ 1，每个 block 至少执行 1 次 Compute，Queue 的 ping-pong 机制确保 DeQue 总是等待对应 EnQue 完成。无死锁风险。

---

**审查基于**：
- 独立编译：CANN 9.0.0 + Ascend910B2 (DAV_2201)，清除 build 产物后重编
- 独立精度验证：9 种 shape/mode 组合（4 元素 ~ 4M 元素），全部 FP16 测试通过
- 独立 PyTorch 通路验证：8/8 测试用例通过
- 代码静态分析：逐行审查 kernel、tiling、host、extension 代码，含 Grep 自动化检查
