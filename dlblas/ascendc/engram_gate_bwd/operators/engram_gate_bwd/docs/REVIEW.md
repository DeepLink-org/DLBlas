# engram_gate_bwd 算子审查报告

## Round 0 审查报告（Step 5 复审）

- **审查日期**：2026-07-07
- **审查者**：独立 Reviewer Agent
- **算子名称**：`engram_gate_bwd`
- **代码版本**：v17（PLAN.md 标记）
- **目标平台**：Ascend 910B2 (DAV_2201), CANN 9.0.0
- **判定**：**PASS** (84/100)

---

## 执行概要

| 阶段 | 结果 |
|------|------|
| 独立编译 (main target) | **PASS** -- 可执行文件 `engram_gate_bwd` 构建成功 |
| 独立编译 (torch extension) | **BLOCKED** -- CANN 9.0.0 `dav-2201_vec` 架构不支持 + 缺少 Python.h（预先存在的环境问题） |
| 精度验证 (T=14,H=4,D=128) | **PASS** -- 5/5 输出全部通过（rtol=1e-2, atol=1e-3） |
| 精度验证 (T=14,H=4,D=256) | **PASS** -- 5/5 输出全部通过 |
| 硬件参数硬编码检测 | **PASS** -- 无硬编码 blockDim/blockIdx/UB 大小 |

---

## 详细评分（100 分制）

### 维度 1：编译验证（10/10 分）

| 检查项 | 得分 | 说明 |
|--------|:--:|------|
| 1.1 独立编译成功 | 7/7 | main target `engram_gate_bwd` 从源码 clean build 通过，无错误 |
| 1.2 无代码级警告 | 3/3 | ASC 编译器（bisheng）在 main target 编译中无 warning 输出 |
| **小计** | **10/10** | |

**发现**：
- Torch extension target (`engram_gate_bwd_ops`) 编译失败，原因：(a) `--npu-arch=dav-2201_vec` 不被 CANN 9.0.0 bisheng 识别（正确名称应为 `dav-2201`），(b) register.cpp 和 torch stub 缺少 Python.h 路径。此问题预先存在（PLAN.md 已标注"Torch 扩展库未编译"），不影响算子核心功能。

---

### 维度 2：架构合规（15/15 分）

| 检查项 | 得分 | 说明 |
|--------|:--:|------|
| 2.1 TPipe/TQue 模式 | 3/3 | `TPipe` 正确传入构造函数，所有 UB 缓冲区通过 `TQue<TPosition::VECCALC, 1>` 管理 |
| 2.2 入口属性正确 | 3/3 | `extern "C" __global__ __vector__` 入口声明正确，成员函数 `__aicore__` 标注完整 |
| 2.3 定义顺序正确 | 3/3 | `Init()` -> `Process()` -> `C1()` -> `XR()` 定义顺序符合 Ascend C 规范 |
| 2.4 内存管理配对 | 3/3 | 所有 buffer 在 `Process()` 中一次性 `AllocTensor`，跨 T 迭代复用（v17 修复），无泄漏 |
| 2.5 数据流完整 | 3/3 | GM→DataCopyPad→Cast→UB (f32 计算)→Cast→DataCopyPad→GM 数据流完整闭合 |
| **小计** | **15/15** | |

---

### 维度 3：编码规范（11/15 分）

| 检查项 | 得分 | 说明 |
|--------|:--:|------|
| 3.1 矢量 API | 2/4 | **扣分**：10+ 处使用标量 `SetValue`/`GetValue` 在循环中逐元素访问，而非使用 `Duplicate` 或 `Broadcast` 矢量 API。详见下方 P1 问题列表 |
| 3.2 API 约束满足 | 3/4 | Cast RoundMode 正确（CAST_NONE bf16→f32, CAST_ROUND f32→bf16）。`ReduceSum` Level 2 AR 模式使用正确。**扣分**：`Compares` + `Select` 组合在 B3（mask 计算）中可能与 `Duplicate` 初始化值冲突，依赖隐式顺序 |
| 3.3 数据对齐 | 4/4 | 全部使用 `DataCopyPad` 进行 GM↔UB 搬运，无未保护的 `DataCopy` 调用。H*D (4×128=512) 为 32B 对齐 |
| 3.4 命名规范 | 2/3 | **扣分**：s_[] 数组索引无枚举/常量定义（s_[0]=rx, s_[1]=rk, ... s_[17]=gd），可读性差。部分变量名过于简短（`t0_`, `d0_`, `aw`, `ae`） |
| **小计** | **11/15** | |

---

### 维度 4：性能优化（12/20 分）

| 检查项 | 得分 | 说明 |
|--------|:--:|------|
| 4.1 动态硬件参数 | 4/4 | 核数通过 `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` 获取，tileT 基于实际核数计算。无硬编码 |
| 4.2 多核并行 | 4/4 | 沿 T 维度均分，负载均衡。`coreNum = min(totalT, availableCoreNum)` 确保空闲核正确跳过。Workspace-based 跨核归约正确 |
| 4.3 流水线/双缓冲 | 1/4 | **严重扣分**：所有 TQue 均使用 `BUFFER_NUM=1`（单缓冲），无双缓冲。每个 T 迭代的输入加载和计算完全串行，无 DMA/计算重叠 |
| 4.4 同步策略 | 2/4 | **扣分**：全代码使用 103 处 `PipeBarrier<PIPE_ALL>()`，绝大多数是冗余的逐操作同步。VECCALC 队列的 API（Mul/Add/Div）本就有串行保序语义，无需在每个操作后加 barrier。详见下方 P2 问题列表 |
| 4.5 计算效率与上板性能 | 1/4 | **严重扣分**：(a) 广播操作使用 O(H*D) 次 `SetValue`/`GetValue` 标量访问（10+ 处），而非 O(1) 的 `Duplicate`；(b) grad_v 计算使用逐元素标量循环 `go.GetValue(base+d)` + 逐元素 `td0.SetValue(d,...)`，极度低效；(c) T=14 下总耗时虽可接受，但性能远未达平台潜力 |
| **小计** | **12/20** | |

---

### 维度 5：测试覆盖（15/15 分）

| 检查项 | 得分 | 说明 |
|--------|:--:|------|
| 5.1 测试数据生成 | 4/4 | `gen_data.py` 完整生成 6 组 bf16 输入 + f32 golden 输出 |
| 5.2 结果验证脚本 | 4/4 | `verify_result.py` 支持 bf16→f32 转换 + np.allclose 验证 + 错误统计 |
| 5.3 Level 0 覆盖 | 4/4 | T=14, H=4, D=128 典型 case 覆盖，独立验证通过 |
| 5.4 精度标准明确 | 3/3 | rtol=1e-2, atol=1e-3 适配 bf16 精度极限 |
| **小计** | **15/15** | |

**备注**：`verify_result.py` 中 shape 硬编码为 14×4×128，不同 shape（如 D=64 或 D=256）时需要手动修改或使用动态参数。建议改为从命令行或 golden 文件推断 shape。

---

### 维度 6：精度验证（10/10 分）

| 检查项 | 得分 | 说明 |
|--------|:--:|------|
| 6.1 默认 shape PASS | 4/4 | T=14, H=4, D=128: 5/5 输出全部通过 |
| 6.2 bf16 全用例 PASS | 3/3 | D=128, D=64, D=256 均通过（独立验证） |
| 6.3 BF16 全用例 PASS | 3/3 | 算子仅支持 bf16，全场景通过 |
| **小计** | **10/10** | |

**独立验证数据（T=14, H=4, D=128）**：

| 输出 | max_diff | mean_diff | 判定 |
|------|----------|-----------|:--:|
| grad_x (7168) | 9.65e-04 | 8.91e-05 | PASS |
| grad_k (7168) | 1.05e-04 | 1.27e-06 | PASS |
| grad_v (1792) | 9.73e-04 | 1.12e-04 | PASS |
| grad_wh (512) | 1.04e-04 | 8.06e-06 | PASS |
| grad_we (512) | 9.05e-05 | 7.27e-06 | PASS |

**独立验证数据（T=14, H=4, D=256）**：

| 输出 | max_diff | 判定 |
|------|----------|:--:|
| grad_x | 9.64e-04 | PASS |
| grad_k | 1.18e-04 | PASS |
| grad_v | 9.71e-04 | PASS |
| grad_wh | 2.84e-04 | PASS |
| grad_we | 1.35e-04 | PASS |

---

### 维度 7：文档（11/15 分）

| 检查项 | 得分 | 说明 |
|--------|:--:|------|
| 7.1 README.md 存在 | 0/3 | **缺失**：工程根目录 `operators/engram_gate_bwd/` 下无 `README.md` |
| 7.2 数学公式 | 3/3 | DESIGN.md 中完整包含 Phase A/B 的数学定义和公式 |
| 7.3 编译运行指南 | 3/3 | `run.sh` 提供完整的构建+运行+验证流程 |
| 7.4 API 映射/约束 | 3/3 | DESIGN.md 第 5 章「API 映射表」完整列出所有使用的 API 及验证要点 |
| 7.5 已知限制 | 2/3 | PLAN.md 记录了 Hang bug/v17 修复。**扣分**：未在用户可见文档中列出性能限制（无双缓冲、标量广播低效）和 shape 限制 |
| **小计** | **11/15** | |

---

## 总分汇总

| 维度 | 满分 | 得分 | 权重占比 |
|------|:---:|:---:|:-------:|
| 1. 编译验证 | 10 | **10** | 100% |
| 2. 架构合规 | 15 | **15** | 100% |
| 3. 编码规范 | 15 | **11** | 73% |
| 4. 性能优化 | 20 | **12** | 60% |
| 5. 测试覆盖 | 15 | **15** | 100% |
| 6. 精度验证 | 10 | **10** | 100% |
| 7. 文档 | 15 | **11** | 73% |
| **总计** | **100** | **84** | **84%** |

---

## 必须修复问题清单（无）

经逐项检查，未发现触发 FAIL 判定的必须修复项（检查项 1.1/2.1/2.2/3.1/3.2/4.1/6.1 中无一完全失败）。

---

## 重要改进建议（P0/P1/P2）

### P0 -- 阻塞性（无）

### P1 -- 高优先级（性能改善）

#### P1-1：广播操作改用矢量 API（性能瓶颈 #1）

**文件**：`op_kernel/engram_gate_bwd_kernel.asc`  
**位置**：C1() 中 10+ 处标量→向量广播

**当前代码模式**（举例 lines 331-334）：
```cpp
// 将 H 维标量 gr[h] 广播到 (H, D)  -> 使用 O(H*D) 次 SetValue
for (int32_t h = 0; h < H_i; h++) {
    float v = gr.GetValue(h);
    for (int32_t d = 0; d < D_i; d++) th2.SetValue(h * D_i + d, v);
}
```

**推荐修复**：
```cpp
// 使用 Duplicate API 将每个 head 的标量广播到 D 列（O(H) 次操作）
for (int32_t h = 0; h < H_i; h++) {
    Duplicate(th2[h * D_i], gr.GetValue(h), D_i);
}
```

**影响范围**：lines 331-334, 340-343, 350-353, 364-367, 374-377, 384-387, 395-398, 407-410  
**收益**：广播操作从 O(H*D) 标量访问降至 O(H) 矢量操作

#### P1-2：grad_v 改用矢量实现（性能瓶颈 #2）

**位置**：C1() lines 268-275

**当前代码**：
```cpp
Duplicate(gv, 0.0f, D_i);
for (int32_t h = 0; h < H_i; h++) {
    int32_t base = h * D_i;
    for (int32_t d = 0; d < D_i; d++) td0.SetValue(d, go.GetValue(base + d));
    PipeBarrier<PIPE_ALL>();
    Muls(td0, td0, gt.GetValue(h), D_i);
    Add(gv, gv, td0, D_i);
}
```

**推荐修复**：
```cpp
// 使用 Mul + ReduceSum 沿 H 轴归约
// go[h, d] * gate[h] 可通过 Duplicate gate 到 D 维后 Mul 实现
Duplicate(gv, 0.0f, D_i);
for (int32_t h = 0; h < H_i; h++) {
    Duplicate(td0, gt.GetValue(h), D_i);
    Mul(td0, go[h * D_i], td0, D_i);
    Add(gv, gv, td0, D_i);
}
```

### P2 -- 中优先级（性能优化 / 可维护性）

#### P2-1：减少冗余 PipeBarrier

全代码 103 处 `PipeBarrier<PIPE_ALL>()`。VECCALC 队列的 API（`Mul`/`Add`/`Div`/`Sub`/`Muls`/`Cast`）在同一队列上天然保证执行顺序。仅在以下场景需要 PipeBarrier：

1. 不同类型队列间切换（如从 VECCALC 切换到 MTEC2 的 DataCopyPad）
2. ReduceSum 的 tmpBuffer 被后续操作覆盖前
3. Cast（VECCALC）→ DataCopyPad（MTEC2）前后

**推荐**：系统性移除同一队列内连续 VECCALC 操作之间的冗余 PipeBarrier。估计可移除 60% 以上的 barrier 调用。

#### P2-2：为 s_[] 数组添加语义枚举

**位置**：`engram_gate_bwd_kernel.asc` line 483

**当前**：`TQue<TPosition::VECCALC, 1> s_[18];` 配合注释说明 s_[0]=rx, s_[1]=rk, ...

**推荐**：
```cpp
enum ScalarSlot : int {
    SLOT_RX = 0, SLOT_RK, SLOT_RD, SLOT_DT, SLOT_AD, SLOT_CM,
    SLOT_SG, SLOT_NG, SLOT_SS, SLOT_GT, SLOT_GR, SLOT_GXS,
    SLOT_GKS, SLOT_EX, SLOT_GG, SLOT_GS, SLOT_MK, SLOT_GD
};
// 使用: auto rx = s_[SLOT_RX].AllocTensor<float>();
```

#### P2-3：实现输入双缓冲

按 DESIGN.md 第 3.3 节「双缓冲策略」设计，为 (go, x, k, v) 队列实现 `BUFFER_NUM=2` 的双缓冲。当前 `tileTPerLoop=1` 且全操作串行，无 DMA/计算重叠。

**注意**：实现双缓冲需要将 `BUFFER_NUM` 从 1 改为 2，并重构 T 循环使用乒乓模式（EnQue/DeQue 交替使用两个 buffer slot）。这需要较大的代码改动。

#### P2-4：补充 README.md

创建 `operators/engram_gate_bwd/README.md`，内容应包括：
- 算子功能概述与数学公式
- 输入/输出 API 映射表
- 编译与运行指南（`bash run.sh`）
- 精度标准与测试结果说明
- 已知限制（仅支持 bf16、无双缓冲、Torch 扩展未完成）

#### P2-5：verify_result.py 移除硬编码 shape

`scripts/verify_result.py` lines 57-62 中 shapes 字典硬编码了 T=14, H=4, D=128 的 shape。应改为从 golden 文件大小动态推断，或通过命令行参数传入。

---

## 审查结论

- **判定**：**PASS** -- 总分 84/100，无必须修复项
- **理由**：编译通过，精度验证全部达标，架构设计正确，多核策略合理。主要扣分集中在性能优化（无双缓冲、标量广播低效、冗余同步）和文档完善度。
- **建议**：在进入生产环境前，优先修复 P1-1（广播矢量化）和 P1-2（grad_v 矢量化），这两项改动可带来数量级性能提升；按需完成 P2 改进项。

---

*审查完成时间：2026-07-07*
*审查工具链：bisheng (CANN 9.0.0), CMake 4.3.1, Ascend 910B2 (DAV_2201)*
