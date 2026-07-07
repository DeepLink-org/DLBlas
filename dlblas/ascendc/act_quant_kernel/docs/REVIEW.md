# act_quant_kernel 审查报告

---

## Round 0 审查报告（Step 4 初审）

- **审查日期**: 2026-06-30
- **审查人**: Reviewer (AscendC Code Review Expert)
- **芯片**: Ascend910B2 (DAV_2201)
- **CANN**: 9.0.0
- **判定**: **PASS**
- **总分**: **92 / 100**

---

## 审查概要

| 维度 | 满分 | 得分 | 判定 |
|------|------|------|------|
| 1. 编译验证 | 10 | 10 | PASS |
| 2. 架构合规 | 15 | 15 | PASS |
| 3. 编码规范 | 15 | 13 | PASS |
| 4. 性能优化 | 20 | 17 | PASS |
| 5. 测试覆盖 | 15 | 13 | PASS |
| 6. 精度验证 | 10 | 9 | PASS |
| 7. 文档 | 15 | 15 | PASS |
| **总计** | **100** | **92** | **PASS** |

---

## 维度 1：编译验证（10/10）

### 1.1 独立编译成功（7/7）

独立执行 `cmake .. && make -j4`，编译通过，无错误。

- **编译器**: `/usr/local/Ascend/cann-9.0.0/bin/bisheng`
- **CMake 配置**: `find_package(ASC REQUIRED)` + `LANGUAGES ASC CXX`，正确
- **目标架构**: `--npu-arch=dav-2201`，与 Ascend910B2 匹配
- **产物**: `act_quant_kernel` (可执行文件) + `libact_quant_kernel_ops.so` (共享库)

### 1.2 无代码级警告（3/3）

编译过程未产生任何 warning。

---

## 维度 2：架构合规（15/15）

### 2.1 TPipe/TQue 模式（3/3）

- 使用 `AscendC::TPipe` 管理流水线
- 使用 `TQue<QuePosition::VECIN, 1>` 和 `TQue<QuePosition::VECOUT, 1>` 管理数据队列
- 使用 `TBuf<QuePosition::VECCALC>` 管理计算缓冲区
- 完全遵循 SIMD/MemBase 路线（DAV_2201 AscendC Pipeline API）

### 2.2 入口属性正确（3/3）

```cpp
extern "C" __global__ __vector__ void act_quant_kernel_kernel(
    GM_ADDR x, GM_ADDR q, GM_ADDR s, GM_ADDR tiling)
```

- `extern "C"`：C 链接
- `__global__`：全局入口
- `__vector__`：VectorCore 执行
- `GM_ADDR` 参数：全局内存地址

### 2.3 定义顺序正确（3/3）

Init → Process → CopyIn → Compute → CopyOut 的顺序符合 AscendC 最佳实践。

### 2.4 内存管理配对（3/3）

| AllocTensor | FreeTensor | 位置 |
|-------------|------------|------|
| `inQueueX.AllocTensor<T>()` | `inQueueX.FreeTensor(inData)` | CopyIn → Compute |
| `outQueueQ.AllocTensor<uint8_t>()` | `outQueueQ.FreeTensor(qLocal)` | Compute → CopyOut |
| `outQueueS.AllocTensor<float>()` | `outQueueS.FreeTensor(sLocal)` | Compute → CopyOut |

全部配对正确，无内存泄漏。

### 2.5 数据流完整（3/3）

CopyIn (GM→UB) → Compute (UB) → CopyOut (UB→GM)，数据流闭环完整。EnQue/DeQue 配对正确：
- CopyIn: AllocTensor → DataCopyPad → EnQue
- Compute: DeQue → 计算 → EnQue(outputs) → FreeTensor(input)
- CopyOut: DeQue → DataCopyPad → FreeTensor

---

## 维度 3：编码规范（13/15）

### 3.1 矢量 API（3/4）

**通过项**：
- `Cast<T, T2>()`：bf16→fp32 升精度转换，使用正确
- `Abs<T>()`：元素级绝对值，使用正确
- `ReduceMax<T>()`：逐行归约，使用 Level 2 API，正确
- `Muls<T>()`：标量广播乘法，使用正确
- `Mins<T>()` / `Maxs<T>()`：元素级 clamp，使用正确
- `DataCopyPad()`：32B 对齐搬运，使用正确

**扣分项（-1）**：
- fp32→fp8_e4m3fn 转换使用**逐元素标量循环**（`GetValue` + `SetValue`）：
  ```cpp
  for (uint32_t j = 0; j < gs; j++) {
      float fval = castWork.GetValue(j);
      uint8_t fp8val = fp32_to_fp8_e4m3fn(fval);
      qLocal.SetValue(g * gs + j, fp8val);
  }
  ```
  虽然是 DAV_2201 不支持硬件 fp8 Cast 的固有限制，但标量循环模式打破了周围的向量化风格，且性能影响显著。建议探索查表法（LUT）或批量位操作优化。

### 3.2 API 约束满足（4/4）

- 未使用被禁止的 `GlobalTensor::SetValue()` / `GlobalTensor::GetValue()`
- 数据搬运使用 `DataCopyPad`（支持非对齐数据），符合规范
- 未使用 `DataCopy`（会对齐限制严格的场景报错），策略正确

### 3.3 数据对齐（4/4）

- `calcGroupSizeAlign()` 计算 32B 对齐的 group size
- `DataCopyPad` 自动处理非对齐场景
- 测试覆盖了 group_size=2 的非对齐场景，验证通过

### 3.4 命名规范（3/3）

- 类名: `KernelActQuant`（PascalCase）
- 函数: `CopyIn`, `Compute`, `CopyOut`（PascalCase）
- 变量: `coreStartGroup`, `tileGroups`（camelCase）
- 常量: `FP8_E4M3FN_MAX`, `REDUCE_BUF_SIZE`（UPPER_SNAKE）
- Tiling 结构体: `ActQuantTiling`（PascalCase）
- 命名一致，符合 C++/AscendC 社区惯例

---

## 维度 4：性能优化（17/20）

### 4.1 动态硬件参数（4/4）

- 核数通过 `aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum)` 动态获取
- `blockIdx` 通过 `AscendC::GetBlockIdx()` 获取
- `tileGroups` 根据 `groupSize` 动态计算
- **Grep 验证通过**：无硬编码 `blockDim = N` / `blockIdx = N`

### 4.2 多核并行（4/4）

- 沿 groups 维度均匀切分：`coreGroups = ceil(numGroups / 48)`
- 空闲核正确跳过：`if (startGroup >= numGroups) startGroup = numGroups; coreGroups = 0;`
- 负载均衡良好：各核分配的 group 数相差不超过 1

### 4.3 流水线/双缓冲（3/4）

- `inQueueX` 和 `outQueueQ` 配置了 `DOUBLE_BUFFER`（BUFFER_NUM=2）
- EnQue/DeQue 机制实现 MTE 搬运与 Vector 计算的异步解耦

**扣分项（-1）**：
- 当前 `Process()` 循环采用串行模式：`CopyIn(tile_i) → Compute(tile_i) → CopyOut(tile_i)`
- 虽然 TQue 机制在 EnQue/DeQue 层面实现了基本的异步搬运，但 tile 级别的流水线重叠（前一个 tile 的 CopyOut 与后一个 tile 的 CopyIn 并行）未显式利用
- 建议：对于 performance-critical 场景，可将 Process 改造为分离式流水线，让 MTE 搬运和 Vector 计算在不同 tile 间真正重叠

### 4.4 同步策略（4/4）

无冗余 PipeBarrier/SetFlag/WaitFlag 调用。全部同步通过 TQue 的 EnQue/DeQue 机制完成，策略高效。

逐项依赖分析：

| 操作 | 输入来源 | 输出去向 | 同步机制 | 判定 |
|------|---------|---------|---------|------|
| DataCopyPad (CopyIn) | GM | inQueueX | EnQue → DeQue | 正确 |
| DeQue (Compute 输入) | inQueueX | castWork | 隐式等待 | 正确 |
| EnQue qLocal (Compute 输出) | qLocal (UB) | outQueueQ | EnQue → DeQue | 正确 |
| EnQue sLocal (Compute 输出) | sLocal (UB) | outQueueS | EnQue → DeQue | 正确 |
| DeQue qLocal (CopyOut) | outQueueQ | GM | 隐式等待 | 正确 |
| DeQue sLocal (CopyOut) | outQueueS | GM | 隐式等待 | 正确 |

冗余率 = 0%，全部 barrier 均为必要。

### 4.5 计算效率与上板性能（2/4）

**通过项**：
- 主计算路径全部使用向量 API（Cast/Abs/ReduceMax/Muls/Mins/Maxs），无循环内逐行 API 调用
- 无重复 GM 读取

**扣分项（-2）**：
- fp32→fp8 标量转换占 **85.7%** 的总执行时间（perf round_001 数据）
- Task Duration = 66.9 us（47 核, 65K elements），理论瓶颈在 scalar 侧
- 原因：DAV_2201 不支持硬件 fp8 Cast，每元素需要 1 次 `GetValue` + 多次位操作 + 1 次 `SetValue`
- 虽有硬件限制，但当前实现未尝试缓解方案（如查表法用 Gather 指令、批量位操作向量化、预计算 LUT 等）

---

## 维度 5：测试覆盖（13/15）

### 5.1 测试数据生成（4/4）

`gen_data.py` 支持：
- 随机数据（seed=42 固定）
- 全零数据
- 极值数据（含 ±400, ±0.0001）
- NaN 数据框架预留

覆盖充分。

### 5.2 结果验证脚本（4/4）

`verify_result.py` 实现：
- x_q (fp8): uint8 逐元素比较，允许 1-ULP 差异
- x_s (fp32): `np.allclose(rtol=1e-5, atol=1e-6)`
- 差异超阈值时输出详情（位置 + 值）

精度标准明确，验证逻辑合理。

### 5.3 Level 0/1/2/3 覆盖（3/4）

已执行的测试用例（本次独立验证）：

| # | Elements | group_size | 模式 | 结果 |
|---|----------|-----------|------|------|
| 1 | 128 | 128 | random | PASS (0 ULP) |
| 2 | 16 | 2 | random | PASS (0 ULP) |
| 3 | 512 | 64 | random | PASS (0 ULP) |
| 4 | 2048 | 128 | random | PASS (3/2048 @ 1-ULP) |
| 5 | 1024 | 32 | random | PASS (0 ULP) |
| 6 | 2048 | 128 | zeros | PASS (0 ULP) |
| 7 | 2048 | 128 | extreme | PASS (2/2048 @ 1-ULP) |
| 8 | 16384 | 128 | random | PASS (9/16384 @ 1-ULP) |
| 9 | 65536 | 128 | random | PASS (20/65536 @ 1-ULP) |

- Level 0: cases 1, 2, 3 ✓
- Level 1: cases 4, 5, 8 ✓
- Level 2: cases 6, 7 ✓
- Level 3: case 9 ✓

9/9 测试全部通过。

**扣分项（-1）**：所有测试用例均使用 **bf16 输入**。fp16 输入路径（template instantiation `KernelActQuant<half>`）虽然代码支持但未独立测试。PyTorch 通路测试同样仅覆盖 bf16。

### 5.4 精度标准明确（2/3）

- fp8 (x_q): 1-ULP 以内 ✓
- fp32 scale (x_s): rtol=1e-5, atol=1e-6 ✓

**扣分项（-1）**：UE8M0 路径精度标准未定义（该路径代码框架预留但未实现，见 DESIGN.md §7.2 和 PLAN.md §8.4）

---

## 维度 6：精度验证（9/10）

### 6.1 FP32 (scale) 全用例 PASS（4/4）

所有 9 个测试用例中 `x_s` max_diff 均为 `0.00e+00`（精确匹配），远优于 rtol=1e-5 标准。

### 6.2 FP16 全用例 PASS（1/3）

**扣分项（-2）**：fp16 输入路径未进行独立精度验证。虽然 kernel 通过 `KernelActQuant<half>` 模板实例化支持 fp16，但：
- `gen_data.py` 仅生成 bf16 数据
- PyTorch 测试仅使用 `torch.bfloat16`
- 未验证 fp16 路径下 `CalcGroupSizeAlign`、`Cast<float, half>`、`DataCopyPad` 参数等是否正确

### 6.3 BF16 全用例 PASS（3/3）

所有 9 个 bf16 测试用例全部通过。1-ULP 差异率极低（最高 20/65536 ≈ 0.03%），且均为舍入方向的 1-ULP 边界差异。

---

## 维度 7：文档（15/15）

### 7.1 README.md 存在（3/3）

`README.md` 存在，内容组织清晰。

### 7.2 数学公式（3/3）

包含完整的算子语义公式（per-group absmax → scale → quantize），与 DESIGN.md 一致。

### 7.3 编译运行指南（3/3）

- 直调方式：`bash run.sh` 一键运行 + 手动分步指导
- PyTorch 调用方式：完整的 Python 示例代码
- 环境依赖明确：CANN 9.0.0 + Ascend910B2 + PyTorch + torch_npu

### 7.4 API 映射/约束（3/3）

参数表完整，关键技术决策说明清晰（SIMD/MemBase 路线、AR 模式 ReduceMax、软件 fp8 转换、Double Buffer）。

### 7.5 已知限制（3/3）

明确列出 3 条已知限制：
1. fp32→fp8 转换的 1-ULP 舍入差异
2. UE8M0 scale 格式未完整实现
3. fp16 输入路径未充分测试

与 PLAN.md §8.4 一致，诚实且准确。

---

## 设计合规检查（对照 DESIGN.md）

| DESIGN.md 要求 | 实现状态 | 判定 |
|---------------|---------|------|
| SIMD/MemBase 路线 (DAV_2201 AscendC Pipeline API) | TPipe + TQue + DataCopyPad | 一致 |
| AR 模式 ReduceMax (Level 2) | `AscendC::ReduceMax<float>()` | 一致 |
| per-group 独立处理 | `for (g = 0; g < tileGroups; g++)` 逐组循环 | 一致 |
| 多核按 groups 切分 | `blockIdx * coreGroups` 分割 | 一致 |
| Double Buffer (inQueueX, outQueueQ) | `pipe_->InitBuffer(..., DOUBLE_BUFFER, ...)` | 一致 |
| 32B 对齐 DataCopyPad | `calcGroupSizeAlign()` + `DataCopyPad` | 一致 |
| fp8 转换：优先向量 Cast，兜底软件 | 软件实现 (`fp32_to_fp8_e4m3fn`) | 一致（已验证向量 Cast 不支持 DAV_2201） |
| UE8M0 路径 | Tiling 结构体预留 `scaleUe8m0` 字段，kernel 未实现 | 部分一致（DESIGN.md 已说明需编译验证，PLAN.md 列为未实现） |
| Broadcast 除（BinaryRepeatParams）| 改为逐组 `Muls` + `invScale` | 功能等价，略有偏差 |

无严重偏离。UE8M0 的未实现状态已在文档中明确标记。

---

## 问题列表

### 必须修复（阻塞项）

无。

### 建议修复（非阻塞）

| # | 严重度 | 维度 | 问题描述 | 修复建议 |
|---|--------|------|---------|---------|
| S1 | 中 | 3.1 + 4.5 | fp32→fp8 标量循环占 85.7% 执行时间，性能瓶颈显著 | 探索查表法（预计算 256×256 项 LUT，用向量 Gather 指令替代逐元素转换）或批量位操作向量化 |
| S2 | 低 | 4.3 | Double Buffer 虽已配置但 tile 级流水线串行，未充分隐藏 MTE 延迟 | 将 Process() 改造为分离式流水线：预启动第一个 tile 的 CopyIn，然后在循环中 Compute(tile_i) + CopyOut(tile_i) 与 CopyIn(tile_{i+1}) 重叠 |
| S3 | 低 | 5.3 + 6.2 | fp16 输入路径未独立测试 | 扩展 gen_data.py 支持 fp16 数据生成，增加至少 3 个 fp16 测试用例 |
| S4 | 低 | 3.1 | UE8M0 功能声明但未实现 | 在 README 中更突出地标注 UE8M0 为"未实现/待开发"，或尽快完成实现 |

### 观察项（供参考）

| # | 描述 |
|---|------|
| O1 | `Gen_data.py` 的 `bf16_encode/bf16_decode` 仅在生成 bf16 数据时使用。如需支持 fp16，需扩展为支持两种编码方式 |
| O2 | `calcGroupSizeAlign` 对 bf16 和 fp16 均使用 `dsize=2`，因此对齐计算结果相同。若后续支持 int8 (dsize=1)，需注意对齐行为变化 |
| O3 | 当前 `ReduceMax` 使用 32KB 工作缓冲区（`REDUCE_BUF_SIZE = 32*1024`），对于未来更大的 group_size 可能需要调整 |
| O4 | `QueuePosition::VECIN` 的 `inQueueX` 在 Compute 阶段通过 `DeQue` 消费后通过 `FreeTensor` 释放，但该 buffer 在 CopyOut 完成后才可被 CopyIn 下一轮使用——当前串行模式没有冲突，但若实现 S2 的流水线重叠需注意队列深度管理 |

---

## 独立精度测试记录

本次审查独立执行了 9 个测试配置，全部通过。

| # | Elements | group_size | Mode | x_q result | x_s result |
|---|----------|-----------|------|-----------|-----------|
| 1 | 128 | 128 | random | PASS (0 ULP) | PASS (max_diff=0.00) |
| 2 | 16 | 2 | random | PASS (0 ULP) | PASS (max_diff=0.00) |
| 3 | 512 | 64 | random | PASS (0 ULP) | PASS (max_diff=0.00) |
| 4 | 2048 | 128 | random | PASS (3/2048 @ 1-ULP) | PASS (max_diff=0.00) |
| 5 | 1024 | 32 | random | PASS (0 ULP) | PASS (max_diff=0.00) |
| 6 | 2048 | 128 | zeros | PASS (0 ULP) | PASS (max_diff=0.00) |
| 7 | 2048 | 128 | extreme | PASS (2/2048 @ 1-ULP) | PASS (max_diff=0.00) |
| 8 | 16384 | 128 | random | PASS (9/16384 @ 1-ULP) | PASS (max_diff=0.00) |
| 9 | 65536 | 128 | random | PASS (20/65536 @ 1-ULP) | PASS (max_diff=0.00) |

PyTorch 通路测试：
| Test | Shape | dtype | result |
|------|-------|-------|--------|
| T1 | [1, 128] | bf16 | PASS (q_mismatch=0, s_max_diff=0.000000) |
| T2 | [8, 128] | bf16 | PASS (q_mismatch=1, s_max_diff=0.000000) |

---

## 审查结论

**PASS**（92/100，优秀，可直接合入）

算子代码质量整体优秀：
- 架构设计严格遵循 DESIGN.md 的 SIMD/MemBase 路线
- 内存管理、同步策略、多核切分无缺陷
- 精度达标（9/9 配置全通过，bf16 输入）
- 编译独立验证通过
- 文档完整、诚实（已知限制清晰标注）

建议在后续迭代中补充 fp16 独立测试和 fp8 标量转换的性能优化。
