## Round 0 审查报告（Step 4 初审）

- **审查日期**：2026-07-03
- **判定**：PASS
- **总分**：91 / 100
- **审查人**：ascendc-kernel-reviewer (独立审查)

---

### 审查概要

Sinkhorn 归一化算子的 AscendC 实现整体质量良好。独立编译验证通过（零警告），8 项 PyTorch 通路测试全部 PASS，FP32 精度表现出色（max_abs_diff=2.38e-07，max_rel_diff=7.76e-07）。主要扣分项为 FP16/BF16 未实现（设计范围外）和少数代码可读性问题。

---

### 1. 独立构建验证

| 检查项 | 状态 | 详情 |
|--------|------|------|
| CMake 配置 | PASS | 独立 `cmake ..` 配置成功，bisheng 编译器路径正确 |
| 编译 (sinkhorn_custom) | PASS | 无错误、无警告 |
| 编译 (libsinkhorn_ops.so) | PASS | 无错误、无警告 |
| Kernel 直调运行 | PASS | 47 核并行，tileBatch=22，执行成功 |
| PyTorch 通路 | PASS | 8/8 用例全部通过 |

---

### 2. 评分明细

#### 维度 1：编译验证（10 / 10）

| # | 检查项 | 得分 | 说明 |
|---|--------|------|------|
| 1.1 | 独立编译成功 | 7/7 | 完全清理 `build/` 后从 cmake 重新构建，两个 target 均成功 |
| 1.2 | 无代码级警告 | 3/3 | bisheng 编译器和 C++ 编译器均零警告 |

#### 维度 2：架构合规（15 / 15）

| # | 检查项 | 得分 | 说明 |
|---|--------|------|------|
| 2.1 | TPipe/TQue 模式 | 3/3 | 使用 `AscendC::TPipe` + `TQue<TPosition::VECIN,1>` + `TQue<TPosition::VECOUT,1>`，模式正确 |
| 2.2 | 入口属性正确 | 3/3 | `extern "C" __global__ __vector__ void sinkhorn_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)` 符合 Ascend C 规范 |
| 2.3 | 定义顺序正确 | 3/3 | Init() -> Process() -> private 成员，顺序符合规范 |
| 2.4 | 内存管理配对 | 3/3 | AllocTensor/EnQue/DeQue/FreeTensor 链条完整，inQueue 和 outQueue 操作配对正确 |
| 2.5 | 数据流完整 | 3/3 | CopyIn (DataCopyPad GM->UB) -> Compute (全部在 UB 内) -> CopyOut (DataCopyPad UB->GM)，数据流清晰完整 |

#### 维度 3：编码规范（15 / 15）

| # | 检查项 | 得分 | 说明 |
|---|--------|------|------|
| 3.1 | 矢量 API | 4/4 | 使用 Ascend C 矢量 API：ReduceMax、ReduceSum、Exp、Adds、Muls、Mul、DataCopyPad，无 PyTorch 退化 |
| 3.2 | API 约束满足 | 4/4 | - DataCopyPad 用于非对齐数据搬运（正确）<br>- LocalTensor::GetValue/SetValue 使用合规（黑名单仅限 GlobalTensor）<br>- DataCopyExtParams 参数传递正确 |
| 3.3 | 数据对齐 | 4/4 | 所有临时缓冲区偏移按 8-float (32B) 对齐，满足 Ascend C 对齐约束 |
| 3.4 | 命名规范 | 3/3 | PascalCase 类名、camelCase 变量名、UPPER_CASE 常量，风格一致 |

#### 维度 4：性能优化（19 / 20）

| # | 检查项 | 得分 | 说明 |
|---|--------|------|------|
| 4.1 | 动态硬件参数 | 4/4 | - 核数通过 `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` 动态获取<br>- 无硬编码 blockDim/blockIdx/TILE/UB 常量（grep 检查通过）<br>- `MAX_TILE_MATRICES=255` 是 DataCopyPad API 约束，不是硬件参数硬编码 |
| 4.2 | 多核并行 | 4/4 | - 沿 batch 维度均匀切分，核间负载均衡<br>- 尾核 tailBatch 正确处理非整除场景<br>- blockNum = usedCoreNum，无空闲核启动 |
| 4.3 | 流水线/双缓冲 | 4/4 | 单 Buffer 模式是刻意的设计选择（见 DESIGN.md Section 6.5）：数据量极小，仅一次 CopyIn/一次 CopyOut，双缓冲 setup 开销大于收益。判断合理。 |
| 4.4 | 同步策略 | 4/4 | **逐项依赖分析**：<br>- 同步点 1：`inQueue.DeQue<float>()` 等待 DataCopyPad DMA 完成，计算开始前数据已就绪 ✓<br>- 同步点 2：`outQueue.DeQue<float>()` 等待元素拷贝完成，DMA 写回前数据已同步 ✓<br>- 无冗余 PipeBarrier ✓<br>- 全部中间计算在 UB 内串行执行，无数据竞争 ✓ |
| 4.5 | 计算效率与上板性能 | 3/4 | - eps 加法使用逐元素 GetValue/SetValue 循环（Line 71-72）替代 `Adds()` API，与 DESIGN.md 设计的 `Adds(xBuf[m*16], xBuf[m*16], eps, 16)` 不一致<br>- 上板性能：487.27us，Scalar 占比 77.7%（主要因 work buffer 逐元素拷贝方案）。PLAN.md 已正确识别此瓶颈并给出优化方向。<br>- 扣 1 分：eps 加法存在可改进的微观效率问题 |

#### 维度 5：测试覆盖（15 / 15）

| # | 检查项 | 得分 | 说明 |
|---|--------|------|------|
| 5.1 | 测试数据生成 | 4/4 | `gen_data.py` 生成随机正态分布数据 + golden 参考输出 |
| 5.2 | 结果验证脚本 | 4/4 | `verify_result.py` 含 MERE/MARE/双随机性检查，NaN/Inf 检测 |
| 5.3 | Level 0 覆盖 | 4/4 | 8 个测试用例覆盖：单矩阵、全量 batch、随机 batch、小 batch、全零、极大正值、极负值、batch 不可整除 |
| 5.4 | 精度标准明确 | 3/3 | FP32: rtol=1e-5, atol=1e-6，在 verify_result.py 和 test_torch.py 中明确声明 |

#### 维度 6：精度验证（4 / 10）

| # | 检查项 | 得分 | 说明 |
|---|--------|------|------|
| 6.1 | FP32 全用例 PASS | 4/4 | 全部 8 个用例通过。max_abs_diff=2.38e-07, max_rel_diff=7.76e-07，远优于标准（rtol=1e-5, atol=1e-5） |
| 6.2 | FP16 全用例 PASS | 0/3 | **未实现**。算子明确设计为 FP32-only（DESIGN.md Section 6.4，torch 接入层 `TORCH_CHECK(x.scalar_type() == at::kFloat)`）。归档为已知限制。 |
| 6.3 | BF16 全用例 PASS | 0/3 | **未实现**。同 FP16。归档为已知限制。 |

> **备注**：FP16/BF16 未实现是设计决策而非遗漏。Sinkhorn 迭代归一化对精度敏感，FP16/BF16 尾数精度不足可能导致收敛性问题。当前 FP32 精度表现优异（误差在 1e-7 量级），若后续需要半精度支持，需评估混合精度策略（FP32 累加器）的可行性。

#### 维度 7：文档（13 / 15）

| # | 检查项 | 得分 | 说明 |
|---|--------|------|------|
| 7.1 | README.md 存在 | 3/3 | `README.md` 存在，内容全面：算子概述、文件结构、快速开始、技术要点、性能、精度 |
| 7.2 | 数学公式 | 3/3 | README 含算法步骤描述，DESIGN.md Section 1 有完整数学定义 |
| 7.3 | 编译运行指南 | 3/3 | `run.sh` 提供一键运行，README 含直调验证和 PyTorch 调用两种方式 |
| 7.4 | API 映射/约束 | 1/3 | API 映射表仅在 DESIGN.md Section 7，README 中未引用。建议在 README 添加 API 使用说明或指向 DESIGN.md 的链接。 |
| 7.5 | 已知限制 | 3/3 | 性能数据、精度实测、已知问题均在 README 和 PLAN.md 中记录。PLAN.md Section 8.4 详细列出 4 项已知偏离及原因。 |

---

### 3. 同步策略逐项依赖分析

| 代码位置 | 操作 | 依赖 | 同步机制 | 判定 |
|----------|------|------|----------|------|
| Line 35-38 | DataCopyPad + EnQue | 无 | - | OK |
| Line 39 | DeQue<float> from inQueue | DataCopyPad DMA 完成 | DeQue 语义保证 | OK |
| Lines 57-108 | 全部计算 (xData) | DeQue 完成 | 串行执行 | OK |
| Line 110-112 | outLocal 赋值 + EnQue | 计算完成 | 串行执行 | OK |
| Line 113 | FreeTensor(xData) | EnQue 完成 | 串行执行 | OK |
| Line 115-117 | DeQue + DataCopyPad | 元素拷贝完成 | DeQue 语义保证 | OK |
| Line 118 | FreeTensor(outData) | DataCopyPad 发射完成 | kernel 退出时 pipeline 自动排空 | OK |

**冗余率**：0%（无冗余 PipeBarrier）。同步策略精简且正确。

---

### 4. 设计合规检查

对照 `docs/DESIGN.md`：

| 设计要求 | 实现 | 一致性 |
|----------|------|--------|
| 技术路线：SIMD/MemBase | 使用 TPipe + Vector API | 一致 |
| 多核切分：沿 batch 维度 | Host 侧计算 tileBatch/tailBatch | 一致 |
| UB 策略：全载，单 Buffer | inQueue/outQueue buffer_num=1 | 一致 |
| Softmax：max-subtract 数值稳定 | ReduceMax -> Adds(-max) -> Exp -> ReduceSum -> Muls(1/sum) | 一致 |
| 列归约：逐列收集 + ReduceSum | colWork 收集 -> ReduceSum -> mult 存倒数 -> Mul 广播 | 一致 |
| 迭代: repeat=10, eps=1e-6 | 与 DESIGN.md 一致 | 一致 |

**微小偏离**：
- DESIGN.md 伪代码用 `Adds` 加 eps（`Adds(xBuf[m*16], xBuf[m*16], eps, 16)`），实际代码用逐元素 GetValue/SetValue 循环（Line 71-72）。功能等价，但可读性和效率略差。

---

### 5. 问题清单

#### 可选修复（非阻塞）

| ID | 严重度 | 位置 | 描述 | 修复建议 |
|----|--------|------|------|----------|
| P1 | LOW | `sinkhorn_kernel.asc:71-72` | eps 加法使用逐元素 GetValue/SetValue 循环，与 DESIGN.md 设计的 `Adds()` 不一致 | 替换为 `AscendC::Adds(xData[matBase], xData[matBase], eps, (int32_t)MATRIX_SIZE)` |
| P2 | LOW | `README.md` | 缺少 API 映射表引用 | 在 README 添加指向 DESIGN.md Section 7 的链接："API 映射详见 docs/DESIGN.md" |
| P3 | INFO | `sinkhorn_kernel.asc:17` | kernel 不检查 `blockIdx >= usedCoreNum`（当前 safe 因为 blockNum=usedCoreNum，但缺少防御性编程） | 在 Init 开头添加 `if (blockIdx >= tiling->usedCoreNum) { numMatrices = 0; totalElements = 0; return; }` |

#### 已知限制（非 bug，归档记录）

| ID | 描述 | 出处 |
|----|------|------|
| L1 | FP16/BF16 不支持（FP32-only 设计） | DESIGN.md Section 6.4 |
| L2 | repeat/eps 编译期常量，不支持运行时修改 | PLAN.md Section 8.4 |
| L3 | Scalar 占比 77.7%：work buffer 逐元素拷贝导致 | PLAN.md Section 8.3 |

---

### 6. 精度验证详情

**独立运行精度测试结果**（2026-07-03 独立执行）：

```
Verification:
  Total elements: 16384
  Max absolute diff: 2.384186e-07
  Mean absolute diff: 2.329688e-08
  Max relative diff: 7.758946e-07
  Mean relative diff: 1.127474e-07
  NaN count (output/golden): 0/0
  Inf count (output/golden): 0/0
  MERE check: PASS (1.127474e-07 < 1.22e-04)
  MARE check: PASS (7.758946e-07 < 1.22e-03)
  np.allclose(rtol=1e-5, atol=1e-6): PASSED
```

**PyTorch 通路全用例结果**：

| Test Case | Result | Max Diff |
|-----------|--------|----------|
| TC001 single_matrix | PASSED | 5.96e-08 |
| TC002 full_batch | PASSED | 2.38e-07 |
| TC003 random_8batch | PASSED | 1.19e-07 |
| TC004 small_batch | PASSED | 5.96e-08 |
| TC006 zeros | PASSED | 0.0 |
| TC007 large_positive | PASSED | 0.0 |
| TC008 large_negative | PASSED | 0.0 |
| TC009 non_divisible_batch | PASSED | 1.19e-07 |

---

### 7. 硬件参数检查

- `blockDim = N` 硬编码：**未发现**（grep 通过）
- `blockIdx = N` 硬编码：**未发现**（grep 通过）
- UB/TILE 大小硬编码：**未发现**

核数和 buffer 大小全部通过 `aclrtGetDeviceInfo` 或 Tiling 运行时计算。

---

### 8. 性能数据

独立采集的性能数据（msprof `sinkhorn_custom` 直调）：

| 指标 | 值 |
|------|-----|
| Task Duration | 487.27 us |
| AI Vector Core 数 | 47 |
| AIV Scalar 占比 | 77.7% |
| AIV Vector 占比 | 15.6% |
| AIV MTE 占比 | < 1% |

Scalar 占比高是已知问题（work buffer 逐元素拷贝方案），PLAN.md Section 8.3 已分析并给出优化方向。对于 64KB 数据量和 487us 的执行时间，该性能在功能验证阶段可接受。

---

### 9. 审查结论

**判定：PASS（91/100）**

无必须修复问题。所有关键检查项（1.1 编译、2.1 TPipe/TQue、2.2 入口属性、3.1 矢量 API、3.2 API 约束、4.1 动态硬件参数、6.1 FP32 精度）全部通过。

建议在后续迭代中修复 P1-P2 两项可选问题以提升代码质量和文档完整性。
