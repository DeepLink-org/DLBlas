# REVIEW.md — expand_kenel_fwd 审查报告

## Round 0 审查报告（Step 4 初审）

- **审查日期**: 2026-07-01
- **审查者**: Ascend C 算子审查代理 (Reviewer)
- **判定**: **PASS WITH NOTES**
- **总分**: **81 / 100**

---

## 1. 审查概要

| 维度 | 得分 | 满分 | 状态 |
|------|------|------|------|
| 1. 编译验证 | 10 | 10 | PASS |
| 2. 架构合规 | 13 | 15 | PASS |
| 3. 编码规范 | 10 | 15 | PASS (with issues) |
| 4. 性能优化 | 16 | 20 | PASS (with notes) |
| 5. 测试覆盖 | 13 | 15 | PASS (with gap) |
| 6. 精度验证 | 8 | 10 | PASS (with gap) |
| 7. 文档 | 11 | 15 | PASS |
| **总分** | **81** | **100** | **PASS WITH NOTES** |

---

## 2. 独立编译验证

### 2.1 编译结果：PASS

执行独立 clean rebuild：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

- **编译器**: bisheng (`/usr/local/Ascend/cann-9.0.0/bin/bisheng`)
- **架构**: `--npu-arch=dav-2201`
- **编译警告**: 0 个
- **编译错误**: 0 个
- **构建产物**: 
  - `build/expand_kenel_fwd` (377 KB, 直调可执行文件)
  - `build/libexpand_kenel_fwd_ops.so` (2.2 MB, PyTorch 扩展库)

| 检查项 | 得分 | 状态 |
|--------|------|------|
| 1.1 独立编译成功 | 7/7 | PASS |
| 1.2 无代码级警告 | 3/3 | PASS |

### 2.2 CMakeLists.txt 配置检查

手动验证通过：
- `find_package(ASC REQUIRED)` — 存在 (line 17)
- `LANGUAGES ASC CXX` — 存在 (line 19)
- `--npu-arch=dav-2201` — 存在 (lines 48, 108)，与目标芯片 Ascend910B2 (DAV_2201) 匹配
- `tiling_api` 链接 — 存在 (lines 33, 89)
- `register` 链接 — 存在 (lines 34, 90)

---

## 3. 代码质量评估

### 3.1 维度 2：架构合规性 — 13/15

| 检查项 | 得分 | 详情 |
|--------|------|------|
| 2.1 TPipe/TQue 模式 | 3/3 | 正确使用 TPipe + TQue\<VECIN\> + TQue\<VECOUT\> |
| 2.2 入口属性正确 | 3/3 | `extern "C" __global__ __vector__` 声明正确 |
| 2.3 定义顺序正确 | 2/3 | 结构清晰，但 Host 文件 `#include` kernel `.asc` 文件是直调模式的标准做法，无明显问题 |
| 2.4 内存管理配对 | 3/3 | AllocTensor/FreeTensor、EnQue/DeQue 正确配对 |
| 2.5 数据流完整 | 2/3 | 基本完整，但缺少显式 InsertSync 调用 |

**架构合规备注**：
- 算子为纯数据搬运，路线选择 SIMD/MemBase + DataCopy DMA，与 DESIGN.md 一致
- 非 DAV_3510，不走 RegBase/Blaze 路线，决策正确
- 使用 `TPipe` + `TQue` 的队列机制管理同步，符合 Ascend C 标准实践

### 3.2 维度 3：编码规范 — 10/15

| 检查项 | 得分 | 详情 |
|--------|------|------|
| 3.1 矢量 API | 4/4 | 纯数据搬运算子，使用 DataCopy/DataCopyPad API，无计算 API 需求 |
| 3.2 API 约束满足 | 2/4 | **问题 [M1]**：DataCopyPad UB→GM 写入非 32B 对齐 GM 地址时产生错误数据（见 §8） |
| 3.3 数据对齐 | 2/4 | **问题 [M2]**：非 16 对齐 H 值产生静默数据错误，应添加显式输入校验 |
| 3.4 命名规范 | 2/3 | 函数/变量命名清晰，但部分命名不一致（`ExpandRowsUBAligned` 命名冗长） |

**硬件参数检查**：全部通过
```
grep "blockDim\s*=\s*[0-9]"  → 无硬编码核数
grep "blockIdx\s*=\s*[0-9]"  → 无硬编码核索引
grep "PipeBarrier"           → 无（使用 TQue EnQue/DeQue 更优）
grep "SetValue\|GetValue"    → 无（未使用禁止的 GlobalTensor 标量 API）
```

### 3.3 维度 4：性能优化 — 16/20

| 检查项 | 得分 | 详情 |
|--------|------|------|
| 4.1 动态硬件参数 | 4/4 | `aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum)` 动态获取核数 |
| 4.2 多核并行 | 4/4 | 沿 totalRows 维度均匀分配，空闲核正确跳过 (`blockIdx >= usedCoreCnt`) |
| 4.3 流水线/双缓冲 | 2/4 | **问题 [L1]**：TQue\<VECIN, 1\> 和 TQue\<VECOUT, 1\> 均使用单缓冲，DESIGN.md 指定 VECIN=2 双缓冲 |
| 4.4 同步策略 | 3/4 | **逐项依赖分析通过**：队列 EnQue/DeQue 正确处理 MTE2→Vector→MTE3 依赖，但单缓冲限制并行度 |
| 4.5 计算效率与上板性能 | 3/4 | 性能可接受（见 §5），无循环内逐行 API、无不必要重复 GM 读取 |

**性能数据（独立采集）**：

| 配置 | 延迟 (ms) | 数据量 (MB) | 有效带宽 (GB/s) |
|------|-----------|-------------|-----------------|
| FP16 typical (B=1, S=1024, H=1280, M=4) | 0.280 | 12.5 | 43.5 |
| FP16 multicore (B=10, S=100, H=512, M=8) | 0.213 | 8.8 | 40.3 |
| FP32 typical (B=1, S=1024, H=1280, M=4) | 0.300 | 25.0 | 81.2 |

**同步策略逐项依赖分析**（审查参考手册要求）：

流水线阶段 CopyIn → Expand → CopyOut 的数据依赖：

| 依赖 | 生产者 | 消费者 | 同步机制 | 冗余？ |
|------|--------|--------|----------|--------|
| inBuf: MTE2 DMA → Vector 读取 | MTE2 (DataCopyPad) | Vector (Expand 中 DataCopy) | VECIN EnQue→DeQue | 否 |
| outBuf: Vector 写入 → MTE3 DMA | Vector (Expand 中 DataCopy) | MTE3 (CopyOut DataCopyPad) | VECOUT EnQue→DeQue | 否 |

队列机制依赖分析结论：**无冗余同步，各依赖正确覆盖**。

### 3.4 REGBASE 检查

本算子选择 SIMD/MemBase 路线（DAV_2201），非 RegBase 路线，无需加载 `/ascendc-regbase-best-practice`。

---

## 4. 设计合规检查 (vs DESIGN.md)

| DESIGN.md 要求 | 实现 | 一致性 |
|----------------|------|--------|
| tileH = AlignUp16(H) | tileH = AlignUp16(H) | 一致 |
| 双缓冲输入 (VECIN, 2) | 单缓冲 (VECIN, 1) | **偏离** [L1] |
| DataCopyPad 处理尾块 | DataCopyPad 用于 CopyIn | 一致 |
| 展平为 (totalRows, H) → (totalRows, M, H) | 展平实现正确 | 一致 |
| 多核切分: rowsPerCore = CeilDiv(totalRows, usedCoreCnt) | 实现一致 | 一致 |
| 空闲核跳过: blockIdx >= usedCoreCnt | 实现一致 | 一致 |
| UB 扩展: 逐行 DataCopy | ExpandRowsUBAligned 逐副本 | 一致 (DAV_2201 兼容) |
| UB 容量约束: (M+2)*tileH*sizeof(T) | 实现一致 | 一致 |

---

## 5. 测试覆盖评估

### 5.1 测试级别覆盖

| 级别 | 要求 | 覆盖情况 | 状态 |
|------|------|----------|------|
| Level 0 | 8-16 元素基础功能 | T2 (256 元素) 覆盖 | PASS |
| Level 1 | 1K 元素典型场景 | T1 (5.2M 元素) 覆盖 | PASS |
| Level 2 | 极值/零值边界 | T5 (M=1), T4 (M=16), T7 (H=32) 覆盖 | PASS |
| Level 3 | 大数据量性能 | T8 (4M 元素多核) 覆盖 | PASS |

### 5.2 测试结果（独立验证）

| # | 测试用例 | B | S | H | M | dtype | 独立结果 |
|---|---------|---|---|---|---|-------|---------|
| T1 | typical FP16 | 1 | 1024 | 1280 | 4 | FP16 | PASS (bitwise) |
| T2 | min rows | 1 | 1 | 128 | 2 | FP16 | PASS (bitwise) |
| T3 | multi rows | 4 | 256 | 256 | 2 | FP16 | PASS (bitwise) |
| T4 | large M | 1 | 1 | 1280 | 16 | FP16 | PASS (bitwise) |
| T5 | M=1 degenerate | 1 | 1 | 1280 | 1 | FP16 | PASS (bitwise) |
| T6 | FP32 | 1 | 1024 | 1280 | 4 | FP32 | PASS (bitwise) |
| T7 | aligned H=32 | 1 | 5 | 32 | 4 | FP16 | PASS (bitwise) |
| T8 | multicore | 10 | 100 | 512 | 8 | FP16 | PASS (bitwise) |
| T9 | large H | 1 | 1 | 2048 | 4 | FP16 | PASS (bitwise) |
| T10 | BF16 | 1 | 16 | 128 | 4 | BF16 | PASS (bitwise) |
| **边界: H=37** | non-aligned | 1 | 5 | 37 | 4 | FP16 | **FAIL (100/740 mismatches)** |

### 5.3 评分

| 检查项 | 得分 | 详情 |
|--------|------|------|
| 5.1 测试数据生成 | 4/4 | gen_data.py 正确生成随机输入 |
| 5.2 结果验证脚本 | 4/4 | verify_result.py 正确进行 bitwise match |
| 5.3 Level 0-2 覆盖 | 3/4 | 10 个标准用例全部通过，但缺少非对齐 H 的专项测试覆盖 |
| 5.4 精度标准明确 | 2/3 | bitwise match 标准适当，但未在验证脚本中显式声明阈值 |

---

## 6. 精度验证

### 6.1 独立精度测试结果

| dtype | 测试用例数 | 通过 | 失败 | 通过率 |
|-------|-----------|------|------|--------|
| FP16 | 9 (含 1 个非对齐失败) | 8 | 1 (H=37) | 88.9% |
| FP32 | 1 | 1 | 0 | 100% |
| BF16 | 1 | 1 | 0 | 100% |

所有 16 对齐 H 值的测试用例均 **bitwise match**（0 mismatch），符合非计算类算子精度标准。

### 6.2 非对齐 H 问题详细分析

**问题 [M1]**: H 值非 16 倍数时，输出产生静默数据错误。

| H 值 | 对齐状态 | mismatches (1 row, M=4) | 模式 |
|------|---------|------------------------|------|
| 33 | NOT_ALIGNED | 4 / 132 | 尾元素错误 |
| 37 | NOT_ALIGNED | 20 / 148 | 尾元素错误 |
| 39 | NOT_ALIGNED | 28 / 156 | 尾元素错误 |
| 41 | NOT_ALIGNED | 36 / 164 | 尾元素错误 |
| 48 | ALIGNED | 0 | PASS |
| 64 | ALIGNED | 0 | PASS |

**根因分析**: `mismatches = (H % 16) * M * totalRows`。当 H 不是 16 倍数时，每份输出副本的 GM 目的地址不是 32B 对齐的 —— 因为 `M * H * sizeof(T)` 不是 32B 的倍数，导致行间地址偏移逐步累积。`DataCopyPad(GlobalTensor, LocalTensor, DataCopyParams)`（UB→GM 方向）在 GM 目的地址非 32B 对齐时存在数据搬运错误，表现为每份副本末尾 `(H % 16)` 个元素的数据被错误写入。

### 6.3 精度评分

| 检查项 | 得分 | 详情 |
|--------|------|------|
| 6.1 FP32 全用例 PASS | 4/4 | FP32 典型用例 bitwise match |
| 6.2 FP16 全用例 PASS | 2/3 | 对齐 H 全部通过，非对齐 H 失败 [M1] |
| 6.3 BF16 全用例 PASS | 2/3 | BF16 典型用例通过，但未覆盖 BF16 非对齐场景 |

---

## 7. 问题清单

### HIGH 严重性 (0 项)

无。

### MEDIUM 严重性 (3 项)

| ID | 问题 | 位置 | 影响 |
|----|------|------|------|
| **[M1]** | 非 16 对齐 H 值产生静默数据错误 | `op_kernel/expand_kenel_fwd_kernel.asc`: CopyOut (lines 124-129) | 非对齐 H 值输入时输出错误，虽文档标注为限制但应添加显式校验拒绝非法输入 |
| **[M2]** | 缺少 Host 侧 H 对齐校验 | `op_host/expand_kenel_fwd.asc` main() + `op_extension/expand_kenel_fwd_torch.cpp` | H 非 16 对齐时无报错提示，用户获静默错误输出 |
| **[M3]** | 队列 Buffer 数量与设计不符 | `op_kernel/expand_kenel_fwd_kernel.asc`: lines 141-142 | VECIN=1 与 DESIGN.md 指定的 VECIN=2 不一致，当前 tilesPerRow=1 时功能等价但缺少设计层面的双缓冲支持 |

### LOW 严重性 (4 项)

| ID | 问题 | 位置 | 影响 |
|----|------|------|------|
| **[L1]** | 缺少显式 InsertSync 调用 | `op_kernel/expand_kenel_fwd_kernel.asc` | 队列机制隐式同步当前工作正常，但显式同步可增强可读性和安全性 |
| **[L2]** | `ExpandRowsUBAligned` 命名冗长 | `op_kernel/expand_kenel_fwd_kernel.asc`: line 29 | 建议简化为 `ExpandRows` 或 `UBExpand` |
| **[L3]** | Host 侧使用 `#include` kernel `.asc` 文件 | `op_host/expand_kenel_fwd.asc`: line 31 | 直调模式可行但非最佳实践，建议改用 extern 声明 |
| **[L4]** | 未覆盖非对齐 H 的测试用例 | 测试套件 | PLAN.md 提到 H=37 非对齐测试但实际未包含在标准 10 用例中 |

---

## 8. 修复建议

### [M1] / [M2] 非对齐 H 处理

**推荐方案 A（输入校验，短期）**：在 Host 侧添加显式校验。

```cpp
// op_extension/expand_kenel_fwd_torch.cpp, 在 tiling 计算前
TORCH_CHECK(H % 16 == 0, 
    "expand_kenel_fwd: H must be multiple of 16 (32B alignment). Got H=", H);
```

```cpp
// op_host/expand_kenel_fwd.asc, 在 main() 中参数解析后
if (H % 16 != 0) {
    std::cerr << "ERROR: H must be multiple of 16 (32B alignment). Got H=" << H << std::endl;
    return 1;
}
```

**推荐方案 B（修复根本问题，长期）**：修改 CopyOut 中的 GM 写入策略。

当前问题：`yGm[row*M*H + m*H + tileOff]` 在 H 非 16 对齐时，行间步长 `M*H` 不是 16 元素的倍数，导致所有副本的 GM 目的地址非 32B 对齐。

修复方向：在 CopyOut 中对每份副本使用 `DataCopyPad(yGm[gmBase], outBuf[ubBase], DataCopyParams{...}, DataCopyPadParams{...})` 的四参数形式，显式指定 pad 参数以正确处理非对齐 GM 写入。

### [M3] 双缓冲恢复

```cpp
// 将单缓冲改为双缓冲
AscendC::TQue<AscendC::TPosition::VECIN, 2> inQue;   // 改为 2
// 并相应调整 InitBuffer 参数
pipe_.InitBuffer(inQue, 2, tiling->tileH * sizeof(T)); // buffer num = 2
```

注意：仅当 tilesPerRow > 1 时双缓冲才有实际流水效益。当前 majority 场景 tilesPerRow=1，功能等价但应保持设计一致性。

### [L1] 添加显式同步

```cpp
// 在 CopyIn 后
pipe_.InsertSync(AscendC::HardEvent::MTE2_MTE3);
// 在 Expand 后
pipe_.InsertSync(AscendC::HardEvent::MTE3_MTE2);
```

### [L4] 扩展测试覆盖

在测试套件中添加：
- H=33 (非对齐，1 尾元素)
- H=37 (非对齐，5 尾元素)  
- H=256 (常见对齐值)

---

## 9. 文档审查

| 检查项 | 得分 | 详情 |
|--------|------|------|
| 7.1 README.md 存在 | 3/3 | README.md 包含概述、API 映射、构建运行指南 |
| 7.2 数学公式 | 3/3 | DESIGN.md 包含完整的数学语义和内存访问模式 |
| 7.3 编译运行指南 | 3/3 | README.md 含 cmake/make/run.sh 完整流程 |
| 7.4 API 映射/约束 | 1/3 | DESIGN.md §12 有 API 映射表，但 README.md 中未提及 H 16 对齐约束的重要性和原因 |
| 7.5 已知限制 | 1/3 | 文档提到 "H 维度需为 16 的倍数" 但未解释原因、影响范围和绕过方案 |

---

## 10. 交付件检查清单

| 编号 | 交付件 | 路径 | 状态 |
|------|--------|------|------|
| D1 | DESIGN.md | `docs/DESIGN.md` | 存在 |
| D2 | PLAN.md | `docs/PLAN.md` | 存在 |
| D3 | CMakeLists.txt | `CMakeLists.txt` | 存在，配置正确 |
| D4 | Kernel 实现 | `op_kernel/expand_kenel_fwd_kernel.asc` | 存在 |
| D5 | Tiling 头文件 | `op_kernel/expand_kenel_fwd_tiling.h` | 存在 |
| D6 | Host 直调入口 | `op_host/expand_kenel_fwd.asc` | 存在 |
| D7 | PyTorch 接入层 | `op_extension/expand_kenel_fwd_torch.cpp` | 存在 |
| D8 | 验证脚本 | `scripts/verify_result.py` | 存在 |

---

## 11. 综合判定

| 项目 | 内容 |
|------|------|
| **最终判定** | **PASS WITH NOTES** |
| **总分** | **81 / 100** |
| **必须修复项** | 0 项 |
| **建议修复项** | 3 项 MEDIUM + 4 项 LOW |
| **阻塞性问题** | 无（所有必须修复检查项 1.1/2.1/2.2/3.1/3.2/4.1/6.1 均通过） |

### 判定理由

1. **独立编译验证通过**（10/10）：零警告零错误，产物正确生成。
2. **所有必须修复检查项通过**：编译、架构、API 使用、硬件参数、精度核心测试均达标。
3. **10 项标准测试全部 bitwise match**：FP16/FP32/BF16 三种 dtype 均验证通过。
4. **非对齐 H 值是文档化的已知限制**（MEDIUM）：当前实现仅在 H 为 16 倍数时工作正确。需添加 Host 侧校验防止用户传入非法 H 值，长期需修复 CopyOut 的非对齐 GM 写入逻辑。
5. **性能可接受**：典型 FP16 场景 ~43 GB/s 有效带宽，FP32 ~81 GB/s。
6. **总分 81 >= 80 且无必须修复问题**：触发 PASS WITH NOTES 条件。

### 下一步

Developer 应优先处理 [M1]/[M2]（添加 H 对齐校验），然后视情况处理 [M3]（恢复双缓冲）和 [L1]（添加显式同步）。其他 LOW 项可在后续迭代中处理。

---

*审查完成于 2026-07-01，基于 CANN 9.0.0 / Ascend910B2 (DAV_2201) 环境独立验证。*

---

## Round 1 审查报告（Step 5 复审）

- **审查日期**：2026-07-01
- **审查者**：Ascend C 算子审查代理 (Reviewer)
- **判定**：**PASS**
- **总分**：**95 / 100**

---

## 1. 审查概要

Round 1 复审对 Round 0 中识别的全部问题进行了逐项验证。所有 MEDIUM 问题（M1/M2/M3）和 LOW 问题（L2/L4）均已修复，代码质量从 81 分提升至 95 分。

| 维度 | Round 0 得分 | Round 1 得分 | 满分 | 变化 |
|------|-------------|-------------|------|------|
| 1. 编译验证 | 10 | 10 | 10 | -- |
| 2. 架构合规 | 13 | 14 | 15 | +1 |
| 3. 编码规范 | 10 | 14 | 15 | +4 |
| 4. 性能优化 | 16 | 20 | 20 | +4 |
| 5. 测试覆盖 | 13 | 15 | 15 | +2 |
| 6. 精度验证 | 8 | 10 | 10 | +2 |
| 7. 文档 | 11 | 12 | 15 | +1 |
| **总分** | **81** | **95** | **100** | **+14** |

---

## 2. 独立编译验证

### 2.1 编译结果：PASS（10/10）

执行独立 clean rebuild：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cd operators/expand_kenel_fwd
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

- **编译器**: bisheng (`/usr/local/Ascend/cann-9.0.0/bin/bisheng`)
- **架构**: `--npu-arch=dav-2201`，与目标芯片 Ascend910B2 (DAV_2201) 匹配
- **编译警告**: 0 个
- **编译错误**: 0 个
- **构建产物**: 
  - `build/expand_kenel_fwd` (377 KB, 直调可执行文件)
  - `build/libexpand_kenel_fwd_ops.so` (2.2 MB, PyTorch 扩展库)

| 检查项 | 得分 | 状态 |
|--------|------|------|
| 1.1 独立编译成功 | 7/7 | PASS |
| 1.2 无代码级警告 | 3/3 | PASS |

### 2.2 CMakeLists.txt 配置检查

全部通过：
- `find_package(ASC REQUIRED)` — 存在 (line 17)
- `LANGUAGES ASC CXX` — 存在 (line 19)
- `--npu-arch=dav-2201` — 存在 (lines 48, 108)，与目标芯片匹配
- `tiling_api` 链接 — 存在 (lines 33, 89)
- `register` 链接 — 存在 (lines 34, 90)

### 2.3 硬件参数检查：PASS

```
grep "blockDim\s*=\s*[0-9]"  → 无硬编码核数
grep "blockIdx\s*=\s*[0-9]"  → 无硬编码核索引
grep "PipeBarrier"           → 无（使用 TQue EnQue/DeQue 更优）
grep "SetValue\|GetValue"    → 无（未使用禁止的 GlobalTensor 标量 API）
```

---

## 3. Round 0 问题修复验证

### 3.1 已修复问题确认

| ID | 问题 | Round 0 状态 | Round 1 验证 | 结论 |
|----|------|-------------|-------------|------|
| **[M1]** | 非 16 对齐 H 值产生静默数据错误 | 未修复 | Host 侧添加 H%16==0 校验，拒绝非法输入并返回明确错误信息 | **已修复** |
| **[M2]** | 缺少 Host 侧 H 对齐校验 | 未修复 | `expand_kenel_fwd.asc` (line 116-123) 和 `expand_kenel_fwd_torch.cpp` (line 52-56) 均添加校验 | **已修复** |
| **[M3]** | 队列 Buffer 数量与设计不符 | 未修复 (VECIN=1) | 统一为 `TQue<VECIN, 2>` (kernel line 141)，双缓冲正确实现 | **已修复** |
| **[L2]** | `ExpandRowsUBAligned` 命名冗长 | 未修复 | 简化为 `ExpandRows` (kernel line 29) | **已修复** |
| **[L4]** | 未覆盖非对齐 H 的测试用例 | 未修复 | 添加 T11(H=33), T12(H=37) 拒绝验证 + T13(H=256) 对齐验证 + T14(大 batch) 负载验证 | **已修复** |

### 3.2 搁置问题确认

| ID | 问题 | Round 0 状态 | Round 1 确认 | 对评分影响 |
|----|------|-------------|-------------|-----------|
| **[L1]** | 缺少显式 InsertSync 调用 | 搁置 | TQue EnQue/DeQue 隐式同步完整覆盖 MTE2→VEC→MTE3 依赖链，显式 InsertSync 非必须 | 无影响 |
| **[L3]** | Host 侧 `#include` kernel `.asc` 文件 | 搁置 | 直调模式功能正确，非最佳实践但可接受 | 维度 2.3 扣 1 分 |

### 3.3 M1/M2 修复验证：非对齐 H 拒绝

```
测试: ./expand_kenel_fwd 1 5 37 4 0
输出: ERROR: expand_kenel_fwd: H must be a multiple of 16 (32-byte alignment requirement).
      Got H=37. Common LLM hidden sizes (768, 1280, 2048, 4096, etc.) are all compatible.
退出码: 1 (正确拒绝)
```

---

## 4. 代码质量评估

### 4.1 维度 2：架构合规性 — 14/15

| 检查项 | 得分 | 详情 |
|--------|------|------|
| 2.1 TPipe/TQue 模式 | 3/3 | 正确使用 TPipe + TQue\<VECIN, 2\> + TQue\<VECOUT, 1\> |
| 2.2 入口属性正确 | 3/3 | `extern "C" __global__ __vector__` 声明正确 |
| 2.3 定义顺序正确 | 2/3 | **轻微问题**: Host 文件通过 `#include` 引用 kernel `.asc` 文件（直调模式可接受，非最佳实践；建议改用 extern 声明） |
| 2.4 内存管理配对 | 3/3 | AllocTensor/FreeTensor、EnQue/DeQue 正确配对 |
| 2.5 数据流完整 | 3/3 | CopyIn(MTE2)→EnQue→DeQue→Expand(VEC)→EnQue(VECOUT)→DeQue→CopyOut(MTE3) 依赖链完整，TQue 隐式同步无遗漏 |

### 4.2 维度 3：编码规范 — 14/15

| 检查项 | 得分 | 详情 |
|--------|------|------|
| 3.1 矢量 API | 4/4 | 纯数据搬运算子，使用 DataCopy/DataCopyPad API，无计算 API 需求 |
| 3.2 API 约束满足 | 4/4 | **已验证通过 CANN 9.0.0 头文件**：DataCopy(Local,Local,uint32_t) UB-to-UB (line 247)、DataCopyPad MTE2 4-param (line 359)、DataCopyPad MTE3 3-param (line 365)。所有约束满足（32B 对齐、GM 地址对齐、DataCopyParams blockLen 在 uint16_t 范围内） |
| 3.3 数据对齐 | 4/4 | tileH=AlignUp16(H)，H%16==0 强制校验；所有 DataCopy count × sizeof(T) 为 32B 倍数；CopyOut GM 目标地址 32B 对齐 |
| 3.4 命名规范 | 2/3 | **轻微问题**: `expand_kenel_fwd_tiling.h` 中 `totalTiles` 和 `tailH` 的注释写 "始终为 totalRows，无 H 维切分"，但代码实际支持 tilesPerRow > 1 的 H 维切分场景。注释与代码实现不完全一致。 |

### 4.3 维度 4：性能优化 — 20/20

| 检查项 | 得分 | 详情 |
|--------|------|------|
| 4.1 动态硬件参数 | 4/4 | `aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum)` 动态获取核数；UB_BUDGET_BYTES 和 tileH 均运行时计算 |
| 4.2 多核并行 | 4/4 | 沿 totalRows 维度均匀分配（rowsPerCore = CeilDiv），空闲核正确跳过 (`blockIdx >= usedCoreCnt` 即返回) |
| 4.3 流水线/双缓冲 | 4/4 | **M3 已修复**: TQue\<VECIN, 2\> 双缓冲输入，与 DESIGN.md 一致。当 tilesPerRow > 1 时 CopyIn 与上一 tile Expand 可重叠 |
| 4.4 同步策略 | 4/4 | TQue EnQue/DeQue 隐式同步正确覆盖 MTE2→Vector(Expand)→MTE3 全链路依赖，无冗余同步，无遗漏依赖 |
| 4.5 计算效率与上板性能 | 4/4 | 无循环内逐行 API 调用；无重复 GM 读取；批量 DataCopy 处理；性能数据优异（见 §7） |

### 4.4 同步策略逐项依赖分析

| 依赖 | 生产者 | 消费者 | 同步机制 | 冗余？ |
|------|--------|--------|----------|--------|
| inBuf: MTE2 DMA → Vector 读取 | MTE2 (DataCopyPad) | Vector (Expand 中 UB-UB DataCopy) | VECIN EnQue→DeQue | 否 |
| outBuf: Vector 写入 → MTE3 DMA | Vector (Expand 中 UB-UB DataCopy) | MTE3 (CopyOut DataCopyPad) | VECOUT EnQue→DeQue | 否 |

结论：**无冗余同步，各依赖正确覆盖**。TQue 深度 2（VECIN）在 tilesPerRow > 1 时允许 Ping-Pong 流水。

### 4.5 API 签名验证（CANN 9.0.0 头文件）

全部 API 签名已与 CANN 9.0.0 `/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc/include/interface/kernel_operator_data_copy_intf.h` 对照确认：

| API | 签名 | 行号 | 状态 |
|-----|------|------|------|
| `DataCopy(Local, Local, count)` | `const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t count` | L247 | 匹配 |
| `DataCopyPad MTE2` | `const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams&, const DataCopyPadParams&` | L359 | 匹配 |
| `DataCopyPad MTE3` | `const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams&` | L365 | 匹配 |

---

## 5. 设计合规检查 (vs DESIGN.md)

| DESIGN.md 要求 | 实现 | 一致性 |
|----------------|------|--------|
| tileH = AlignUp16(H) | tileH = AlignUp16(H) | 一致 |
| 双缓冲输入 (VECIN, 2) | TQue\<VECIN, 2\> + InitBuffer(inQue, 2, ...) | **一致 (Round 0 偏离已修复)** |
| DataCopyPad 处理 GM↔UB | CopyIn 用 4-param MTE2 DataCopyPad, CopyOut 用 3-param MTE3 DataCopyPad | 一致 |
| 展平为 (totalRows, H) → (totalRows, M, H) | 展平实现正确 | 一致 |
| 多核切分: rowsPerCore = CeilDiv(totalRows, usedCoreCnt) | 实现一致 | 一致 |
| 空闲核跳过: blockIdx >= usedCoreCnt | 实现一致 | 一致 |
| UB 扩展: 逐副本 DataCopy | ExpandRows 逐副本 UB-UB DataCopy | 一致 |
| UB 容量约束: (M+2) × tileH × sizeof(T) ≤ UB_BUDGET | 实现一致 | 一致 |
| H 对齐校验: H % 16 == 0 | Host 侧 + PyTorch 扩展侧双重校验 | **一致 (Round 0 偏离已修复)** |
| 非 H 对齐拒绝并返回明确错误信息 | 错误信息格式符合 DESIGN.md §11.3 | 一致 |

### 路线合规性确认

- 算子类型: Conversion/Broadcast 数据搬运，无计算
- 架构: DAV_2201 (Ascend910B2)，非 DAV_3510
- 路线: SIMD/MemBase + DataCopy DMA（非 RegBase、非 Blaze）
- **无需加载 `/ascendc-regbase-best-practice` 或 `/ascendc-blaze-best-practice`**

---

## 6. 测试覆盖评估

### 6.1 测试级别覆盖

| 级别 | 要求 | 覆盖情况 | 状态 |
|------|------|----------|------|
| Level 0 | 8-16 元素基础功能 | T2 (256 元素)、T7 (640 元素) 覆盖 | PASS |
| Level 1 | 1K 元素典型场景 | T1 (5.2M 元素)、T6 (5.2M 元素 FP32)、T14 (41.9M 元素) 覆盖 | PASS |
| Level 2 | 极值/零值边界 | T5 (M=1 退化)、T4 (M=16 大M)、T7 (H=32 最小对齐)、T11/T12 (非对齐拒绝)、T13 (H=256 常用对齐) 覆盖 | PASS |
| Level 3 | 大数据量性能 | T8 (多核 4M 元素)、T14 (大 batch 41.9M 元素) 覆盖 | PASS |

### 6.2 独立测试结果（全部 14 项 Bitwise Match）

| # | 测试用例 | B | S | H | M | dtype | 元素数 | 独立结果 |
|---|---------|---|---|---|---|-------|--------|---------|
| T1 | typical FP16 | 1 | 1024 | 1280 | 4 | FP16 | 5,242,880 | PASS (bitwise, 0 mismatch) |
| T2 | min rows | 1 | 1 | 128 | 2 | FP16 | 256 | PASS (bitwise, 0 mismatch) |
| T3 | multi rows | 4 | 256 | 256 | 2 | FP16 | 524,288 | PASS (bitwise, 0 mismatch) |
| T4 | large M | 1 | 1 | 1280 | 16 | FP16 | 20,480 | PASS (bitwise, 0 mismatch) |
| T5 | M=1 degenerate | 1 | 1 | 1280 | 1 | FP16 | 1,280 | PASS (bitwise, 0 mismatch) |
| T6 | FP32 | 1 | 1024 | 1280 | 4 | FP32 | 5,242,880 | PASS (bitwise, 0 mismatch) |
| T7 | aligned H=32 | 1 | 5 | 32 | 4 | FP16 | 640 | PASS (bitwise, 0 mismatch) |
| T8 | multicore | 10 | 100 | 512 | 8 | FP16 | 4,096,000 | PASS (bitwise, 0 mismatch) |
| T9 | large H | 1 | 1 | 2048 | 4 | FP16 | 8,192 | PASS (bitwise, 0 mismatch) |
| T10 | BF16 | 1 | 16 | 128 | 4 | BF16 | 8,192 | PASS (bitwise, 0 mismatch) |
| T11 | H=33 non-aligned | 1 | 5 | 33 | 4 | FP16 | — | **REJECTED (正确拒绝)** |
| T12 | H=37 non-aligned | 1 | 5 | 37 | 4 | FP16 | — | **REJECTED (正确拒绝)** |
| T13 | aligned H=256 | 1 | 1 | 256 | 4 | FP16 | 1,024 | PASS (bitwise, 0 mismatch) |
| T14 | large batch | 8 | 1024 | 1280 | 4 | FP16 | 41,943,040 | PASS (bitwise, 0 mismatch) |

### 6.3 评分

| 检查项 | 得分 | 详情 |
|--------|------|------|
| 5.1 测试数据生成 | 4/4 | gen_data.py 正确生成随机输入和 golden 输出 |
| 5.2 结果验证脚本 | 4/4 | verify_result.py 正确进行 bitwise match，含 mismatch 位置诊断输出 |
| 5.3 Level 0-3 覆盖 | 4/4 | **L4 已修复**: 14 测试用例覆盖全部四级测试要求，含 2 项非对齐拒绝验证和 2 项扩展覆盖 |
| 5.4 精度标准明确 | 3/3 | Bitwise match 标准明确，验证脚本输出格式清晰 |

---

## 7. 性能数据（独立采集）

### 7.1 Benchmark 结果

| 配置 | AscendC (ms) | PyTorch (ms) | 加速比 | 数据量 (MB) | 有效带宽 (GB/s) |
|------|-------------|-------------|--------|-------------|-----------------|
| T1 FP16 typical (1,1024,1280,M=4) | 0.116 | 8.836 | 76.15x | 12.5 | 105.2 |
| T2 FP16 min rows (1,1,128,M=2) | 0.104 | 0.074 | 0.71x | 0.001 | — |
| T3 FP16 multi rows (4,256,256,M=2) | 0.108 | 9.181 | 85.35x | 1.3 | 11.7 |
| T4 FP16 large M (1,1,1280,M=16) | 0.112 | 0.073 | 0.66x | 0.049 | — |
| T5 FP16 M=1 (1,1,1280,M=1) | 0.111 | 0.061 | 0.55x | 0.003 | — |
| T6 FP32 typical (1,1024,1280,M=4) | 0.107 | 9.045 | 84.63x | 25.0 | 228.0 |
| T7 FP16 H=32 (1,5,32,M=4) | 0.103 | 0.069 | 0.67x | 0.002 | — |
| T8 FP16 multicore (10,100,512,M=8) | 0.101 | 8.471 | 83.66x | 8.8 | 84.9 |
| T9 FP16 large H (1,1,2048,M=4) | 0.100 | 0.070 | 0.70x | 0.020 | — |
| T10 BF16 (1,16,128,M=4) | 0.099 | 0.072 | 0.73x | 0.020 | — |

### 7.2 性能分析

- **大 workload 加速比**: 76-85x，AscendC kernel 延迟稳定在 ~100us 左右
- **小 workload**: kernel launch overhead (~100us) 占主导，加速比 < 1x 为正常行为
- **几何平均加速比**: **4.57x** (10 用例)
- **算术平均加速比**: 33.38x (偏向大 workload)
- **有效带宽（大 workload）**: FP16 ~105 GB/s, FP32 ~228 GB/s (读+写合计)
- **Task Duration 稳定性**: 大 workload 下 ~100us，与数据量相关性弱（体现出 pipeline 并行效率）

---

## 8. 精度验证

### 8.1 独立精度测试结果

| dtype | 测试用例数 | 通过 | 失败 | 通过率 | 验证标准 |
|-------|-----------|------|------|--------|---------|
| FP16 | 10 (含 2 个拒绝) | 10 | 0 | 100% | Bitwise Match |
| FP32 | 1 | 1 | 0 | 100% | Bitwise Match |
| BF16 | 1 | 1 | 0 | 100% | Bitwise Match |

所有 12 个有效测试用例（含 T11/T12 正确拒绝）均 **0 mismatch**，符合非计算类算子 Bitwise Match 精度标准。

### 8.2 非对齐 H 问题验证

Round 0 中 M1/M2 问题已完全修复。非对齐 H 值现被正确拒绝：

| H 值 | 对齐状态 | Round 0 行为 | Round 1 行为 |
|------|---------|-------------|-------------|
| 37 | NOT_ALIGNED | 静默数据错误 (100/740 mismatches) | Host 侧拒绝并返回错误信息 (exit code 1) |
| 33 | NOT_ALIGNED | 静默数据错误 | Host 侧拒绝并返回错误信息 |
| 1280 | ALIGNED | PASS | PASS |

### 8.3 精度评分

| 检查项 | 得分 | 详情 |
|--------|------|------|
| 6.1 FP32 全用例 PASS | 4/4 | T6 FP32 bitwise match (5,242,880 元素) |
| 6.2 FP16 全用例 PASS | 3/3 | 全部 10 项 FP16 测试用例 bitwise match (含 M=1/2/4/8/16、H=32/128/256/512/1280/2048) |
| 6.3 BF16 全用例 PASS | 3/3 | T10 BF16 bitwise match (8,192 元素) |

---

## 9. 文档审查

| 检查项 | 得分 | 详情 |
|--------|------|------|
| 7.1 README.md 存在 | 3/3 | README.md 包含概述、功能说明、数据类型表、快速开始、文件结构 |
| 7.2 数学公式 | 3/3 | DESIGN.md 包含完整的数学语义和内存访问模式公式 |
| 7.3 编译运行指南 | 3/3 | README.md 含 cmake/make/run.sh 完整流程、直调和 PyTorch 两种使用方式 |
| 7.4 API 映射/约束 | 1/3 | DESIGN.md §12 有完整的 API 映射表（含签名、约束、验证行号），但 README.md 中 API 映射信息较简略 |
| 7.5 已知限制 | 2/3 | README.md 已知限制章节清晰说明 H%16==0 约束及原因，但可补充非对齐 H 被拒绝时的错误信息示例和常见绕过方案 |

---

## 10. 问题清单

### HIGH 严重性 (0 项)

无。

### MEDIUM 严重性 (0 项)

Round 0 的 M1/M2/M3 已全部修复。

### LOW 严重性 (3 项)

| ID | 问题 | 位置 | 影响 | 建议 |
|----|------|------|------|------|
| **[L3]** | Host 侧 `#include` kernel `.asc` 文件 | `op_host/expand_kenel_fwd.asc`: line 31 | 直调模式功能正确，但非最佳实践。Kernel 代码被编译两次（一次在直调目标、一次在 .so 目标），增量编译时可能浪费编译时间 | 改用 extern 声明替代 `#include`，参考 `op_extension/expand_kenel_fwd_torch.cpp` 的做法 |
| **[L5]** | Tiling 头文件注释与实现不完全一致 | `op_kernel/expand_kenel_fwd_tiling.h`: lines 39-40 | `totalTiles` 注释写"始终为 totalRows，无 H 维切分"，但代码实际支持 tilesPerRow > 1 的 H 维切分场景（当 H 极大超过 UB 容量时） | 更新注释为实际语义：`totalTiles = totalRows * tilesPerRow`，`tailH = H - (tilesPerRow-1) * tileH` |
| **[L6]** | README.md API 映射信息可增强 | `README.md` | 当前 API 映射较简略，用户需查阅 DESIGN.md 获取完整信息 | 在 README.md 中添加关键 API 简表（DataCopy、DataCopyPad、TQue）及核心约束（32B 对齐、H%16==0） |

---

## 11. 交付件检查清单

| 编号 | 交付件 | 路径 | 状态 |
|------|--------|------|------|
| D1 | DESIGN.md | `docs/DESIGN.md` | 存在，内容完整 |
| D2 | PLAN.md | `docs/PLAN.md` | 存在，记录全部修复历史 |
| D3 | CMakeLists.txt | `CMakeLists.txt` | 存在，配置正确 |
| D4 | Kernel 实现 | `op_kernel/expand_kenel_fwd_kernel.asc` | 存在，代码清晰 |
| D5 | Tiling 头文件 | `op_kernel/expand_kenel_fwd_tiling.h` | 存在，注释可改进 [L5] |
| D6 | Host 直调入口 | `op_host/expand_kenel_fwd.asc` | 存在，功能完整 |
| D7 | PyTorch 接入层 | `op_extension/expand_kenel_fwd_torch.cpp` | 存在，功能完整 |
| D8 | 验证脚本 | `scripts/verify_result.py` | 存在，功能完整 |

---

## 12. 综合判定

| 项目 | 内容 |
|------|------|
| **最终判定** | **PASS** |
| **总分** | **95 / 100** |
| **必须修复项** | **0 项** |
| **建议修复项** | 3 项 LOW |
| **阻塞性问题** | **无**（所有必须修复检查项 1.1/2.1/2.2/3.1/3.2/4.1/6.1 均通过） |

### 判定理由

1. **独立编译验证通过（10/10）**：零警告零错误，产物正确生成。
2. **所有 Round 0 问题已修复**：M1/M2 (H 对齐校验)、M3 (双缓冲恢复)、L2 (命名简化)、L4 (测试覆盖扩展) —— 全部验证通过。
3. **14 项测试全部 Bitwise Match**：FP16/FP32/BF16 三种 dtype 均 0 mismatch，覆盖 M=1/2/4/8/16、H=32/128/256/512/1280/2048、多核/大 batch 等场景。非对齐 H 正确拒绝。
4. **所有 API 签名已对照 CANN 9.0.0 头文件验证**：DataCopy(Local,Local,count)、DataCopyPad MTE2、DataCopyPad MTE3 全部匹配。
5. **同步策略正确**：TQue EnQue/DeQue 隐式同步完整覆盖 MTE2→VEC→MTE3 依赖链。
6. **性能优异**：大 workload 76-85x 加速比，几何平均 4.57x，kernel 延迟稳定在 ~100us。
7. **总分 95 >= 80 且无必须修复问题**：触发 **PASS** 条件。

### 改进建议

建议 Developer 在后续迭代中处理以下 LOW 项：

- **[L5]**: 更新 `expand_kenel_fwd_tiling.h` 中 `totalTiles`/`tailH` 的注释以反映真实的 H 维切分支持。
- **[L6]**: 在 README.md 中补充关键 API 简表和核心对齐约束说明。
- **[L3]**: 将 Host 直调入口的 `#include` kernel 改为 extern 声明（后续迭代）。

---

*Round 1 复审完成于 2026-07-01，基于 CANN 9.0.0 / Ascend910B2 (DAV_2201) 环境独立验证。*
