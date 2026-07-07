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
