# engram_hash AscendC 算子审查报告

> **审查阶段**: Round 0（初审）
> **审查日期**: 2026-07-01
> **审查人**: Reviewer (独立审查)
> **算子**: engram_hash
> **目标芯片**: Ascend 910B2 (DAV_2201)
> **CANN**: 9.0.0
> **判定**: **PASS**
> **总分**: **97 / 100**

---

## 审查概要

| 项目 | 结果 |
|------|:--:|
| 独立编译 | 通过（无警告） |
| CMake 配置验证 | 通过 |
| 精度验证（完整矩阵） | 72/72 bit-exact |
| PyTorch 集成测试 | 7/7 bit-exact |
| Geomean speedup vs PyTorch | 5.90x |
| 多核扩展效率 | 99.5% (47.78x / 48x ideal) |
| 同步策略 | 零冗余 barrier（TQue EnQue/DeQue 同步） |
| 硬件参数硬编码 | 无 |
| GlobalTensor::SetValue/GetValue (黑名单) | 无使用 |

---

## 1. 独立构建验证

### 1.1 CMake 配置验证

```bash
python3 workflows/scripts/verify_cmake_config.py CMakeLists.txt
```

**结果**: 通过。已确认：
- `find_package(ASC REQUIRED)` 存在
- `project(... LANGUAGES ASC CXX)` 正确
- `--npu-arch=dav-2201` 与目标芯片匹配
- 链接 `tiling_api`、`register`、`platform` 等必需库

### 1.2 独立编译

从源码全新编译（清空 build 目录）：

| Target | 状态 | 编译器 |
|--------|:--:|------|
| `engram_hash_custom` (直调可执行) | 通过 | bisheng, `--npu-arch=dav-2201` |
| `libengram_hash_ops.so` (PyTorch 扩展) | 通过 | bisheng + GCC 11.4.0 |

编译过程无任何警告或错误。

---

## 2. 代码质量评估 -- 7 维度评分

### 维度 1: 编译验证 (10/10)

| # | 检查项 | 结果 | 得分 |
|---|--------|:--:|:--:|
| 1.1 | 独立编译成功 | 通过 | 7/7 |
| 1.2 | 无代码级警告 | 通过 | 3/3 |

### 维度 2: 架构合规 (15/15)

| # | 检查项 | 结果 | 得分 |
|---|--------|:--:|:--:|
| 2.1 | TPipe/TQue 模式 | 通过 -- 使用 TPipe + TQue，正确的 UB 管理范式 | 3/3 |
| 2.2 | 入口属性正确 | 通过 -- `extern "C" __global__ __aicore__` + `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` | 3/3 |
| 2.3 | 定义顺序正确 | 通过 -- 核函数定义在前，无前向声明 | 3/3 |
| 2.4 | 内存管理配对 | 通过 -- AllocTensor/FreeTensor 5 对全部配对，EnQue/DeQue 5 对全部配对 | 3/3 |
| 2.5 | 数据流完整 | 通过 -- 清晰的 CopyIn(常驻表) -> TileLoop(CopyIn -> Compute -> CopyOut) 模式 | 3/3 |

**同步策略逐项依赖分析**：

本 kernel 使用 TQue EnQue/DeQue 机制实现所有跨 pipe 同步，**无任何 PipeBarrier 调用**。逐项分析如下：

| 序号 | 前操作 | 前 Pipe | 后操作 | 后 Pipe | 同步机制 | 判定 |
|:--:|--------|---------|--------|---------|---------|:--:|
| 1 | DataCopyPad(mulUb<-GM) | MTE2 | GetValue(mulUb) | Scalar | mulQ.EnQue->DeQue | 正确 |
| 2 | DataCopyPad(vocUb<-GM) | MTE2 | GetValue(vocUb) | Scalar | vocQ.EnQue->DeQue | 正确 |
| 3 | DataCopyPad(offUb<-GM) | MTE2 | GetValue(offUb) | Scalar | offQ.EnQue->DeQue | 正确 |
| 4 | DataCopyPad(ngUb<-GM) | MTE2 | GetValue(ngUb) | Scalar | ngQ.EnQue->DeQue | 正确 |
| 5 | SetValue(outUb) | Scalar | DataCopyPad(GM<-outUb) | MTE3 | outQ.EnQue->DeQue | 正确 |

**同步冗余率 = 0%（0/0 冗余 barrier）**。所有跨 pipe 依赖均由 EnQue/DeQue 正确同步，无需额外的 PipeBarrier。精细且正确的同步设计。

### 维度 3: 编码规范 (15/15)

| # | 检查项 | 结果 | 得分 |
|---|--------|:--:|:--:|
| 3.1 | 矢量 API 使用 | 通过 -- 标量路径是 DAV_2201 上 int64 运算的唯一正确选择（Vector 单元不支持 int64 mul/xor/mod）。LocalTensor GetValue/SetValue 在 UB 上允许使用 | 4/4 |
| 3.2 | API 约束满足 | 通过 -- 全过程使用 DataCopyPad（GM<->UB），无 GlobalTensor GetValue/SetValue（黑名单 API） | 4/4 |
| 3.3 | 数据对齐 | 通过 -- `outLayerStride` padding 确保每 layer 输出段 32B 对齐，`tileTokens` 对齐到 8 | 4/4 |
| 3.4 | 命名规范 | 通过 -- 变量名清晰自文档化（`mulUb`, `ngUb`, `outLayerStride` 等），符合工程风格 | 3/3 |

**黑名单 API 检查**：确认 kernel 中未使用 `GlobalTensor::SetValue` / `GlobalTensor::GetValue`。所有元素级读写通过 `LocalTensor::GetValue` / `LocalTensor::SetValue` 在 UB 上完成（允许）。

### 维度 4: 性能优化 (19/20)

| # | 检查项 | 结果 | 得分 |
|---|--------|:--:|:--:|
| 4.1 | 动态硬件参数 | 通过 -- 无硬编码核数（`GetBlockIdx()` + `td->blockNum`），coreNum 由外部传入。grep `blockDim\s*=\s*\d+` / `blockIdx\s*=\s*\d+` 均无匹配 | 4/4 |
| 4.2 | 多核并行 | 通过 -- Token 维切分，尾核正确处理，空闲核 return 跳过，负载差异 <=1 tile。多核扩展效率 99.5%（独立验证确认: 47.78x vs 48x ideal） | 4/4 |
| 4.3 | 流水线/双缓冲 | 通过 (3/4) -- 使用 TQue 单深度队列（非双缓冲），EnQue/DeQue 提供流水线同步。**减分理由**: 双缓冲未实现。由于本算子 MTE 利用率 <2%（scalar-bound），双缓冲实际收益有限，此项设计决策合理但不符满分标准 | 3/4 |
| 4.4 | 同步策略 | 通过 -- 零冗余 barrier，EnQue/DeQue 精确同步，精细 pipe 标识 | 4/4 |
| 4.5 | 计算效率与上板性能 | 通过 -- 无循环内逐行 API 调用（标量是唯一选择）；常驻表一次载入 UB 复用；无重复 GM 读取。独立 benchmark: geomean 5.90x vs torch，大 batch (NT=65536) 1.31x；scalar ratio=99.8% 接近饱和 | 4/4 |

### 维度 5: 测试覆盖 (15/15)

| # | 检查项 | 结果 | 得分 |
|---|--------|:--:|:--:|
| 5.1 | 测试数据生成 | 通过 -- `gen_data.py` 正确生成所有输入类型 + golden，值域匹配设计规格 | 4/4 |
| 5.2 | 结果验证脚本 | 通过 -- `verify_result.py` 使用 `np.array_equal`（bit-exact），`run_verify_matrix.py` 覆盖完整矩阵 | 4/4 |
| 5.3 | Level 0 覆盖 | 通过 -- Level 0 (8 元素) 到 Level 3 (65536 元素) 全覆盖：72 个 shape x core 组合 | 4/4 |
| 5.4 | 精度标准明确 | 通过 -- DESIGN.md 和 README.md 明确指出 bit-exact 标准 | 3/3 |

### 维度 6: 精度验证 (10/10)

| # | 检查项 | 结果 | 得分 |
|---|--------|:--:|:--:|
| 6.1 | 整数 bit-exact (全用例) | 通过 -- 72/72 bit-exact（完整验证矩阵） | 4/4 |
| 6.2 | PyTorch 集成 bit-exact | 通过 -- 7/7 bit-exact（torch.ops.npu.engram_hash） | 3/3 |
| 6.3 | 边界用例 | 通过 -- W=1/7/21 对齐边界、NT<核数欠核、尾核不整除、大 NT 多 tile 全通过 | 3/3 |

### 维度 7: 文档 (13/15)

| # | 检查项 | 结果 | 得分 |
|---|--------|:--:|:--:|
| 7.1 | README.md 存在 | 通过 -- 完整 README，含算子规格、构建运行指南、API 映射 | 3/3 |
| 7.2 | 数学公式 | 通过 -- DESIGN.md 1.1 含完整算法描述、数学公式、数值范围核算 | 3/3 |
| 7.3 | 编译运行指南 | 通过 -- README.md 构建/直接调用测试 + PLAN.md 6 构建与运行速查 | 3/3 |
| 7.4 | API 映射/约束 | 通过 (2/3) -- DESIGN.md 6.3 含完整 API 映射表。**减分理由**: README.md 未单独列出 API 映射/约束表（DESIGN.md 有，但 README 仅列出架构概述未涵盖 API 层面） | 2/3 |
| 7.5 | 已知限制 | 通过 -- PLAN.md 5 风险与对策记录 9 项风险及处置，README 有基本限制说明 | 3/3 |

**总计**: **10 + 15 + 15 + 19 + 15 + 10 + 13 = 97 / 100**

---

## 3. 设计合规检查

对照 DESIGN.md v1.1 逐项检查实现一致性：

| 设计项 | 设计规格 | 实现 | 判定 |
|--------|---------|------|:--:|
| 核类型 | `KERNEL_TYPE_AIV_ONLY` | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` | 一致 |
| 计算方式 | 标量 int64 (mul/xor/mod) | `int64_t` 原生 `*`/`^`/`%` | 一致 |
| 多核切分 | Token 维 (NT) | `bid * tokensPerCore` + `myTokens` 尾核处理 | 一致 |
| UB 常驻表 | mulUb, vocUb, offUb tile 循环外一次性载入 | 第 74-105 行实现 | 一致 |
| 输出分段写 | 按 layer 分段 DataCopyPad | 第 156-161 行实现 | 一致 |
| outLayerStride | padding 确保 32B 对齐 | `(tileTokens*W + 7)/8*8` | 一致 |
| tileTokens 对齐 | 8 的倍数 | tiling 计算中 `tileTokens/8*8` | 一致 |
| N=1 退化 | W==0 提前 return | 第 41 行: `if (W==0||...) return;` | 一致 |
| 多核边界 | bid>=blockNum / myTokens==0 return | 第 44, 48 行实现 | 一致 |
| Torch 注册 | TORCH_LIBRARY + PrivateUse1 + Meta | register.cpp 完整实现 | 一致 |

**结论**: 设计框架无偏离，所有关键设计决策已在代码中正确实现。

---

## 4. 最终轮附加检查（交付件 + 代码清洁）

### 4.1 交付件检查清单

| # | 交付件 | 路径 | 状态 |
|---|--------|------|:--:|
| D1 | 算子源码 | `op_kernel/engram_hash_kernel.asc` | 独立编译通过，无警告 |
| D2 | 构建文件 | `CMakeLists.txt` | 依赖项完整，双目标 (可执行 + .so) |
| D3 | Golden 数据生成 | `scripts/gen_data.py` | 支持所有 shape，值域匹配规格 |
| D4 | 运行脚本 | `scripts/build_and_test.sh` | 一键构建+测试可用 |
| D5 | 算子文档 | `README.md` | 算子概述、数学公式(引 DESIGN)、编译运行指南、性能数据、已知限制 |
| D6 | 设计文档 | `docs/DESIGN.md` | 需求分析、技术路线决策、数据流、并行策略、UB 规划、精度策略、API 映射 |
| D7 | 开发计划 | `docs/PLAN.md` | 6 阶段全部标记完成，测试结果详细记录 |
| D8 | 审查报告 | `docs/REVIEW.md` | 本文档 |

全部交付件齐全。

### 4.2 代码清洁检查

| # | 检查项 | 搜索范围 | 结果 |
|---|--------|---------|:--:|
| C1 | printf/cout 残留 | `*.asc` + `*.cpp` | 仅 host CLI 工具必要输出（非 kernel），无调试残留 |
| C2 | TODO/FIXME 残留 | `*.asc` + `*.cpp` | 无 |
| C3 | 注释掉的代码块 | 目视 | 无 -- 仅有 Javadoc 块注释和行注释 |
| C4 | 调试用硬编码 | `*.asc` | 无 -- host main 默认值是 CLI 参数预设值，非调试写死 |

代码清洁，无调试残留。

---

## 5. 精度全覆盖验证

**精度验收状态**: 通过

算子为整数计算类，精度标准为 **bit-exact（绝对误差 = 0）**。

### 5.1 完整验证矩阵（72 用例，独立执行）

全部 72 个 (shape, core) 组合 bit-exact 通过：

| 维度 | 取值 | 用例数 | 结果 |
|------|------|:----:|:--:|
| NT (num_tokens) | 32, 100, 256, 1000, 1024, 4096, 65536 | 涵盖全部 | PASS |
| N (ngram_size) | 2, 3, 4, 5 | 涵盖全部 | PASS |
| L (num_layers) | 1, 2, 4 | 涵盖全部 | PASS |
| T (num_tables) | 1, 4, 8, 16 | 涵盖全部 | PASS |
| Cores | 1, 8, 24, 48 | 涵盖全部 | PASS |

### 5.2 PyTorch 集成验证

7 个 torch 通路用例全部 bit-exact：
NT=32/256/1000/4096/65536, N=2/3/4/5, L=1/2/4, T=1/4/8/16 -- 全部 PASS。

### 5.3 边界验证

| 边界场景 | 参数 | 结果 |
|---------|------|:--:|
| 最小 W (N=2, T=1) | W=1 | PASS |
| 最小 L (N=2, L=1, T=1) | L=1, W=1 | PASS |
| 最大 W (N=5, L=4, T=16) | W=64 | PASS |
| 欠核 (NT=32, 48 cores) | 16 核空闲 | PASS |
| 尾核不整除 (NT=100, 48 cores) | 尾核 token=14 | PASS |
| 大 batch 多 tile (NT=65536) | tokensPerCore=1366 | PASS |

---

## 6. 性能分析

### 6.1 独立 benchmark 结果 (wall-clock, torch.ops)

| # | NT | N | L | T | W | Torch (ms) | Ascend (ms) | Speedup |
|:--:|--:|:-:|:-:|:-:|:-:|-----:|------:|------:|
| 1 | 32 | 3 | 2 | 8 | 16 | 0.645 | 0.088 | 7.30x |
| 2 | 256 | 3 | 2 | 8 | 16 | 0.646 | 0.084 | 7.72x |
| 3 | 1024 | 3 | 2 | 8 | 16 | 0.635 | 0.083 | 7.66x |
| 4 | 4096 | 3 | 2 | 8 | 16 | 0.667 | 0.090 | 7.44x |
| 5 | 65536 | 3 | 2 | 8 | 16 | 1.563 | 1.197 | 1.31x |
| 6 | 4096 | 5 | 4 | 16 | 64 | 1.737 | 0.565 | 3.08x |
| 7 | 256 | 5 | 4 | 16 | 64 | 1.756 | 0.091 | 19.31x |

**几何平均加速比**: **5.90x** vs PyTorch（与 Developer 自报 5.94x 基本一致）。

### 6.2 多核扩展验证

| 核数 | 1 | 8 | 24 | 48 |
|------|:-:|:-:|:-:|:-:|
| 理论加速比 | 1.00x | 8.00x | 24.00x | 48.00x |
| 实际加速比 (msprof kernel time) | 1.00x | 7.96x | 23.87x | 47.78x |
| 效率 | 100% | 99.5% | 99.5% | 99.5% |

### 6.3 瓶颈分析

| 指标 | 值 | 说明 |
|------|:--:|------|
| Scalar ratio | 99.8% | 标量流水线接近 100% 利用率 |
| Vector ratio | ~0.02% | 预期值（int64 无 Vector 指令） |
| MTE2 ratio | <1% | 访存不是瓶颈 |
| MTE3 ratio | <2% | 访存不是瓶颈 |
| ~cycles/output element | ~50 | 主要由 int64 `%`（多周期除法）构成 |

瓶颈确认为 **scalar-bound 的 int64 取模** -- 这是硬件物理限制，非代码优化可解决的问题。算子性能已达到该路线的天花板。

---

## 7. 问题列表

### 7.1 建议优化（非阻塞）

| # | 优先级 | 描述 | 建议 |
|---|:---:|------|------|
| S-1 | P2 | TilingData 含 4 个 kernel 未用字段 (`ngramPos`, `lastTileTokens`, `inAligned`, `outAligned`) | 可清理或加注释说明用途 |
| S-2 | P2 | README 未独立列出 API 映射表 | 可在 README 中增加 API 映射/约束小节或引用 DESIGN.md 6.3 |
| S-3 | P3 | torch 路径每调用 malloc/free device tiling buffer (~85us overhead) | 可缓存/复用 device tiling buffer 消除分配开销 |
| S-4 | P3 | benchmark.py 直调缩放探针使用 wall-clock（含 ACL init/teardown） | 改用 msprof kernel 时间作为缩放探针指标 |

### 7.2 无必须修复项

所有阻塞项（1.1/2.1/2.2/3.1/3.2/4.1/6.1）均通过，无必须修复问题。

---

## 8. 审查结论

| 项目 | 内容 |
|------|------|
| **总分** | **97 / 100** |
| **判定** | **PASS** |
| **必须修复问题** | 无 |
| **建议优化** | 4 条（P2-P3，非阻塞） |
| **精度验证** | 72/72 bit-exact + 7/7 torch bit-exact |
| **性能验证** | Geomean 5.90x vs PyTorch，多核扩展 99.5% |
| **代码清洁** | 通过（无调试残留） |
| **交付件** | 齐全（D1-D8） |

---

## 9. 审查记录

| 轮次 | 日期 | 阶段 | 判定 | 总分 |
|:--:|------|------|:--:|:--:|
| Round 0 | 2026-07-01 | 初审 | PASS | 97/100 |
