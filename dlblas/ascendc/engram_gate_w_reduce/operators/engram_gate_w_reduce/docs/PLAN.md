# engram_gate_w_reduce 开发计划 (PLAN)

## 1. 需求概述

### 1.1 算子定义

```
grad_w_sum = sum(grad_w_partial, dim=0)           // [108, 4, H] → [4, H]
grad_weight_hidden += grad_w_sum ⊙ weight_embed   // [4, H] 广播乘加
grad_weight_embed += grad_w_sum ⊙ weight_hidden   // [4, H] 广播乘加
```

### 1.2 输入输出

| # | 名称 | Shape | dtype | 说明 |
|---|------|-------|-------|------|
| I1 | grad_w_partial | [108, 4, hidden_size] | float32 | 归约输入 |
| I2 | weight_hidden | [4, hidden_size] | bfloat16 | 乘法因子 |
| I3 | weight_embed | [4, hidden_size] | bfloat16 | 乘法因子 |
| I4 | grad_weight_hidden | [4, hidden_size] | float32 | in-place 累加目标，兼输出 |
| I5 | grad_weight_embed | [4, hidden_size] | float32 | in-place 累加目标，兼输出 |

### 1.3 技术路线

- **架构**: DAV_2201 (Ascend 910B2), CANN 9.0.0
- **编程模型**: AscendC SIMD / MemBase
- **融合策略**: 单 kernel 内完成 Reduction + Broadcast Elementwise

## 2. 测试用例

### 2.1 测试 Shape 矩阵

| Case | hidden_size | 说明 |
|------|-------------|------|
| TC01 | 64 | 小 shape，单核 / 少核 |
| TC02 | 256 | 小 shape |
| TC03 | 1024 | 中 shape |
| TC04 | 4096 | 典型场景（来自原始需求 generate_test_data） |
| TC05 | 8192 | 大 shape |

### 2.2 数据类型

| 输入 | dtype |
|------|-------|
| grad_w_partial | float32 |
| weight_hidden, weight_embed | bfloat16 |
| grad_weight_hidden, grad_weight_embed | float32 |

### 2.3 精度验收标准

| 指标 | 要求 |
|------|------|
| 验收方法 | 与 PyTorch 标杆对比（FP32 全精度） |
| 相对误差容限 | `1e-4` |
| 标杆策略 | PyTorch 原生 `sum(dim=0)` + `+=` 操作 |
| 异常值处理 | BF16→FP32 转换的固有精度损失允许 |

### 2.4 边界测试

| Case | 场景 | 说明 |
|------|------|------|
| TC11 | hidden_size=1 | 极小 shape，退化 case |
| TC12 | hidden_size=4 | 最小的 4 对齐 case |
| TC13 | hidden_size=13 | 非对齐 case |

### 2.5 功能验证检查项

- [ ] grad_w_sum 计算正确（与 PyTorch sum(dim=0) 对比）
- [ ] grad_weight_hidden 累加正确
- [ ] grad_weight_embed 累加正确
- [ ] in-place 语义正确（grad_weight_hidden/embed 原地修改）
- [ ] 各 hidden_size 测试用例全通过
- [ ] BF16 输入正确处理（类型转换无误）

## 3. 开发阶段

### Phase 0: 环境准备

**检查项**:
- [ ] 确认 CANN 9.0.0 环境变量配置
- [ ] 确认 AscendC 编译器（ascendc 命令行工具）可用
- [ ] 确认 AI Core 架构为 DAV_2201

### Phase 1: Tiling 实现 (Host 侧)

**文件**: `engram_gate_w_reduce_tiling.h` / `engram_gate_w_reduce.cpp` (Host)

**任务**:
- [ ] 实现 `EngramGateWReduceTiling` 结构体
  - `blockDim` 计算：`min(coreNum, hidden_size)`
  - `tileHiddenLen` 计算：`(hidden_size + blockDim - 1) / blockDim`
  - `tileA0Len` 计算：`tileHiddenLen * 4`
  - 尾部核的 tail 处理
- [ ] 实现 Host 侧算子入口函数
  - 获取平台信息 (`PlatformAscendC`)
  - 调用 `GetCoreNumAiv()` 获取可用 Vector 核数
  - 设置 TilingData
  - 调用 Kernel

**检查项**:
- [ ] Tiling 参数计算正确（各 hidden_size 下 blockDim 和 tileA0Len 合理）
- [ ] 尾部核 tileA0Len 正确（last core 处理更少的 hidden_size 段）

### Phase 2: Kernel 实现 (Device 侧)

**文件**: `engram_gate_w_reduce_kernel.asc` 或 `.cpp`

**子阶段 2.1: 基础框架**

- [ ] 定义 `KernelEngramGateWReduce` 类
- [ ] 使用 `KERNEL_LAUNCH` 宏
- [ ] 初始化 Pipe 和 TQue
- [ ] 从 TilingData 读取参数

**子阶段 2.2: Phase 1 — Reduction（逐行累加）**

- [ ] 创建 pingBuf/pongBuf (FP32) 和 accumBuf (FP32)
- [ ] 启用 Double Buffer (`SetQueue`)
- [ ] 实现逐行循环:
  - [ ] `EnQue` 预加载 row 0
  - [ ] `DeQue` + `Duplicate` 初始化 accumBuf（row 0）
  - [ ] 循环 row 1..107:
    - [ ] `EnQue` 加载下一行
    - [ ] `DeQue` 当前行
    - [ ] `Add` 累加到 accumBuf
  - [ ] `DeQue` 最后一行 + `Add` 最终累加
- [ ] 验证 accumBuf 逻辑（每列独立累加）

**子阶段 2.3: Phase 2 — Multiply-Accumulate**

- [ ] 从 GM 加载 weight_hidden/weight_embed 到 UB (BF16)
- [ ] `Cast<float, bfloat16_t>` 转换 BF16 → FP32（或 DataCopy+EnhancedParams 路径）
- [ ] 从 GM 加载 grad_weight_hidden/embed 到 UB (FP32)
- [ ] 使用 `MulAddDst<float, float>`:
  - `grad_weight_hidden += grad_w_sum * weight_embed_fp32`
  - `grad_weight_embed += grad_w_sum * weight_hidden_fp32`
- [ ] `DataCopy` 写回 GM

**检查项**:
- [ ] BF16→FP32 转换正确（Cast 或 DataCopyEnhancedParams 两种路径至少通一种）
- [ ] MulAddDst 语义验证（dst += src0 * src1）
- [ ] 双缓冲流水线无死锁
- [ ] 所有 tileA0Len 下 kernel 不溢出 UB

### Phase 3: 精度调试

**任务**:
- [ ] 构造标杆（PyTorch 实现）
- [ ] 单元测试框架搭建
- [ ] 逐 case 精度对比
- [ ] 精度不达标时诊断（按优先级）:
  1. 检查 BF16→FP32 转换是否正确
  2. 检查累加顺序（是否遗漏行、重复加）
  3. 检查 MulAddDst 输入顺序（weight_embed 和 weight_hidden 对应正确输出）

### Phase 4: 性能优化（可选）

**检查项**:
- [ ] Profiler 分析 kernel 执行时间
- [ ] 确认 Double Buffer 实际生效（GM→UB 搬运与计算重叠）
- [ ] 如有必要，尝试多行批量加载（如每次加载 2 行减少循环开销）
- [ ] 如有必要，Phase 2 增加流水线（加载 weight 与 MulAddDst 重叠）

## 4. 关键风险与应对

| # | 风险 | 影响 | 应对 |
|---|------|------|------|
| R1 | `bfloat16_t` 类型在 Cast/DataCopy 中支持不完整 | BF16 输入无法处理 | 优先使用 Cast 路径（通用）；若不行，Host 侧预转换 BF16→FP32 输入（需修改需求） |
| R2 | Double Buffer 流水线死锁 | Kernel 卡死 | 仔细核对 EnQue/DeQue 配对；先用单缓冲验证正确性再启用双缓冲 |
| R3 | 尾部核 tileA0Len 计算错误 | 最后核结果错误 | 独立测试尾部核路径，确保 tail 参数正确 |
| R4 | hidden_size 极大导致 UB 溢出 | 编译期校验失败 | 增加 `AS_LIMITED_CAPCITY` 静态检查 |

## 5. 交付物清单

| 文件 | 说明 |
|------|------|
| `docs/DESIGN.md` | 技术设计文档 |
| `docs/PLAN.md` | 开发计划文档（本文件） |
| `docs/perf/round_001/` | 性能采集数据 |
| `op_kernel/engram_gate_w_reduce_tiling.h` | Tiling 结构体（kernel + host 共用） |
| `op_kernel/engram_gate_w_reduce_kernel.asc` | Device 侧 Kernel 实现 |
| `op_host/engram_gate_w_reduce.asc` | Host 侧算子入口 + main |
| `op_host/data_utils.h` | 文件读写工具 |
| `scripts/gen_data.py` | 测试数据生成 |
| `scripts/golden.py` | Golden 计算 |
| `scripts/verify_result.py` | 精度验证 |
| `CMakeLists.txt` | CMake 构建配置 |
| `run.sh` | 一键运行脚本 |

## 6. 里程碑与测试结果

### 6.1 里程碑完成状态

| 里程碑 | 完成标准 | 状态 |
|--------|---------|------|
| M1: Tiling 实现 | Tiling 参数计算正确，所有 test case 下 blockDim / tileA0Len 合理 | ✅ 完成 |
| M2: Kernel 基本功能 | 所有 test case 精度通过（相对误差 ≤ 1e-4） | ✅ 完成 |
| M3: 性能验收 | 性能数据已采集，分析已记录 | ✅ 完成 |

### 6.2 测试结果

所有测试用例通过，精度 max_diff = 0.0（FP32 完美匹配）。

| Case | hidden_size | Status | blockNum | tileHiddenLen |
|------|-------------|--------|----------|---------------|
| TC11 | 1 | PASSED | 1 | 1 |
| TC12 | 4 | PASSED | 4 | 1 |
| TC13 | 13 | PASSED | 13 | 1 |
| TC01 | 64 | PASSED | 32 | 2 |
| TC02 | 256 | PASSED | 43 | 6 |
| TC03 | 1024 | PASSED | 47 | 22 |
| TC04 | 4096 | PASSED | 48 | 86 |
| TC05 | 8192 | PASSED | 48 | 171 |

### 6.3 PyTorch 集成测试

PyTorch TORCH_LIBRARY 集成测试通过，max_diff = 0.0。

| hidden_size | Status |
|-------------|--------|
| 4096 | PASSED |

### 6.4 性能数据 (hidden_size=4096, round_002)

| 指标 | 数值 |
|------|------|
| Task Duration | **186.3 us** |
| AIV 总时间 | 154.7 us |
| Vector 计算时间 | 7.9 us (5.1%) |
| MTE2 (GM→UB) 时间 | 129.8 us (83.9%) |
| Scalar 时间 | 7.9 us (5.1%) |
| MTE3 (UB→GM) 时间 | 0.9 us (0.6%) |
| 头开销 | 31.6 us (17.0%) |
| BlockDim | 48 |

**瓶颈分析**: Kernel 重度内存受限 (MTE2 占 83.9%)。主要时间花在 Phase 1 逐行数据加载上 (108 行 × 4 channel = 432 次 DMA)。优化方向可考虑:
- 增大 tileHiddenLen 以减少循环次数
- 使用 Double Buffer 流水线隐藏 DMA 延迟
- 多行批量加载

### 6.5 实现说明

1. **数据同步**: 使用 `PipeBarrier<PIPE_ALL>()` 保证 DataCopyPad (async DMA) 与 Vector 计算之间的同步。此方案虽非性能最优，但正确性优先。
2. **Per-channel 独立处理**: 避免 DataCopyPad multi-block stride 语义和 GetWithOffset 32B 对齐问题。
3. **BF16 转换**: 使用 `Cast<float, bfloat16_t>(RoundMode::CAST_NONE)` 进行 BF16→FP32 零损失转换。
4. **In-place 语义**: grad_weight_hidden/embed 直接在输入 buffer 上修改后读回。
5. **Tiling 修正**: 修正了原始公式中 tileHiddenLen 过大导致 tailHiddenLen 下溢的问题。
