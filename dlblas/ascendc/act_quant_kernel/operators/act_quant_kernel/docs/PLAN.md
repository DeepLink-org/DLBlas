# act_quant_kernel 开发计划 (PLAN.md)

> **状态**: 开发完成 | **目标平台**: Ascend910B2 (DAV_2201) | **CANN**: 9.0.0

---

## 1. 需求概述

实现 AscendC kernel `act_quant_kernel`, 对输入激活张量进行 per-group 量化:

- **输入**: x (bf16/fp16, 任意维度, 末维 N 可被 group_size 整除)
- **输出1**: x_q (fp8_e4m3fn, 同 x shape)
- **输出2**: x_s (fp32, scale 因子)
- **核心逻辑**: per-group absmax → scale → quantize
- **特殊路径**: 可选 UE8M0 scale 格式

---

## 2. 开发阶段划分

### Phase 0: 环境验证与模板搭建 (1-2 天)

**目标**: 可编译的空 kernel 骨架

- [ ] 从 `$ASCEND_TOOLKIT_HOME/tools/op_project_templates/ascendc/elemwise/` 复制模板
- [ ] 修改 CMakeLists.txt, 适配 CANN 9.0.0 和 Ascend910B2
- [ ] 修改 `CMakeLists.txt` 使 npu-arch 指定为 DAV_2201
- [ ] 创建 `act_quant_kernel.cpp` (kernel 入口) 和 `act_quant_kernel_tiling.h` (tiling 数据)
- [ ] 验证编译通过 (空 compute 逻辑)

**交付物**: 可编译的 op 项目骨架

### Phase 1: 数据搬运验证 (1 天)

**目标**: 确认 GM↔UB 搬运正确

- [ ] 实现 CopyIn: DataCopyPad 将输入 x 从 GM 搬到 UB (单 tile)
- [ ] 实现 CopyOut: DataCopyPad 将 UB 数据写回 GM (x_q, x_s)
- [ ] 使用单 tile (tile_groups=1) 验证搬运正确性
- [ ] 验证 bf16 和 fp16 两种数据类型搬运

**交付物**: Passthrough kernel (直接 CopyIn→CopyOut 不做计算)

### Phase 2: 核心计算实现 (2-3 天)

**目标**: 完整计算流程走通

**2.1 Abs + ReduceMax (1 天)**
- [ ] 实现 Abs: 对 UB 数据做逐元素绝对值
- [ ] 实现 ReduceMax: 对每组 group_size 个元素做 max reduction
- [ ] 验证: 与 PyTorch `x_.abs().max(dim=-1)` 结果对比

**2.2 Scale 计算 (0.5 天)**
- [ ] 实现 clamp(amax, min=eps) → fp32
- [ ] 实现 scale = amax / fp8_max
- [ ] 验证: 与 PyTorch scale 结果对比

**2.3 Quantize 计算 (1 天)**
- [ ] 实现 Cast bf16/fp16 → fp32
- [ ] 实现 broadcast 除法 (x_fp32 / scale)
- [ ] 实现 clamp (fp8_min, fp8_max)
- [ ] 实现 fp32 → fp8_e4m3fn 转换
- [ ] 验证: 与 PyTorch quantize 结果对比

**2.4 UE8M0 路径 (0.5 天)**
- [ ] 实现 exp2(ceil(log2(max(|s|, 1e-10)))) 计算
- [ ] 验证: scale 结果符合 UE8M0 格式

**交付物**: 单 tile 完整计算 kernel

### Phase 3: Tiling 与多核 (1-2 天)

**目标**: 完整的多 tile 迭代和多核并行

- [ ] 实现 Host 侧 tiling 参数计算 (`ActQuantTiling`)
- [ ] 实现 kernel 侧的 tile 循环
- [ ] 实现多 VectorCore 并行 (按 num_groups 切分)
- [ ] 处理末 tile 尾部对齐 (DataCopyPad 自动处理)
- [ ] 验证: 多核结果与单核一致

**交付物**: 完整的多核 kernel

### Phase 4: 性能优化 (2-3 天)

**目标**: 打开双缓冲, 优化 UB 利用率和计算效率

- [ ] 开启 Double Buffer (inQueueX, outQueueQ)
- [ ] 优化 tile_groups 大小 (平衡 UB 使用和 loop 开销)
- [ ] 优化 BinaryRepeatParams 批量广播除 (避免逐组循环)
- [ ] 尝试 Abs 与输入搬运重叠 (如硬件支持)
- [ ] Profile 并识别瓶颈

**交付物**: 优化后的高性能 kernel

### Phase 5: 精度验证与边界测试 (1-2 天)

**目标**: 全面的精度和边界验证

- [ ] 多 shape 精度测试 (≥ 5 种 shape)
- [ ] 边界值测试: eps 极小/极大, group_size 极小/极大
- [ ] 特殊值测试: 输入含 0, Inf, NaN
- [ ] bf16 vs fp16 精度对比
- [ ] UE8M0 vs 常规 scale 格式精度对比
- [ ] 精度报告输出

**交付物**: 精度验证报告

### Phase 6: 代码清理与文档 (0.5 天)

**目标**: 生产就绪代码

- [ ] 代码注释完善
- [ ] 移除调试代码
- [ ] 添加 README
- [ ] 最终集成测试

---

## 3. 关键里程碑

| 里程碑 | 预计耗时 | 判定标准 |
|--------|---------|---------|
| M1: 骨架编译 | Phase 0 完成 | CMake 构建成功, 无编译错误 |
| M2: 搬运通路 | Phase 1 完成 | CopyIn→CopyOut 不改变数据 |
| M3: 计算正确 | Phase 2 完成 | 单 tile 计算结果与 PyTorch 一致 (fp8 1-ULP) |
| M4: 多核跑通 | Phase 3 完成 | 多核并行结果与单核一致 |
| M5: 性能达标 | Phase 4 完成 | Double Buffer 生效, 搬运与计算重叠 |
| M6: 精度验证 | Phase 5 完成 | ≥ 5 种 shape 全部通过精度标准 |
| M7: 交付就绪 | Phase 6 完成 | 代码规范, 文档齐全 |

---

## 4. 验证策略

### 4.1 功能测试用例

| # | Shape | group_size | dtype_in | scale_ue8m0 | 验证点 |
|---|-------|-----------|----------|-------------|--------|
| 1 | [1, 128] | 128 | bf16 | false | 最小 shape, 1 group |
| 2 | [16, 128] | 128 | bf16 | false | 多 group, 常规场景 |
| 3 | [32, 64] | 64 | fp16 | false | fp16 输入 |
| 4 | [4, 32, 64] | 32 | bf16 | false | 3D 输入, 小 group |
| 5 | [128, 256] | 256 | bf16 | false | 大 group_size |
| 6 | [16, 128] | 128 | bf16 | true | UE8M0 路径 |
| 7 | [1, 65536] | 128 | bf16 | false | 大 N, 多 tile |
| 8 | [1024, 16] | 16 | fp16 | true | 小 group, UE8M0 |

### 4.2 精度测试标准

- **量化输出 (fp8)**: 与 PyTorch 标杆比对, 容许最大绝对误差 ≤ 1 (以 fp8 位表示为 uint8 的差值)
- **Scale 输出 (fp32)**: 与 PyTorch 标杆比对, RMSE ratio ≤ 1.2
- **UE8M0 输出**: exp2(ceil(log2(...))) 结果是精确的 2 的幂, 对比时容许浮点舍入误差

### 4.3 性能测试方法

- 使用 `msprof` 或 `profiler` 工具测量 kernel 延迟
- 对比单核 vs 多核性能, 验证并行效率
- 对比 Double Buffer 开启/关闭的延迟, 验证隐藏搬运的效果

### 4.4 边界测试

- 输入含 Inf/NaN 时的行为
- eps=0 (最小保护)
- group_size=1 (每组 1 个元素, 退化为逐元素量化)
- 极大 group_size (如 65536)

---

## 5. 风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| DAV_2201 向量 Cast 不支持 fp8 目标类型 | 中 | 高 | 兜底: 手写 fp32→fp8 位转换函数; 已在 DESIGN.md §7 提供算法 |
| ReduceMax 在大 group_size 下性能差 | 低 | 中 | group_size 通常 ≤ 128; 大 group_size 时可分 chunk (ColSplit) |
| Double Buffer 事件 ID 不够 | 低 | 中 | A2/A3 系列支持 8 个 eventID; 本设计使用 3 个 TQue (2 个 Double, 1 个 Single), 占 5 个 ID, 安全 |
| UE8M0 的 Exp/Ln 计算在 Vector 上精度不稳定 | 低 | 低 | 可改用 CPU 端预计算或查表法; 或手写位操作提取 exponent |
| 多核切分尾部不均匀导致部分核空闲 | 低 | 低 | 负载不均衡 < 2%, 可接受 |
| bf16 ReduceMax 与 fp32 ReduceMax 精度差异影响 scale | 低 | 低 | bf16 absmax 与 fp32 差异 < 1 ULP; eps clamp 提供额外保护 |

---

## 6. 依赖与前置条件

| 依赖项 | 版本 | 说明 |
|--------|------|------|
| CANN | 9.0.0 | Ascend C 编译器 + 运行时 |
| CMake | ≥ 3.16 | 构建系统 |
| Python | ≥ 3.8 | 测试驱动 (PyTorch 标杆) |
| PyTorch | ≥ 2.0 | 标杆实现 |
| Ascend910B2 | DAV_2201 | 目标硬件 |

---

## 7. 文件清单

```
operators/act_quant_kernel/
├── CMakeLists.txt                    # CMake 构建配置
├── op_common.mk                      # (可选) 公共构建规则
├── act_quant_kernel.cpp              # Kernel 入口 (host 侧 tiling + kernel 注册)
├── act_quant_kernel_tiling.h         # Tiling 数据结构定义
├── act_quant_kernel_vec.h            # VectorCore 计算实现
├── act_quant_kernel_fp8_convert.h    # fp32↔fp8 转换工具函数 (如需要)
├── scripts/
│   └── test_act_quant.py             # Python 测试脚本 (PyTorch 标杆比对)
├── docs/
│   ├── DESIGN.md                     # 架构设计文档 (本文)
│   └── PLAN.md                       # 开发计划文档 (本文件)
└── output/                           # (构建产物)
```

---

## 8. 实现结果 (2026-06-30)

### 8.1 实现摘要

| 项目 | 状态 |
|------|------|
| 编译 | 通过 (ASC executable + libact_quant_kernel_ops.so) |
| 直调通路 (direct invoke) | 通过 |
| PyTorch 通路 (TORCH_LIBRARY) | 通过 |
| Level 0 测试 (8-128 元素) | 全部通过 |
| Level 1 测试 (512-16384 元素) | 全部通过 |
| Level 2 测试 (极值/零值) | 全部通过 |
| 非对齐场景 (group_size=2) | 通过 |
| 性能采集 | 完成 (见 docs/perf/round_001/) |

### 8.2 测试结果详情

| # | Shape | group_size | 结果 | x_q 偏差 | x_s 偏差 |
|---|-------|-----------|------|---------|---------|
| 1 | 128 el, 1 group | 128 | PASS | 0 ULP | 0.00 |
| 2 | 8 el, 4 groups | 2 | PASS | 0 ULP | 0.00 |
| 3 | 512 el, 8 groups | 64 | PASS | 0 ULP | 0.00 |
| 4 | 2048 el, 16 groups | 128 | PASS | 3/2048 @ 1-ULP | 0.00 |
| 5 | 1024 el, 32 groups | 32 | PASS | 0 ULP | 0.00 |
| 6 | 2048 zeros | 128 | PASS | 0 ULP | 0.00 |
| 7 | 2048 extreme | 128 | PASS | 0 ULP | 0.00 |
| 8 | 16384 el, 128 groups | 128 | PASS | 9/16384 @ 1-ULP | 0.00 |
| T1 | [1,128] bf16 | 128 | PASS (PyTorch) | 0 | 0.000000 |
| T2 | [8,128] bf16 | 128 | PASS (PyTorch) | 0 | 0.000000 |

### 8.3 性能数据

| 指标 | 值 |
|------|-----|
| 测试规模 | 65536 elements, 128 groups |
| Kernel 耗时 | 66.9 us |
| 使用核数 | 47 VectorCores |
| 主要瓶颈 | Scalar 计算 (85.7%, fp32→fp8 转换) |

### 8.4 已知限制

1. **fp32→fp8_e4m3fn 转换**: DAV_2201 不支持硬件 fp8 Cast API，使用标量软件实现，存在 1-ULP 舍入差异
2. **UE8M0 路径**: 代码框架已预留，但 Ln/Exp 向量 API 在标量场景下使用受限，未完整验证
3. **fp16 输入**: 代码框架支持 fp16 输入 (模板类)，但未做完整的 fp16 测试

### 8.5 二次验证结果 (2026-06-30, Round 2)

基于 REVIEW.md (PASS 92/100) 后，进行了完整独立的二次验证。

#### 8.5.1 编译验证

- `cmake .. && make -j4` 零错误零警告通过
- 产物: `act_quant_kernel` (可执行文件) + `libact_quant_kernel_ops.so` (共享库)

#### 8.5.2 精度测试 (独立执行)

| # | Elements | group_size | Mode | x_q | x_s | Status |
|---|----------|-----------|------|-----|-----|--------|
| 1 | 128 | 128 | random | 0 ULP | max_diff=0 | PASS |
| 2 | 16 | 2 | random | 0 ULP | max_diff=0 | PASS |
| 3 | 512 | 64 | random | 0 ULP | max_diff=0 | PASS |
| 4 | 2048 | 128 | random | 3/2048 @ 1-ULP | max_diff=0 | PASS |
| 5 | 1024 | 32 | random | 0 ULP | max_diff=0 | PASS |
| 6 | 16384 | 128 | random | 9/16384 @ 1-ULP | max_diff=0 | PASS |
| 7 | 2048 | 128 | zeros | 0 ULP | max_diff=0 | PASS |
| 8 | 2048 | 128 | extreme | 1/2048 @ 1-ULP | max_diff=0 | PASS |
| 9 | 65536 | 128 | random | 20/65536 @ 1-ULP | max_diff=0 | PASS |

全部 9 个测试用例通过。fp8 (x_q) 0-ULP 或 1-ULP 差异 (可接受), fp32 scale (x_s) 精确匹配。

#### 8.5.3 PyTorch 通路验证

| Test | Shape | dtype | x_q | x_s | Status |
|------|-------|-------|-----|-----|--------|
| T1 | [1, 128] | bf16 | 0 mismatch | max_diff=0.000000 | PASS |
| T2 | [8, 128] | bf16 | 1 @ 1-ULP | max_diff=0.000000 | PASS |

#### 8.5.4 Benchmark 性能 (AscendC vs PyTorch NPU)

| Shape | AscendC (us) | TorchNPU (us) | Speedup | Status |
|-------|-------------|---------------|---------|--------|
| 1K, gs=128 | 121.60 | 291.52 | 2.40x | PASS |
| 4K, gs=128 | 118.40 | 277.10 | 2.34x | PASS |
| 16K, gs=128 | 116.04 | 299.14 | 2.58x | PASS |
| 65K, gs=128 | 121.85 | 297.98 | 2.45x | PASS |
| 256K, gs=128 | 185.63 | 311.31 | 1.68x | PASS |

**几何平均加速比: 2.26x**

#### 8.5.5 msprof op 深度分析 (Round 002)

| 指标 | 值 |
|------|-----|
| Task Duration | 49.54 us (47 cores) |
| AIV Scalar Time | 46.5 us (97.6%) |
| AIV Vector Time | 1.48 us (3.1%) |
| AIV MTE2 Time | 0.30 us (0.6%) |
| AIV MTE3 Time | 0.25 us (0.5%) |

瓶颈: Scalar 计算占 97.6% (fp32→fp8 逐元素转换)。结论与 Round 001 一致: DAV_2201 无硬件 fp8 Cast 的固有限制。

### 8.6 设计问题 (design_issue)

无。实现严格遵循 DESIGN.md 的设计框架：
- SIMD/MemBase 路线 ✓
- AR 模式 ReduceMax (Level 2) ✓
- per-group 独立处理 ✓
- 多核按 groups 切分 ✓
- Double Buffer 已使能 ✓
- 32B 对齐 DataCopyPad ✓
