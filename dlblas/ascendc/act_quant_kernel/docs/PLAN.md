# act_quant_kernel 开发计划

## 1. 需求概述

| 项目 | 说明 |
|------|------|
| 算子名 | act_quant_kernel |
| 功能 | 将浮点激活值沿最后一维分组量化到 FP8 格式，输出量化值 x_q (fp8) 和 scale 因子 x_s (fp32) |
| 输入 | x [bf16/fp16], group_size (int), eps (float, default 1e-10), dtype (fp8_e4m3fn), scale_ue8m0 (bool) |
| 输出 | x_q [fp8], x_s [fp32] |
| 目标平台 | Ascend910B2 (DAV_2201), CANN 9.0.0 |
| 技术路线 | SIMD/MemBase, AR-FullLoad + Elementwise, Double Buffer + 3 级流水 |

## 2. 开发阶段

### Phase 1: 工程搭建 (预计 0.5 天)

**目标**: 建立可编译的 AscendC 算子工程骨架

**任务列表**:
- [ ] 1.1 创建算子工程目录结构（按 AscendC 工程模板规范）
- [ ] 1.2 配置 CMakeLists.txt（指定 `--npu-arch=dav_2201`, SocVersion=Ascend910B2）
- [ ] 1.3 编写 Host 侧入口 `op_host.cc`（算子注册、参数校验、Tiling 数据准备）
- [ ] 1.4 编写 Device 侧骨架 `op_kernel.asc`（空 kernel 函数，编译通过即可）
- [ ] 1.5 编译验证（`bash run.sh` 编译通过）

**关键里程碑**: 工程可编译，kernel 可加载（虽然无实际计算）

### Phase 2: Tiling 实现 (预计 0.5 天)

**目标**: 实现 Host 侧 Tiling 逻辑，正确切分任务

**任务列表**:
- [ ] 2.1 实现 Tiling 参数计算函数
  - `totalGroups = x.numel() / group_size`
  - `groupsPerCore = ceil(totalGroups / coreNum)`
  - `groupsPerBatch` 动态计算（取 UB 容量约束下的最大值）
  - `rLengthAlign = AlignUp(group_size * sizeof(float), 32) / sizeof(float)`
- [ ] 2.2 实现 TilingData 结构体序列化/反序列化
- [ ] 2.3 实现 Host → Device 的 Context 传递
- [ ] 2.4 单元测试: 验证不同 shape/group_size 下的 Tiling 参数正确性

**关键里程碑**: Tiling 参数在所有预期 shape 下正确

### Phase 3: 数据搬运与基础归约 (预计 1 天)

**目标**: 实现 GM↔UB 数据搬运和 ReduceMax 归约

**任务列表**:
- [ ] 3.1 实现 bf16/fp16 输入加载管道（DataCopy + Cast → fp32）
- [ ] 3.2 实现 fp32 中间结果的 Abs 计算
- [ ] 3.3 实现逐组 ReduceMax（Level 2 API, AR 模式）
- [ ] 3.4 实现 amax clamp(eps) 并计算 scale = amax / fp8_max
- [ ] 3.5 实现 x_s (scale) 的 UB→GM 写回
- [ ] 3.6 单元测试: 单 group 场景，验证 scale 与 PyTorch 参考一致

**关键里程碑**: 单 group 场景 scale 计算结果正确

### Phase 4: 量化计算 (预计 1 天)

**目标**: 实现完整量化管道 (broadcast scale → div → clamp → fp8)

**任务列表**:
- [ ] 4.1 实现 scalar scale → 向量广播（Duplicate）
- [ ] 4.2 实现逐元素除 scale（Div）
- [ ] 4.3 实现 clamp(fp8_min, fp8_max)（Compare + Select 组合）
- [ ] 4.4 实现 float32 → fp8_e4m3fn 软件转换（位操作）
  - 提取 float32 符号/指数/尾数
  - 映射到 e4m3 格式（1-4-3）
  - 特殊值处理（NaN, Inf, 次正规数, 饱和）
  - 舍入模式：就近舍入偶数
- [ ] 4.5 实现 x_q (int8) 的 UB→GM 写回
- [ ] 4.6 单元测试: 单 group 全流程，验证 x_q 与 PyTorch 参考一致（容许 ±1 ULP 差异）

**关键里程碑**: 单 group 场景全流程正确，FP8 精度匹配

### Phase 5: 多 Group 批处理与流水线 (预计 1 天)

**目标**: 实现多 group 批处理和 Double Buffer 流水线

**任务列表**:
- [ ] 5.1 实现多 group 循环（外层: batch 循环; 内层: 逐 group 计算）
- [ ] 5.2 实现 Double Buffer（`input_buf[2]`, `x_q_buf[2]`）
- [ ] 5.3 实现 3 级流水线同步（EnQue/DeQue）
- [ ] 5.4 实现尾块处理（mask 控制实际 group 数）
- [ ] 5.5 单元测试: 多 group 场景，验证批处理结果正确

**关键里程碑**: 多 group 批处理 + 流水线正确

### Phase 6: 多核并行 (预计 0.5 天)

**目标**: 实现多核切分和并行调度

**任务列表**:
- [ ] 6.1 实现沿 B 轴（group 维度）的多核任务分配
- [ ] 6.2 实现每核 GM 偏移计算（输入/输出 base address + core_offset）
- [ ] 6.3 确保无跨核数据依赖
- [ ] 6.4 多核场景集成测试

**关键里程碑**: 多核并行结果与单核一致

### Phase 7: scale_ue8m0 分支 (预计 0.5 天)

**目标**: 实现可选的 ue8m0 scale 舍入

**任务列表**:
- [ ] 7.1 实现 float32 → nearest pow2 (ceil) 位操作算法
- [ ] 7.2 根据 `scale_ue8m0` 参数执行分支
- [ ] 7.3 单元测试: 验证 scale_ue8m0 结果与 PyTorch 参考一致

**关键里程碑**: ue8m0 分支与 PyTorch 结果一致

### Phase 8: 集成测试与精度验证 (预计 0.5 天)

**目标**: 端到端测试，确认多场景正确性

**任务列表**:
- [ ] 8.1 编写多 shape 测试用例（覆盖 §3 测试用例表的所有场景）
- [ ] 8.2 编写 PyTorch 黄金参考脚本
- [ ] 8.3 精度比对: x_q (fp8) 容许 ±1 ULP; x_s (fp32) 容许 MSE < 1e-10
- [ ] 8.4 边界测试: eps=0, 全零输入, 极大值输入, 极小值输入
- [ ] 8.5 性能测试基线采集

**关键里程碑**: 所有测试用例精度通过

### Phase 9: 性能调优 (预计 0.5 天)

**目标**: 优化性能到可接受水平

**任务列表**:
- [ ] 9.1 Vector 指令利用率分析（减少标量操作，增加向量批量处理）
- [ ] 9.2 MTE 搬运效率优化（合并小搬运为大搬运）
- [ ] 9.3 流水线平衡（确保 CopyIn/Compute/CopyOut 时间均衡）
- [ ] 9.4 若 groupsPerBatch 受 UB 限制，评估拆分批次 vs 增大批次 tradeoff

**关键里程碑**: 算子延迟在目标 shape 下满足业务需求

## 3. 测试用例

### 3.1 基础功能测试

| 用例 | x shape | group_size | input dtype | scale_ue8m0 | 预期结果 |
|------|---------|-----------|-------------|-------------|---------|
| TC01 | [7, 512] | 128 | bf16 | false | 标准量化 |
| TC02 | [7, 512] | 512 | fp16 | false | 大 group |
| TC03 | [1, 1024] | 128 | bf16 | false | 多 group |
| TC04 | [7, 4096] | 512 | bf16 | false | 大 shape |
| TC05 | [32, 256] | 128 | fp16 | false | 多 token |

### 3.2 scale_ue8m0 测试

| 用例 | x shape | group_size | scale_ue8m0 | 预期结果 |
|------|---------|-----------|-------------|---------|
| TC06 | [7, 512] | 128 | true | scale 为 2 的幂次 |
| TC07 | [1, 1024] | 512 | true | scale 为 2 的幂次 |

### 3.3 边界测试

| 用例 | 场景 | group_size | 预期行为 |
|------|------|-----------|---------|
| TC08 | 全零输入 | 128 | scale=eps/fp8_max, x_q=0 |
| TC09 | 极大值输入 (1e5) | 512 | 正确 clamp 到 fp8_max |
| TC10 | 极小值输入 (1e-8) | 128 | scale 受 eps 保护 |
| TC11 | 负值混合 | 256 | 正确处理符号 |
| TC12 | 单 group (B=1) | 128 | 多核退化为单核 |

### 3.4 多核测试

| 用例 | totalGroups | coreNum | 预期 |
|------|-------------|---------|------|
| TC13 | < coreNum | 自动 | 部分核空闲，其余正确 |
| TC14 | >> coreNum | 自动 | 负载均衡 |

## 4. 风险点

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| **FP8 软件转换精度不足** | 量化结果与 PyTorch 不一致 | 中 | Phase 4 早期验证 float32→fp8 转换精度；参考 pyTorch 的 float8_e4m3fn 转换实现 |
| **Ceil/Log2/Exp2 API 缺失** | scale_ue8m0 需用位操作实现 | 高 | 已准备位操作备选方案，不依赖 Ceil/Log2/Exp2 API |
| **UB 容量不足** | Double Buffer 下 groupsPerBatch 过小 | 低 | 已计算 UB 预算，192KB 足够；若不足可降级为 Single Buffer |
| **非对齐数据搬运性能差** | DataCopyPad 开销大 | 低 | bf16 输入 32B 对齐概率高；必要时填充到对齐 |
| **repeatTimes 超限** | 影响循环展开效率 | 极低 | group_size ≤ 512 < 255×32B = 8160B，不受限 |
| **CANN 版本兼容性** | API 签名变化 | 低 | 已通过头文件验证当前 CANN 9.0.0 API 签名 |

## 5. 文件清单

```
operators/act_quant_kernel/
├── CMakeLists.txt               # 编译配置
├── op_host.cc                   # Host 侧入口 + Tiling
├── op_kernel.asc                # Device 侧 Kernel 实现
├── scripts/
│   └── run.sh                   # 编译运行脚本
├── test/
│   ├── test_op.py               # Python 测试入口
│   └── golden_ref.py            # PyTorch 黄金参考
└── docs/
    ├── DESIGN.md                # 本设计文档
    └── PLAN.md                  # 本开发计划文档
```

## 6. 时间线

| 阶段 | 内容 | 预计工期 |
|------|------|---------|
| Phase 1 | 工程搭建 | 0.5 天 |
| Phase 2 | Tiling 实现 | 0.5 天 |
| Phase 3 | 数据搬运与基础归约 | 1 天 |
| Phase 4 | 量化计算 | 1 天 |
| Phase 5 | 多 Group 批处理与流水线 | 1 天 |
| Phase 6 | 多核并行 | 0.5 天 |
| Phase 7 | scale_ue8m0 分支 | 0.5 天 |
| Phase 8 | 集成测试与精度验证 | 0.5 天 |
| Phase 9 | 性能调优 | 0.5 天 |
| **总计** | | **6 天** |
