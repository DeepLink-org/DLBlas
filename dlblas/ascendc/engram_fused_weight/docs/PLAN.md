# engram_fused_weight 算子开发计划 (PLAN.md)

## 1. 需求概述

| 项目 | 描述 |
|------|------|
| 算子名 | engram_fused_weight |
| 数学定义 | `output = wh_data.float() * we_data.float()` |
| 输入1 | wh_data: bfloat16, shape=(4, 128) |
| 输入2 | we_data: bfloat16, shape=(4, 128) |
| 输出 | output: float32, shape=(4, 128) |
| 算子类型 | Elementwise 二元（逐元素乘 + BF16→FP32 类型提升） |
| 目标平台 | Ascend910B2, DAV_2201, CANN 9.0.0 |
| 技术路线 | SIMD/MemBase |
| 核心 API | DataCopy, Cast, Mul |
| 精度标准 | MERE < 2^-13, MARE < 10 × 2^-13 |

---

## 2. 开发阶段

### Phase 1: 工程搭建

| 任务 | 说明 | 预计完成 |
|------|------|----------|
| 1.1 创建算子目录 | `operators/engram_fused_weight/` 下建立标准 AscendC 工程结构 | □ |
| 1.2 配置 CMakeLists | 设置源文件、AscendC 头文件路径、--npu-arch=dav_2201_vec | □ |
| 1.3 Tiling 实现 | `tiling/engram_fused_weight_tiling.h` — 实现 `ComputeTiling()` 函数 | □ |
| 1.4 Host 侧实现 | `op_host/engram_fused_weight_host.cpp` — aclnn 算子注册、Tiling 调用 | □ |

### Phase 2: Device 侧实现

| 任务 | 说明 | 预计完成 |
|------|------|----------|
| 2.1 Kernel 主框架 | `op_kernel/engram_fused_weight_kernel.asc` — Pipeline 初始化, Process() 主循环 | □ |
| 2.2 CopyIn 实现 | wh_data/we_data 的 DataCopy (GM→UB)，双缓冲管理 | □ |
| 2.3 Compute 实现 | Cast(BF16→FP32) + Mul(FP32×FP32) | □ |
| 2.4 CopyOut 实现 | out_fp32 的 DataCopy (UB→GM)，双缓冲写回 | □ |

### Phase 3: 精度验证

| 任务 | 说明 | 预计完成 |
|------|------|----------|
| 3.1 Golden 脚本 | PyTorch CPU 参考实现 `enram_fused_weight_torch.py` | □ |
| 3.2 精度测试 | 运行 `mare_mere_threshold.py`，验证 MERE/MARE 达标 | □ |
| 3.3 边界测试 | 验证极小 shape、INF/NAN、全零等边界情况 | □ |

### Phase 4: 性能验证

| 任务 | 说明 | 预计完成 |
|------|------|----------|
| 4.1 Profiling 测试 | 使用 msprof 采集算子耗时 | □ |
| 4.2 性能基线 | 记录 op 延迟作为基线（预期 < 50us，数据量极小） | □ |

---

## 3. 测试用例

### 3.1 基础功能测试

| 用例ID | hc_mult | hidden_size | 说明 | 预期 |
|--------|---------|-------------|------|------|
| TC-01 | 4 | 128 | 标准用例（与需求一致） | Pass |
| TC-02 | 1 | 128 | 最小 hc_mult | Pass |
| TC-03 | 4 | 1 | 最小 hidden_size | Pass |
| TC-04 | 1 | 1 | 单元素 | Pass |
| TC-05 | 8 | 256 | 较大 shape（单核仍能处理） | Pass |

### 3.2 数据类型测试

| 用例ID | 输入 dtype | 输出 dtype | 说明 | 预期 |
|--------|-----------|-----------|------|------|
| TC-10 | bfloat16 | float32 | 标准类型组合 | Pass |

### 3.3 边界值测试

| 用例ID | 输入 | 说明 | 预期 |
|--------|------|------|------|
| TC-20 | 全零 | wh_data=0, we_data=0 | 输出全零 |
| TC-21 | 含 INF | 输入含 ±inf | 输出按 IEEE 754 |
| TC-22 | 含 NAN | 输入含 nan | 输出含 nan |
| TC-23 | 正负混合 | 含正负数 | 符号正确处理 |

### 3.4 精度测试

| 用例ID | 用例 | 指标 | 阈值 | 预期 |
|--------|------|------|------|------|
| TC-30 | TC-01 数据 | MERE | < 2^-13 (0.000122) | Pass |
| TC-31 | TC-01 数据 | MARE | < 10 × 2^-13 (0.00122) | Pass |
| TC-32 | 随机数据 ×5 | MERE/MARE | 同 TC-30/31 | 全 Pass |

---

## 4. 阶段检查项

### 4.1 开发前检查

- [ ] 确认 CANN 9.0.0 环境生效 (`source set_env.sh`)
- [ ] 确认 AscendC 头文件可正常 include
- [ ] 确认 `--npu-arch dav_2201_vec` 编译器支持
- [ ] 阅读 DESIGN.md §4 Tiling 设计（理解参数公式）
- [ ] 阅读 DESIGN.md §5 Buffer 规划（理解内存布局）

### 4.2 实现中检查

- [ ] TilingData 结构与 DESIGN.md §4.3 一致
- [ ] Host 侧 Tiling 计算逻辑与 DESIGN.md §10 模板一致
- [ ] Kernel 侧 Buffer 初始化大小与 DESIGN.md §5.3 计算一致
- [ ] Cast 使用 `RoundMode::CAST_NONE`（BF16→FP32 无损转换）
- [ ] Mul 三参数均为 float32 类型
- [ ] DataCopy 参数正确设置 blockCount/blockLen（1 / curLen / 0 / 0）
- [ ] Pipeline 同步使用 EnQue/DeQue/FreeTensor 标准模式
- [ ] 不使用禁止 API（GlobalTensor::SetValue/GetValue）

### 4.3 验证前检查

- [ ] Golden 脚本与算子语义完全一致 (`wh_data.float() * we_data.float()`)
- [ ] 测试数据覆盖正常值、边界值
- [ ] 精度验证脚本 (`mare_mere_threshold.py`) 可用

### 4.4 交付前检查

- [ ] 所有 TC-01 ~ TC-32 测试通过
- [ ] 代码无编译警告
- [ ] README 包含编译/运行说明

---

## 5. 文件规划

```
operators/engram_fused_weight/
├── CMakeLists.txt                              # 工程配置
├── op_host/
│   └── engram_fused_weight_host.cpp            # Host 侧算子注册 + Tiling
├── op_kernel/
│   └── engram_fused_weight_kernel.asc          # Device 侧 Kernel 实现
├── tiling/
│   └── engram_fused_weight_tiling.h            # TilingData 结构 + ComputeTiling()
├── test/
│   ├── engram_fused_weight_torch.py            # Golden 参考(PyTorch)
│   ├── test_engram_fused_weight.py             # 测试驱动
│   └── scripts/
│       └── mare_mere_threshold.py              # 精度验证脚本
└── docs/
    ├── DESIGN.md                               # 技术设计文档（本文档）
    └── PLAN.md                                 # 开发计划（本文件）
```

---

## 6. 风险与注意事项

| 风险 | 级别 | 缓解措施 |
|------|------|----------|
| 极小数据量（512元素）下 pipeline 无实际并行收益 | 低 | 不影响正确性，结构保留标准范式 |
| Cast API 的 RoundMode 在不同 CANN 版本中枚举值可能变化 | 低 | 查阅当前版本头文件确认，BF16→FP32 使用 CAST_NONE |
| 多输入 EleWise 无法直接复用 ElemwiseFrame 模板 | 中 | 手动管理双输入队列，参照 ElemwiseFrame 的 Pipeline 模式 |
