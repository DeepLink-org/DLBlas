# Sinkhorn 算子开发计划 (PLAN.md)

> **关联设计文档**: `operators/sinkhorn/docs/DESIGN.md`
> **目标芯片**: Ascend910B2 (DAV_2201)
> **CANN**: 9.0.0

---

## 1. 需求概述

| 项目 | 内容 |
|------|------|
| 算子名称 | Sinkhorn Normalize |
| 数学定义 | 将 4x4 矩阵 batch 经 Softmax + 迭代行列归一化转换为双随机矩阵 |
| 输入 | `[1, 1024, 4, 4]` float32 |
| 输出 | `[1, 1024, 4, 4]` float32 |
| 常量 | repeat=10, eps=1e-6 |
| 精度标准 | 浮点计算类社区标准: MERE < 2^-13, MARE < 10 * 2^-13 |

---

## 2. 算子分解

### 2.1 子功能模块

| 模块 | 功能 | 依赖 |
|------|------|------|
| **Tiling 参数计算** | 多核切分、UB Buffer 大小计算 | 无 |
| **数据搬运** | GM→UB CopyIn, UB→GM CopyOut | Tiling |
| **Softmax 行计算** | ReduceMax→Sub→Exp→ReduceSum→Div (逐行) | 数据搬运 |
| **行归一化** | ReduceSum→Muls (逐行) | Softmax |
| **列归一化** | 列收集→ReduceSum→Mul (逐矩阵) | 行归一化 |
| **主循环控制** | repeat 次迭代编排 | 全部 |

### 2.2 关键 API 清单

| API | 用途 | 所在模块 |
|-----|------|---------|
| `DataCopyPad` | GM↔UB 数据搬运 | 数据搬运 |
| `ReduceMax` (Level 2) | Softmax 行最大值 | Softmax |
| `ReduceSum` (Level 2) | Softmax/归一化 行/列求和 | Softmax, 归一化 |
| `Exp` | Softmax 指数运算 | Softmax |
| `Adds` | 加标量 (eps / -max) | Softmax |
| `Muls` | 乘标量 (1/sum) | Softmax, 行归一化 |
| `Mul` | 逐元素乘 (列倒数广播) | 列归一化 |

---

## 3. 开发任务分解

### Phase 0: 工程搭建

| # | 任务 | 产出物 | 预估 |
|---|------|--------|------|
| 0.1 | 创建 CMakeLists.txt (顶层 + 子目录) | CMakeLists.txt × 3 | 1h |
| 0.2 | 创建算子注册配置文件 | `sinkhorn_config.json` + `sinkhorn_op_info.json` | 0.5h |
| 0.3 | 编写测试数据生成脚本 | `scripts/gen_test_data.py` | 0.5h |

### Phase 1: Host 侧 Tiling 实现

| # | 任务 | 产出物 | 预估 |
|---|------|--------|------|
| 1.1 | 定义 `SinkhornTilingData` 结构体 | `op_host/sinkhorn_tiling.h` | 0.5h |
| 1.2 | 实现 Tiling 参数计算 (`InitTilingData`) | `op_host/sinkhorn.cpp` | 1h |
| 1.3 | 实现 Kernel 启动函数 | `op_host/sinkhorn.cpp` | 1h |
| 1.4 | 编译 Host 侧目标 | `libsinkhorn_host.so` | 0.5h |

### Phase 2: Device 侧 Kernel 实现

| # | 任务 | 产出物 | 预估 |
|---|------|--------|------|
| 2.1 | 实现 Kernel 类框架 (Init/Process) | `op_kernel/sinkhorn_kernel.h` | 1h |
| 2.2 | 实现 Softmax(dim=-1) 计算逻辑 | 同上 (Phase 1 部分) | 1.5h |
| 2.3 | 实现列归一化 (sum dim=-2) | 同上 (Phase 2 部分) | 1.5h |
| 2.4 | 实现行归一化 (sum dim=-1) | 同上 (Phase 3a 部分) | 0.5h |
| 2.5 | 实现主循环迭代编排 | 同上 (Phase 3 整体) | 0.5h |
| 2.6 | 实现数据搬运 (CopyIn/CopyOut) | 同上 (Phase 0/4 部分) | 1h |
| 2.7 | 编译 Device 侧目标 | `libsinkhorn_kernel.so` | 0.5h |

### Phase 3: 测试与验证

| # | 任务 | 产出物 | 预估 |
|---|------|--------|------|
| 3.1 | 编写单元测试用例 (含边界) | `test_sinkhorn.py` | 1h |
| 3.2 | 功能正确性验证 (vs PyTorch 标杆) | 测试报告 | 1h |
| 3.3 | 精度验证 (MERE/MARE) | 精度报告 | 0.5h |
| 3.4 | 性能测试 (profiling) | 性能数据 | 0.5h |

### Phase 4: 优化 (按需)

| # | 任务 | 说明 | 触发条件 |
|---|------|------|---------|
| 4.1 | 向量化行操作 | 合并邻接行的 Reduce 操作 | 性能不达预期 |
| 4.2 | 列收集优化 | 使用 DataCopy stride 替代逐元素 GetValue | 列归约成为瓶颈 |
| 4.3 | 多核负载均衡调优 | 调优 usedCoreNum | 核利用率低 |

---

## 4. 测试用例设计

### 4.1 核心功能用例

| 用例 ID | 输入形状 | repeat | eps | 验证点 |
|---------|---------|--------|-----|--------|
| TC001 | [1, 1, 4, 4] | 10 | 1e-6 | 单矩阵基本功能 |
| TC002 | [1, 1024, 4, 4] | 10 | 1e-6 | 全量 batch 功能 |
| TC003 | [1, 1, 4, 4] | 1 | 1e-6 | repeat=1 (仅 Softmax+列归一化) |
| TC004 | [1, 1, 4, 4] | 2 | 1e-6 | repeat=2 (Softmax+列+行+列) |
| TC005 | [1, 1, 4, 4] | 10 | 0.0 | eps=0 (无 epsilon 保护) |

### 4.2 边界用例

| 用例 ID | 输入形状 | 场景 |
|---------|---------|------|
| TC006 | [1, 1024, 4, 4] | 全零输入 |
| TC007 | [1, 1024, 4, 4] | 极大正值 (1000) |
| TC008 | [1, 1024, 4, 4] | 极负值 (-1000) |
| TC009 | [1, 63, 4, 4] | batch 不可整除 coreNum |
| TC010 | [1, 1, 4, 4] | 单核最小输入 |

### 4.3 精度用例

| 用例 ID | 验证指标 | 标准 |
|---------|---------|------|
| TC011 | MERE vs PyTorch | < 2^-13 (≈ 0.000122) |
| TC012 | MARE vs PyTorch | < 10 × 2^-13 (≈ 0.00122) |
| TC013 | 行和接近 1.0 | max(|row_sum - 1.0|) < 1e-4 |
| TC014 | 列和接近 1.0 | max(|col_sum - 1.0|) < 1e-4 |
| TC015 | 双随机性验证 | 同时满足行和≈1 且列和≈1 |

### 4.4 测试数据生成

使用 PyTorch 生成随机数据并保存为二进制文件：

```python
# gen_test_data.py
import torch
import numpy as np

def gen_test_cases():
    cases = [
        {"shape": [1, 1, 4, 4], "dtype": "float32", "seed": 42},
        {"shape": [1, 1024, 4, 4], "dtype": "float32", "seed": 42},
        {"shape": [1, 63, 4, 4], "dtype": "float32", "seed": 42},
        {"shape": [1, 1, 4, 4], "dtype": "float32", "seed": 42, "special": "zeros"},
        {"shape": [1, 1, 4, 4], "dtype": "float32", "seed": 42, "special": "large_values"},
    ]
    for i, case in enumerate(cases):
        if "special" in case:
            if case["special"] == "zeros":
                x = torch.zeros(case["shape"])
            elif case["special"] == "large_values":
                x = torch.ones(case["shape"]) * 1000
        else:
            torch.manual_seed(case["seed"])
            x = torch.randn(case["shape"])
        # Save x
        np.save(f"input_{i}.npy", x.numpy())
        # Compute reference output using PyTorch Model
        model = Model(repeat=10, eps=1e-6)
        y = model(x)
        np.save(f"golden_{i}.npy", y.detach().numpy())
```

---

## 5. 阶段检查项

### 5.1 Phase 1 检查项 (Host Tiling)

- [ ] `SinkhornTilingData` 结构体字段完整且类型正确
- [ ] `blockCount` ≤ 4095 约束已处理 (通过 `tile_batch ≤ 255`)
- [ ] 尾核 tailBatch 计算正确
- [ ] Tiling 参数通过 Context 正确传递给 Kernel
- [ ] Host 侧编译通过 (无警告)

### 5.2 Phase 2 检查项 (Device Kernel)

- [ ] CopyIn 搬运数据量与 Tiling 参数一致
- [ ] Softmax 每行使用 `int32_t` 类型的 count 参数
- [ ] `ReduceSum`/`ReduceMax` 的 tmpBuffer 类型与 `float` 一致
- [ ] `rowTmpBuf` 使用 stride=2 float 存储 (满足 8B dst 对齐)
- [ ] `colTmpBuf` 使用 stride=2 float 存储
- [ ] 列收集不涉及 `GlobalTensor::GetValue/SetValue` (仅 `LocalTensor`)
- [ ] 迭代循环范围正确 (repeat-1 次行+列归一化)
- [ ] CopyOut 搬运数据量与 Tiling 参数一致
- [ ] 所有 `DataCopyPad` 使用 `DataCopyExtParams` + `DataCopyPadExtParams`
- [ ] Device 侧编译通过 (无警告, `-D__NPU_ARCH__=2201`)

### 5.3 Phase 3 检查项 (测试验证)

- [ ] 全部功能用例通过 (输出 shape 正确, 无 NaN/Inf 异常)
- [ ] 精度验证: MERE < 0.000122, MARE < 0.00122
- [ ] 双随机性验证: |行和-1| < 1e-4, |列和-1| < 1e-4
- [ ] 多核场景验证 (不同核数场景均通过)

---

## 6. 文件清单

```
operators/sinkhorn/
├── CMakeLists.txt
├── op_host/
│   ├── sinkhorn_tiling.h
│   ├── sinkhorn.cpp
│   └── CMakeLists.txt
├── op_kernel/
│   ├── sinkhorn_kernel.h
│   └── CMakeLists.txt
├── scripts/
│   └── gen_test_data.py
└── docs/
    ├── DESIGN.md
    └── PLAN.md
```

### 6.1 文件依赖关系

```
sinkhorn_tiling.h        (无依赖)
       │
       v
sinkhorn.cpp             → sinkhorn_tiling.h
       │
       v
sinkhorn_kernel.h        → sinkhorn_tiling.h
       │
       v
CMakeLists.txt × 3       → 编译上述源文件
gen_test_data.py         (独立)
```

---

## 7. 风险识别

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|---------|
| ReduceSum 在 count=4 时行为异常 | 低 | 功能错误 | 先用单矩阵单元测试验证 |
| 列收集 GetValue/SetValue 性能瓶颈 | 低 | perf 劣化 | 数据量极小 (≤ 16 ops/matrix) |
| blockCount 超 4095 | 低 | 编译/运行错误 | Tiling 中 clip tile_batch ≤ 255 |
| repeat=10 迭代精度累积 | 中 | 精度劣化 | 每步加 eps 保护, 精度测试覆盖 |
| 纯零输入 softmax 除零 | 中 | NaN 输出 | 标注为已知边界，eps 提供基本保护 |
| CANN 9.0.0 API 签名与设计假设不一致 | 低 | 编译错误 | API 已通过头文件验证，编译阶段可发现 |

---
## 8. 开发结果记录

### 8.1 实现完成情况

| 模块 | 状态 | 备注 |
|------|------|------|
| Tiling 结构体 | 已完成 | `sinkhorn_tiling.h`，支持多核沿 batch 维度切分 |
| Host 侧 Tiling 计算 | 已完成 | `sinkhorn.asc`，47 核，tileBatch=22 |
| Kernel 核心逻辑 | 已完成 | `sinkhorn_kernel.asc`，全载 UB 策略 |
| CMakeLists.txt | 已完成 | 双目标：可执行文件 + libsinkhorn_ops.so |
| PyTorch 扩展 | 已完成 | TORCH_LIBRARY 注册，8/8 测试通过 |
| 测试脚本 | 已完成 | gen_data.py, golden.py, verify_result.py, test_torch.py |

### 8.2 测试结果摘要

| 测试项 | 结果 | 详情 |
|--------|------|------|
| 编译 (可执行文件) | PASS | sinkhorn_custom 编译成功 |
| 编译 (PyTorch .so) | PASS | libsinkhorn_ops.so 编译成功 |
| TC001 单矩阵 | PASS | max_diff=5.96e-08 |
| TC002 全量 1024 batch | PASS | max_diff=1.79e-07 |
| TC006 零值 | PASS | max_diff=0 |
| TC007 极大正值 | PASS | max_diff=0 |
| TC008 极负值 | PASS | max_diff=0 |
| TC009 非整除 batch (63) | PASS | max_diff=1.19e-07 |
| PyTorch 接入 | PASS | 8/8 用例全部通过 |
| MERE (平均相对误差) | PASS | 1.13e-07 < 0.000122 |
| MARE (最大相对误差) | PASS | 7.76e-07 < 0.00122 |
| 双随机性 | PASS | 行和/列和约等于 1.0 |

### 8.3 性能数据

| 指标 | 值 |
|------|-----|
| Kernel 执行时间 | 487.27 us |
| 使用 AI Vector Core 数 | 47 |
| AIV Scalar 占比 | 77.7% |
| AIV Vector 占比 | 15.6% |
| AIV MTE 占比 | < 1% |

**性能分析**:
- Scalar 占比高 (77.7%) 是因为当前采用 work buffer 方案（GetValue/SetValue 逐元素拷贝），每行/每列计算都需要拷贝进出工作缓冲区
- 根本原因：`LocalTensor::operator[]` 在 DeQue 出来的张量上会产生错误行为（输出全零），必须使用 work buffer 绕开
- 优化方向：使用 RepeatReduce/BlockReduce API 直接处理带 stride 的数据，避免元素级拷贝

### 8.4 已知问题与偏离

| 偏离项 | 原因 | 影响 |
|--------|------|------|
| 使用 work buffer 替代子张量 | DeQue 张量 operator[] 存在 bug | Scalar 性能占比升高，但功能正确 |
| 融合临时缓冲区为单个 TBuf | 多个独立 TBuf 存在未知限制 | 需要手动管理偏移和对齐 |
| TBuf::GetWithOffset 需要 32B 对齐 | AscendC 对齐约束 | 缓冲区布局已调整为 8-float 对齐 |
| repeat/eps 硬编码 | 设计固定常量 | 不支持运行时修改参数 |
