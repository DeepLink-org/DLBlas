# Sinkhorn Normalize 算子

Sinkhorn 归一化算子的 AscendC Kernel 直调实现。将 4x4 矩阵 batch 经 Softmax + 迭代行列归一化转换为双随机矩阵。

## 算子说明

- **输入**: `[1, 1024, 4, 4]` float32
- **输出**: `[1, 1024, 4, 4]` float32 (双随机矩阵)
- **算法**: Softmax(dim=-1) + eps → 列归一化 → 重复 9 轮 (行归一化 + 列归一化)
- **常量**: repeat=10, eps=1e-6
- **目标芯片**: Ascend910B2 (DAV_2201)
- **精度**: MERE < 2^-13, MARE < 10 * 2^-13

## 文件结构

```
├── op_kernel/
│   ├── sinkhorn_tiling.h      Tiling 结构体 (kernel 和 host 共用)
│   └── sinkhorn_kernel.asc    纯 kernel 代码 (全载 UB 策略)
├── op_host/
│   ├── sinkhorn.asc           Host + main 入口
│   └── data_utils.h           数据读写工具
├── op_extension/
│   ├── sinkhorn_torch.cpp     PyTorch host 实现
│   ├── register.cpp           TORCH_LIBRARY 注册
│   └── ops.h                  函数声明
├── scripts/
│   ├── gen_data.py            测试数据生成
│   ├── golden.py              Golden 计算 (PyTorch 参考实现)
│   ├── verify_result.py       精度验证
│   └── test_torch.py          PyTorch 通路测试
├── CMakeLists.txt             双 target：可执行文件 + libsinkhorn_ops.so
├── run.sh                     一键运行脚本
└── docs/
    ├── DESIGN.md              技术设计方案
    └── PLAN.md                开发计划与结果记录
```

## 快速开始

### 方式一：直调验证

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
bash run.sh
```

### 方式二：PyTorch 调用

```python
import torch
import torch_npu

torch.ops.load_library("build/libsinkhorn_ops.so")

x = torch.randn(1, 1024, 4, 4, dtype=torch.float32).npu()
y = torch.ops.npu.sinkhorn_normalize(x)
```

## 技术要点

### 全载 UB 策略
- 每核一次性加载全部 tile 矩阵到 UB，计算完成后一次性写回
- 无 Double Buffer（数据量极小，流水线 setup 开销大于收益）

### 多核并行
- 沿 batch 维度均匀切分，各核独立处理
- 使用 47 个 AI Vector Core，每核处理 22 个矩阵

### Work Buffer 方案
- 使用合并临时缓冲区 + GetWithOffset 避免 DeQue 张量的 operator[] 问题
- 所有临时缓冲区 32 字节对齐

## 性能

| 指标 | 值 |
|------|-----|
| Kernel 执行时间 | 487.27 us |
| AI Vector Core 数 | 47 |
| Scalar 占比 | 77.7% |
| Vector 占比 | 15.6% |

## 精度

| 指标 | 标准 | 实测 |
|------|------|------|
| MERE (平均相对误差) | < 1.22e-04 | 1.13e-07 |
| MARE (最大相对误差) | < 1.22e-03 | 7.76e-07 |
| 行和 ≈ 1.0 | < 1e-4 | 8.65e-03 (与 golden 一致) |
| 列和 ≈ 1.0 | < 1e-4 | 1.13e-06 |
