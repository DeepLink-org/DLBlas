# Expand Kernel Backward 算子

`expand_kenel_bwd` 是 Expand 操作的反向传播算子。给定前向 broadcast 的梯度输入 `o_grad`，沿 dim=-2 轴做 reduce sum。

```
forward:  input(n0, n1, h) → output(n0, n1, mhc_mult, h)  [broadcast expand]
backward: o_grad(n0, n1, mhc_mult, h) → sum(o_grad, dim=-2) → (n0, n1, h)
```

## 技术规格

| 项目 | 内容 |
|------|------|
| 算子类型 | Reduction (归约类) |
| 架构路线 | SIMD/MemBase (DAV_2201) |
| 数据流 | ARA-FullLoad |
| 典型输入 | `(2, 1024, 4, 1280)` FP16 |
| 输出 | `(2, 1024, 1280)` FP16 |
| 计算精度 | FP32 中间累加，输出截断为 FP16 |
| 多核策略 | 沿 A1=2048 均匀分配给 48 个 AIV 核 |

## 快速开始

### 环境要求

- CANN 9.0.0
- Ascend 910B2 (DAV_2201)
- CMake >= 3.16
- Python >= 3.8, PyTorch, torch_npu

### 编译与测试

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
bash run.sh
```

或分步执行：

```bash
mkdir -p build && cd build && cmake .. && make -j4
cd build && python3 ../scripts/gen_data.py
cd build && ./expand_kenel_bwd
cd build && python3 ../scripts/verify_result.py output/output.bin output/golden.bin
cd build && python3 ../scripts/test_torch.py
```

### PyTorch 调用

```python
import torch
import torch_npu

torch.ops.load_library("build/libexpand_kenel_bwd_ops.so")

o_grad = torch.randn(2, 1024, 4, 1280, dtype=torch.float16).npu()
output = torch.ops.npu.expand_kenel_bwd(o_grad)
# output.shape = (2, 1024, 1280)
```

## 文件结构

```
expand_kenel_bwd/
├── op_kernel/
│   ├── expand_kenel_bwd_tiling.h        # Tiling 结构体 (kernel/host 共用)
│   └── expand_kenel_bwd_kernel.asc      # Device Kernel
├── op_host/
│   ├── expand_kenel_bwd.asc             # Host 入口 + main
│   └── data_utils.h                     # 文件读写工具
├── op_extension/
│   ├── expand_kenel_bwd_torch.cpp       # PyTorch 接入层
│   ├── register.cpp                     # TORCH_LIBRARY 注册
│   └── ops.h                            # 函数声明
├── scripts/
│   ├── gen_data.py                      # 测试数据生成
│   ├── golden.py                        # Golden 参考计算 (FP32 累加)
│   ├── verify_result.py                 # 直调精度验证
│   └── test_torch.py                    # PyTorch 通路测试
├── CMakeLists.txt
├── run.sh
└── README.md
```

## 核心算法

合轴: `(n0, n1, mhc_mult, h)` -> `(A1=2048, R=4, A0=1280)`

每核处理 tilesPerCore 个 tile 的循环:

1. **CopyIn**: `DataCopyPad` GM->UB, 搬运 R x tileA0Len 个 half (4 行连续存放)
2. **Compute**: FP32 累加 `Cast(half->float) + Add(float) x3 + Cast(float->half)`
3. **CopyOut**: `DataCopyPad` UB->GM

Double Buffer (inQueueX(2) + outQueueY(2)) 实现 MTE/VEC 流水重叠。

## 性能

| 指标 | 数值 |
|------|------|
| Task Duration | 30.1 us (48 cores) |
| 瓶颈类型 | 内存带宽瓶颈 (MTE2=61.3%) |
| 精度 | max_diff = 7.81e-03 (FP16 截断误差) |

## 已知限制

- R 硬编码为 4 (当前需支持 R=4 场景)
- A0 需为 128 的整数倍
- 仅支持 FP16 数据类型
