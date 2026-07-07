# engram_fused_weight - Ascend C 算子

## 概述

engram_fused_weight 是一个 Ascend C 逐元素乘法算子，对两个 bfloat16 输入做逐元素乘法，输出 float32 结果。

- **数学定义**: `output[i][j] = float32(wh_data[i][j]) * float32(we_data[i][j])`
- **输入**: wh_data (hc_mult x hidden_size, bf16), we_data (hc_mult x hidden_size, bf16)
- **输出**: (hc_mult x hidden_size, float32)
- **默认参数**: hc_mult=4, hidden_size=128 (dim0=512)
- **架构**: Ascend910B2 (DAV_2201), CANN 9.0.0
- **技术路线**: SIMD/MemBase, Direct Invoke

## 文件结构

```
├── op_kernel/
│   ├── engram_fused_weight_tiling.h    Tiling 结构体 + ComputeTiling()
│   └── engram_fused_weight_kernel.asc  纯 kernel 代码 (KernelEngramFusedWeight)
├── op_host/
│   ├── engram_fused_weight.asc         Host + main 入口 (Direct Invoke)
│   └── data_utils.h                    数据读写工具
├── op_extension/
│   ├── engram_fused_weight_torch.cpp   PyTorch 接入层
│   ├── register.cpp                    TORCH_LIBRARY 注册 (含 Meta backend)
│   └── ops.h                           函数声明
├── scripts/
│   ├── gen_data.py                     测试数据生成 (BF16 输入 + FP32 golden)
│   ├── golden.py                       Golden 计算 (FP32 逐元素乘)
│   ├── verify_result.py                精度验证 (FP32 阈值: 2^-13)
│   └── test_torch.py                   PyTorch 路径测试
├── CMakeLists.txt                      双 target: 可执行文件 + libengram_fused_weight_ops.so
├── run.sh                              一键运行 (编译 + 测试 + 验证)
└── README.md                           本文档
```

## 快速开始

### 环境准备

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
```

### 方式一：一键运行

```bash
cd operators/engram_fused_weight/
bash run.sh
```

### 方式二：分步执行

```bash
# 编译
mkdir -p build && cd build && cmake .. && make -j4 && cd ..

# 生成测试数据
python3 scripts/gen_data.py              # 生成 input/*.bin 和 output/golden.bin

# 运行 Direct Invoke
cd build && ./engram_fused_weight && cd ..

# 精度验证
python3 scripts/verify_result.py build/output/output.bin build/output/golden.bin

# PyTorch 路径测试
python3 scripts/test_torch.py
```

### 方式三：PyTorch 调用

```python
import torch
import torch_npu

torch.ops.load_library("build/libengram_fused_weight_ops.so")

wh = torch.randn(4, 128, dtype=torch.bfloat16).npu()
we = torch.randn(4, 128, dtype=torch.bfloat16).npu()
y = torch.ops.npu.engram_fused_weight(wh, we)  # 返回 float32

# 验证精度 (FP32 vs FP32)
golden = wh.float() * we.float()
assert torch.allclose(y.float().cpu(), golden, atol=2**-13, rtol=2**-13)
```

## 技术方案

| 项目 | 说明 |
|------|------|
| API | DataCopy (GM-UB), Cast (BF16->FP32, CAST_NONE), Mul (FP32) |
| Tiling | 单核 (coreNum=1), ubFormer <= 2048, ubLoop 自适应 |
| 数据流 | BF16 GM -(DataCopy)-> BF16 UB -(Cast)-> FP32 UB -(Mul)-> FP32 UB -(DataCopy)-> FP32 GM |
| Buffer | 5 个 UB Buffer (whQ, weQ, tmpWH, tmpWE, outQ), QUE_DEPTH=1 |
| 精度 | BF16 输入 -> FP32 内部计算 -> FP32 输出, 二进制精确 |

## 性能数据

| 指标 | 值 | 说明 |
|------|-----|------|
| Kernel 耗时 | **5.42 us** | 512 elements, msprof 采集 (device 1) |
| AIV Vector | 0.079 us (1.6%) | 纯 Mul 计算占比极低 |
| AIV Scalar | 2.688 us (54.1%) | Cast + 地址计算 |
| MTE2 (copy-in) | 0.560 us (11.3%) | BF16 输入搬运 |
| MTE3 (copy-out) | 0.331 us (6.7%) | FP32 输出搬运 |
| Cube | 0% | 无 Cube 操作 |

> 小数据量下为 I/O bound + Scalar bound (kernel launch overhead 和 Cast 操作主导)，符合预期。

## 精度标准

| 指标 | 阈值 | 实测值 | 说明 |
|------|------|--------|------|
| MERE (平均相对误差) | < 2^-13 (0.000122) | 0.0 | FP32 二进制精确 |
| MARE (最大相对误差) | < 10*2^-13 (0.00122) | 0.0 | 无舍入误差 |

> 精度阈值按输出 dtype (FP32) 选取。BF16->FP32 Cast 无损，FP32 Mul 为 IEEE 754 精确运算。

## 构建产物

| 文件 | 用途 |
|------|------|
| `build/engram_fused_weight` | Direct Invoke 可执行文件 |
| `build/libengram_fused_weight_ops.so` | PyTorch TORCH_LIBRARY 扩展库 |

## 设计偏差

| 项目 | DESIGN.md | 实际 | 理由 |
|------|-----------|------|------|
| QUE_DEPTH | 2 (双缓冲) | 1 (单缓冲) | 双缓冲在 multi-tile 场景存在同步问题 |
| coreNum | 动态计算 | 固定 1 | 数据量小，单核足够 |
| ubFormer | 公式推导 (max 65536) | 硬限制 2048 | DataCopy DAV_2201 实际限制 |
| --npu-arch | dav_2201_vec | dav-2201 | 编译器实际支持格式 |
| Kernel 入口 | __aicore__ | __vector__ | 纯 Vector 算子语义更正确 |
| DataCopy blockLen | 字节数 | 元素数 | API 规范：blockLen 单位为元素，非字节 |

## 已知限制

| 限制 | 说明 |
|------|------|
| 仅 BF16 输入 | 不支持 FP16 / FP32 输入，Check 在 TORCH_LIBRARY 层 |
| 超大 shape (dim0 > 28672) | ubLoop >= 15 时存在末 tile 数据损坏，属 DAV_2201 DMA 描述符资源限制 |
| 单核执行 | coreNum 固定为 1，未启用多核并行（小数据量下单核足够） |
| 性能加速比 | 512 elements 下 AscendC 比 CPU 慢（~0.6x），NPU launch overhead 占主导 |
