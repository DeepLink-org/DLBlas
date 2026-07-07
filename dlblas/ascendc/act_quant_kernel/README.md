# act_quant_kernel - Activation Per-Group FP8 Quantization

Ascend C 算子实现：对输入激活张量进行 per-group FP8 量化，输出量化值 (fp8_e4m3fn) 和 scale 因子 (fp32)。

## 算子语义

```
x (bf16/fp16, [..., N]) → x_q (fp8_e4m3fn, [... N]), x_s (fp32, [... N//group_size])

每组 group_size 个元素独立计算:
  amax[b]  = clamp(max(|x[b,j]|), min=eps)
  scale[b] = amax[b] / 448.0
  x_q[b,j] = clamp(x[b,j] / scale[b], -448, 448) → fp8_e4m3fn
```

## 文件结构

```
├── op_kernel/
│   ├── act_quant_kernel_tiling.h      Tiling 结构体 + 常量
│   └── act_quant_kernel_kernel.asc    Kernel 实现 (模板类, bf16/fp16)
├── op_host/
│   ├── act_quant_kernel.asc           Host + main (直调通路)
│   └── data_utils.h                   文件读写工具
├── op_extension/
│   ├── act_quant_kernel_torch.cpp     PyTorch 接入层
│   ├── register.cpp                   TORCH_LIBRARY 注册
│   └── ops.h                          函数声明
├── scripts/
│   ├── gen_data.py                    测试数据生成
│   ├── golden.py                      Golden 计算 (fp8 参考实现)
│   ├── verify_result.py              精度验证
│   └── test_torch.py                  PyTorch 通路测试
├── CMakeLists.txt                     构建配置
├── run.sh                             一键运行脚本
└── README.md
```

## 快速开始

### 前置条件

- CANN 9.0.0 + Ascend910B2
- PyTorch + torch_npu (PyTorch 通路)

### 方式一：直调验证

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
bash run.sh
```

或手动执行:

```bash
mkdir -p build && cd build && cmake .. && make -j4
python3 ../scripts/gen_data.py 2048 128
./act_quant_kernel 2048 128
python3 ../scripts/verify_result.py output/output_q.bin output/output_s.bin output/golden_q.bin output/golden_s.bin
```

### 方式二：PyTorch 调用

```python
import torch
import torch_npu

torch.ops.load_library("build/libact_quant_kernel_ops.so")

x = torch.randn(16, 128, dtype=torch.bfloat16).npu()
x_q, x_s = torch.ops.npu.act_quant_kernel(x, group_size=128, eps=1e-10, scale_ue8m0=False)

# x_q: fp8_e4m3fn (stored as uint8), same shape as x
# x_s: fp32 scale, shape [... N//group_size]
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| x | Tensor | - | 输入, bf16/fp16 |
| group_size | int | - | 每组元素数 |
| eps | float | 1e-10 | amax 下限 |
| scale_ue8m0 | bool | false | UE8M0 scale (未实现) |

## 关键技术决策

- **路线**: SIMD/MemBase (DAV_2201 AscendC Pipeline API)
- **归约**: AR 模式 ReduceMax (Level 2, 无对齐要求)
- **多核**: 按 groups 维度切分 (每组独立)
- **fp8 转换**: 软件实现 (DAV_2201 无硬件 fp8 Cast)
- **Double Buffer**: 输入/输出队列双缓冲

## 性能

| 指标 | 值 |
|------|-----|
| 测试规模 | 65,536 elements, 128 groups, bf16 |
| Kernel 耗时 | 66.9 us (47 核) |
| 瓶颈 | Scalar 计算 85.7% (fp32→fp8 转换) |

## 已知限制

1. fp32→fp8 转换存在 1-ULP 舍入差异 (DAV_2201 限制)
2. UE8M0 scale 格式未完整实现
3. fp16 输入路径未充分测试
