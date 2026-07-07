# norm_fn 算子

## 概述

`norm_fn` 是一个融合算子，实现 einsum 点积 + RMS 归一化 + 可选权重乘法的端到端计算。

- **芯片**: Ascend 910B2 (DAV_2201)
- **路线**: SIMD/MemBase (单核 Vector)
- **精度**: FP32 输出，Max Diff < 4.6e-5 (满足 DESIGN.md Max Diff < 1e-4 标准)

## 数学定义

```
输入:
  residual: (1, 13, 4, 1280) bf16
  mhc_fn: (24, 5120) float32
  weight: (5120,) float32 (可选)

计算:
  1. 可选权重乘法: mhc_fn *= weight
  2. residual flatten → (13, 5120) float32
  3. mixes[m,n] = sum_k residual[m,k] * mhc_fn[n,k]  (einsum 'mk,nk->mn')
  4. rms[m] = rsqrt(sum_k residual[m,k]^2 / 5120 + eps)
  5. result[m,n] = mixes[m,n] * rms[m]

输出:
  result: (1, 13, 24) float32
```

## 文件结构

```
operators/norm_fn/
├── docs/
│   ├── DESIGN.md                 # 技术设计文档
│   ├── PLAN.md                   # 开发计划与结果
│   └── perf/round_001/           # 性能采集数据
│   └── perf/round_002/           # 性能采集数据 (Rsqrt 版本)
├── op_host/
│   ├── norm_fn_tiling.h          # Tiling 参数定义
│   ├── norm_fn.asc               # Host 侧算子入口
│   └── data_utils.h              # 文件读写工具
├── op_device/
│   └── norm_fn_kernel.asc        # Device 侧 Kernel 实现
├── op_extension/
│   ├── ops.h                     # PyTorch 函数声明
│   ├── norm_fn_torch.cpp         # PyTorch 接入层
│   └── register.cpp              # TORCH_LIBRARY 注册
├── scripts/
│   ├── gen_data.py               # 测试数据生成
│   ├── golden.py                 # 参考实现 (NumPy)
│   ├── verify_result.py          # 精度验证
│   └── test_torch.py             # PyTorch 通路测试
├── CMakeLists.txt                # CMake 构建配置
├── run.sh                        # 一键运行脚本
└── README.md                     # 本文件
```

## 编译与运行

### 前置条件

- CANN 9.0.0 或更高版本
- Ascend 910B2 NPU
- bisheng 编译器

### 编译

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cd operators/norm_fn
bash run.sh
```

### 仅生成测试数据

```bash
cd build
python3 ../scripts/gen_data.py            # 无权重
python3 ../scripts/gen_data.py --with-weight  # 有权重
```

### 仅运行算子

```bash
cd build
./norm_fn
```

### 精度验证

```bash
cd build
python3 ../scripts/verify_result.py output/output.bin output/golden.bin
```

### PyTorch 通路

```python
import torch
import torch_npu

torch.ops.load_library("build/libnorm_fn_ops.so")

residual = torch.randn(1, 13, 4, 1280, dtype=torch.bfloat16).npu()
mhc_fn = torch.randn(24, 5120, dtype=torch.float32).npu()
weight = torch.randn(5120, dtype=torch.float32).npu()

result = torch.ops.npu.norm_fn(residual, mhc_fn, weight, 1e-6)
# or without weight:
# result = torch.ops.npu.norm_fn(residual, mhc_fn, None, 1e-6)
```

## 技术要点

### Tiling 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| total_M | 13 | 输出行数 |
| total_N | 24 | 输出列数 |
| total_K | 5120 | 内维大小 |
| TILE_K | 512 | K 轴分块大小 |
| num_K_tiles | 10 | K 轴分块数 |
| Block Dim | 1 | 单核算子 |

### API 使用

| 操作 | API |
|------|-----|
| 数据搬运 | `DataCopyPad` (Ext 版本) |
| 精度转换 | `Cast<float, bfloat16_t>` (CAST_NONE) |
| 逐元素乘 | `Mul` |
| 批量归约 | `ReduceSum<Pattern::Reduce::AR>` |
| 逐行归约 | `ReduceSum<float>` (Level 2) |
| 标量乘 | `Muls` |
| 平方根 | `Rsqrt` (倒数平方根，融合 Sqrt+Div) |

### 精度

- 输入: residual bf16 (7 位尾数), mhc_fn float32
- 中间计算: 全部 float32
- 输出: float32
- RMS 归一化: 使用 `Rsqrt` 融合指令 (替代 Sqrt+Div)
- 实测精度: Max Diff < 4.6e-5 (满足 DESIGN.md Max Diff < 1e-4 标准)

## 性能

| 指标 | 数值 (Round 002) |
|------|------|
| Task Duration | 378 us |
| Block Dim | 1 (单核) |
| Vector 利用率 | 58.7% |
| FP32 向量利用率 | 39.6% |
| 总浮点操作 | 4.40M |

详细性能数据见 `docs/perf/round_002/summary.txt`。
