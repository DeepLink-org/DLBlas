# hc_split_sinkhorn 算子

将输入张量 `mixes` 拆分为三个分量并执行计算：
- **Pre**: `sigmoid(x * s0 + bias_pre) + eps`
- **Post**: `sigmoid(x * s1 + bias_post) * 2`
- **Comb**: Sinkhorn 迭代双随机归一化

## 文件结构

```
├── op_kernel/
│   ├── hc_split_sinkhorn_tiling.h    Tiling 结构体 + calcTileRows + ComputeTiling
│   └── hc_split_sinkhorn_kernel.asc  纯 kernel 代码
├── op_host/
│   ├── hc_split_sinkhorn.asc         Host 直调入口（main + ACL）
│   └── data_utils.h                  文件读写工具
├── op_extension/
│   ├── hc_split_sinkhorn_torch.cpp   PyTorch 调度 + ACL 内存管理
│   ├── register.cpp                  TORCH_LIBRARY 注册
│   └── ops.h                         函数声明
├── scripts/
│   ├── gen_data.py                   测试数据生成
│   ├── golden.py                     PyTorch 参考实现
│   ├── verify_result.py              精度验证（直调通路）
│   └── test_torch.py                 PyTorch 通路测试
├── CMakeLists.txt                    双 target 构建
└── run.sh                            一键运行
```

## 快速开始

### 环境要求

- CANN 9.0.0+, Ascend910B2 (DAV_2201)
- Python 3.10+, torch + torch_npu

### 直调验证

```bash
source ${ASCEND_HOME_PATH}/set_env.sh

# 完整流程（编译 + 测试）
bash run.sh

# 指定测试参数
cd build
python3 ../scripts/gen_data.py <b> <s> <hc> <iters> <eps> [seed]
ASCEND_RT_VISIBLE_DEVICES=<device_id> ./hc_split_sinkhorn
python3 ../scripts/verify_result.py
```

### PyTorch 调用

```python
import torch, torch_npu

torch.ops.load_library("build/libhc_split_sinkhorn_ops.so")

# mixes: (b, s, mix_hc), hc_scale: (3,), hc_base: (mix_hc,)
hc = 4; mix_hc = (2 + hc) * hc

mixes = torch.randn(2, 8, mix_hc, dtype=torch.float32).npu()
hc_scale = torch.tensor([1.0, 1.0, 0.5], dtype=torch.float32).npu()
hc_base = torch.randn(mix_hc, dtype=torch.float32).npu()

pre = torch.empty(2, 8, hc, dtype=torch.float32).npu()
post = torch.empty(2, 8, hc, dtype=torch.float32).npu()
comb = torch.empty(2, 8, hc, hc, dtype=torch.float32).npu()

torch.ops.npu.hc_split_sinkhorn(
    mixes, hc, 20, 1e-6, hc_scale, hc_base,
    pre, post, comb)
```

## 设计文档

- 技术设计: [docs/DESIGN.md](docs/DESIGN.md)
- 开发计划: [docs/PLAN.md](docs/PLAN.md)
- 性能数据: [docs/perf/round_002/](docs/perf/round_002/)

## 测试结果

| Case | 配置 | MERE (pre) | MERE (post) | MERE (comb) |
|------|------|-----------|-------------|-------------|
| C1 | b=2,s=8,hc=4,iters=20 | 8.77e-09 | 6.08e-09 | 6.89e-08 |
| C4 | b=1,s=1,hc=1,iters=5 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| C5 | b=8,s=4,hc=8,iters=20 | 9.53e-09 | 8.43e-09 | 8.99e-08 |
| C6 | b=64,s=8,hc=4,iters=20 | 8.68e-09 | 8.65e-09 | 7.71e-08 |

精度阈值: MERE < 1.22e-4 (浮点社区标准), 实测约为阈值的 1/10000。

## 性能 (C1, NPU 2)

| 指标 | 值 |
|------|-----|
| Task Duration | 24.22 us |
| Block Dim | 16 |
| Vector Utilization | 14.8% |
| Scalar Ratio | 83.7% |

瓶颈: Sinkhorn 列归一化的逐元素 SetValue/GetValue (Scalar-bound)。
