# head_compute_mix_fwd — Ascend C 算子

## 算子功能

计算混合头前向传播：

```
output = sigmoid(input_mix * mhc_scale + mhc_base) + mhc_pre_eps
```

## 接口定义

### 输入

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `input_mix` | [batch, n1, mhc_mult] | FP16 | 主输入张量 |
| `mhc_scale` | [1] | FP16 | 标量缩放因子 |
| `mhc_base` | [mhc_mult] | FP16 | 逐通道偏置（mhc_mult=4） |
| `mhc_pre_eps` | scalar | FP32 | eps 常量 |

### 输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `output` | [batch, n1, mhc_mult] | FP16 | 计算结果 |

## 实现方案

- **架构**: Ascend910B2 (DAV_2201), CANN 9.0.0
- **路线**: 通用 SIMD/MemBase Vector 路线
- **策略**: 3D 张量展平为 1D Elementwise 处理
- **精度**: FP16 输入/输出，FP32 中间计算（Sigmoid 链全程 FP32）
- **并行**: 多核切分（Elementwise 标准 tiling），双缓冲流水线

## 精度标准

| 数据类型 | RTOL | ATOL |
|---------|------|------|
| FP16 | 1e-2 | 1e-3 |

## 性能

| 指标 | 值 |
|------|-----|
| Shape | [16, 16384, 4] (1M 元素) |
| 核数 | 48 |
| Duration | 51.921 us |
| vs CPU | ~1792x |

## 已知限制

1. **仅支持 FP16 dtype**: 算子输入/输出固定为 FP16，不兼容 FP32 或 BF16 输入。精度验证以 PyTorch FP32 参考实现为标杆。
2. **mhc_mult 必须为 4**: 由于 mhc_base 扩展逻辑（SetValue 循环）固定以 4-by-4 模式填充 UB tile，当前仅支持 mhc_mult=4 的输入 shape。
3. **DAV_2201 平台限制**: UbFormer 扩展（Duplicate tensor-to-tensor）在 DAV_2201 上不可用，mhc_base 扩展使用手动 SetValue 循环（每核执行一次，9728 次迭代，性能影响可忽略）。
4. **小 shape 性能**: 当 blockSize 远小于 ubFormer=9728 时，单个 core 的 tile 利用率降低，边缘 shape（如 [1, 128, 4]）的核级并行度受限于 dim0 大小。

## 编译和运行

```bash
cd operators/head_compute_mix_fwd
bash run.sh
```

### 单独步骤

```bash
# 编译
mkdir -p build && cd build && cmake .. && make -j4

# 生成测试数据
cd build && python3 ../scripts/gen_data.py 16 16384 4

# 运行算子
./head_compute_mix_fwd 16 16384 4

# 精度验证
python3 ../scripts/verify_result.py output/output.bin output/golden.bin

# PyTorch 集成测试
python3 ../scripts/test_torch.py
```

## PyTorch 使用

```python
import torch
import torch_npu

torch.ops.load_library("build/libhead_compute_mix_fwd_ops.so")

x = torch.randn(16, 16384, 4, dtype=torch.float16).npu()
scale = torch.randn(1, dtype=torch.float16).npu()
base = torch.randn(4, dtype=torch.float16).npu()
eps = 0.01

y = torch.ops.npu.head_compute_mix_fwd(x, scale, base, eps)
```

## 文件结构

```
operators/head_compute_mix_fwd/
├── CMakeLists.txt
├── run.sh
├── README.md
├── op_kernel/
│   ├── head_compute_mix_fwd_tiling.h    # Tiling 数据结构（kernel/host 共用）
│   └── head_compute_mix_fwd_kernel.asc  # Kernel 实现
├── op_host/
│   ├── head_compute_mix_fwd.asc         # Host 入口 + main
│   └── data_utils.h                     # 文件读写工具
├── op_extension/
│   ├── head_compute_mix_fwd_torch.cpp   # PyTorch 接入层
│   ├── register.cpp                     # TORCH_LIBRARY 注册
│   └── ops.h                            # 函数声明
├── scripts/
│   ├── gen_data.py                      # 测试数据生成
│   ├── golden.py                        # Golden 参考计算
│   ├── verify_result.py                 # 精度验证
│   └── test_torch.py                    # PyTorch 通路测试
└── docs/
    ├── DESIGN.md                        # 架构设计
    ├── PLAN.md                          # 开发计划与结果
    └── perf/                            # 性能采集数据
```
