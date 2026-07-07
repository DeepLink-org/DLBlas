# expand_kenel_fwd

AscendC 算子实现：Tensor 维度扩展 (Broadcast/Expand)。

## 功能

```
PyTorch: x.unsqueeze(-2).expand(*shape[:-1], mhc_mult, shape[-1]).contiguous()
```

将输入 `(..., H)` 在倒数第二维插入新维度并扩展 `mhc_mult` 倍，输出 `(..., mhc_mult, H)`。

**特点**：纯数据搬运算子，无浮点计算，输出与输入 Bitwise Match。

## 支持的数据类型

| 类型 | 状态 |
|------|------|
| FP16 | 已验证 |
| FP32 | 已验证 |
| BF16 | 已验证 |

## 快速开始

### 环境准备

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
```

### 方式一：直调验证

```bash
bash run.sh [B] [S] [H] [M] [dtype]
# 默认: B=1 S=1024 H=1280 M=4 dtype=fp16
```

### 方式二：PyTorch 调用

```python
import torch
import torch_npu

torch.ops.load_library("build/libexpand_kenel_fwd_ops.so")
x = torch.randn(1, 1024, 1280, dtype=torch.float16).npu()
y = torch.ops.npu.expand_kenel_fwd(x, 4)  # mhc_mult=4
# y.shape = (1, 1024, 4, 1280)
```

## 构建

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

产物：
- `build/expand_kenel_fwd` — 直调可执行文件
- `build/libexpand_kenel_fwd_ops.so` — PyTorch 扩展库

## 文件结构

```
├── CMakeLists.txt
├── op_kernel/
│   ├── expand_kenel_fwd_tiling.h    # Tiling 结构体
│   └── expand_kenel_fwd_kernel.asc  # Device 侧 Kernel
├── op_host/
│   ├── expand_kenel_fwd.asc         # Host 直调入口
│   └── data_utils.h                 # 文件读写工具
├── op_extension/
│   ├── expand_kenel_fwd_torch.cpp   # PyTorch 接入层
│   ├── register.cpp                 # TORCH_LIBRARY 注册
│   └── ops.h                        # 函数声明
├── scripts/
│   ├── gen_data.py                  # 测试数据生成
│   ├── golden.py                    # 参考计算
│   ├── verify_result.py             # 精度验证
│   └── test_torch.py                # PyTorch 测试
├── run.sh                           # 一键运行
└── README.md
```

## 测试结果

10 个 PyTorch 通路测试用例全部 Bitwise Match (0 mismatch)：

| Case | B | S | H | M | dtype | Result |
|------|---|---|---|-----|-------|--------|
| T1 typical | 1 | 1024 | 1280 | 4 | FP16 | PASSED |
| T2 min rows | 1 | 1 | 128 | 2 | FP16 | PASSED |
| T3 multi rows | 4 | 256 | 256 | 2 | FP16 | PASSED |
| T4 large M | 1 | 1 | 1280 | 16 | FP16 | PASSED |
| T5 M=1 | 1 | 1 | 1280 | 1 | FP16 | PASSED |
| T6 FP32 | 1 | 1024 | 1280 | 4 | FP32 | PASSED |
| T7 boundary | 1 | 5 | 32 | 4 | FP16 | PASSED |
| T8 multicore | 10 | 100 | 512 | 8 | FP16 | PASSED |
| T9 large H | 1 | 1 | 2048 | 4 | FP16 | PASSED |
| T10 BF16 | 1 | 16 | 128 | 4 | BF16 | PASSED |

## 已知限制

- H 维度需为 16 的倍数（常见值 1280, 768, 2048, 4096 均满足）
- NPU 架构: DAV_2201 (Ascend910B2), CANN 9.0.0

## License

CANN Open Software License Agreement Version 2.0
