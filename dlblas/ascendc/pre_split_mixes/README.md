# pre_split_mixes -- Ascend C 算子

## 概述

对 `input_mixes` 施加 per-channel scale+bias，经 sigmoid 激活和标量变换后拆分为 pre_mix、post_mix、comb_mix 三路输出。

## 算子签名

```
输入:
  input_mixes: [batch, seq_len, M3]  FP32
  mhc_scale:   [3]                   FP32  (权重)
  mhc_base:    [M3]                  FP32  (权重)
  mhc_mult:    int                   (m 值)
  mhc_pre_eps: float                 (默认 0.01)
  mhc_post_mult_value: float          (默认 2.0)

输出:
  pre_mix:  [batch, seq_len, m]      FP32
  post_mix: [batch, seq_len, m]      FP32
  comb_mix: [batch, seq_len, m*m]    FP32

其中 M3 = 2*m + m*m
```

## 构建与运行

```bash
# 设置环境
source /usr/local/Ascend/cann-9.0.0/set_env.sh

# 编译并运行 (默认测试用例 T2)
bash run.sh

# 指定测试用例
bash run.sh T3

# 跳过编译 (复用已有产物)
bash run.sh T2 --skip-build
```

## 测试用例

| 用例 | batch | seq_len | m | M3 | 说明 |
|------|-------|---------|---|----|------|
| T1 | 1 | 1 | 4 | 24 | 极小 shape |
| T2 | 1 | 1024 | 4 | 24 | 基准 |
| T3 | 8 | 512 | 4 | 24 | 大 batch |
| T4 | 1 | 2048 | 4 | 24 | 大 seq_len |
| T5 | 1 | 1024 | 1 | 3 | m=1 边界 |
| T6 | 1 | 1024 | 8 | 80 | m=8 |
| T7 | 1 | 1024 | 16 | 288 | m=16 |
| T8 | 2 | 256 | 4 | 24 | 小 batch x 小 seq_len |

## 文件结构

```
operators/pre_split_mixes/
├── op_kernel/
│   ├── pre_split_mixes_tiling.h     # Tiling 数据结构
│   └── pre_split_mixes_kernel.asc   # Kernel 实现
├── op_host/
│   ├── pre_split_mixes.asc          # Host 入口
│   └── data_utils.h                 # 文件 I/O 工具
├── op_extension/
│   ├── pre_split_mixes_torch.cpp    # PyTorch 扩展 (已知问题)
│   ├── register.cpp                 # TORCH_LIBRARY 注册
│   └── ops.h                        # 函数声明
├── scripts/
│   ├── gen_data.py                  # 测试数据生成
│   ├── golden.py                    # 参考实现
│   ├── verify_result.py             # 精度验证
│   └── test_torch.py                # PyTorch 测试
├── CMakeLists.txt
├── run.sh
└── README.md
```

## 实现细节

### 架构
- **目标**: DAV_2201 (Ascend910B2), CANN 9.0.0
- **路线**: SIMD/MemBase Elementwise
- **编译选项**: `--npu-arch=dav-2201`

### 算法
1. Per-channel scale+bias: `x = input_mixes * scale_cat + mhc_base`
2. Pre 段 (通道 0..m-1): `sigmoid(x) + mhc_pre_eps`
3. Post 段 (通道 m..2m-1): `sigmoid(x) * mhc_post_mult_value`
4. Comb 段 (通道 2m..M3-1): `x` (直接输出)

### Kernel 策略
- 逐行处理，每行分为 pre/post/comb 三段
- 每段独立执行 CopyIn (GM→UB) → Compute (scale+bias+sigmoid) → CopyOut (UB→GM)
- 权重 (per-segment scale/bias) 在 Init 时一次性加载并长期持有
- 多核按行切分，每核处理 `rowsPerCore` 行

## 精度

直接调用路径: 所有 8 个测试用例 max_diff=0.0 (完全一致), rtol=1e-4, atol=1e-6。

## 已知限制

1. PyTorch 扩展 (`torch.ops.npu`) 路径不可用，需使用直接调用路径验证
2. 极小 shape (totalRows=2) 单核多行处理有数据写入错误
3. Sigmoid 临时缓冲区使用硬编码 8KB 保守值

## 依赖

- CANN 9.0.0 (ASC 编译器 + ACL 运行时)
- bisheng (CANN 内置)
- Ascend910B2 (DAV_2201)
- cmake >= 3.16
- Python 3.10 + NumPy
