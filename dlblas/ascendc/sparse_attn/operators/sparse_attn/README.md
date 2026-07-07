# SparseAttn 算子

稀疏注意力（Sparse Attention）算子，来自 DeepSeek-V4-Pro 推理内核。

## 算子定义

```
输入: q [b, m, h, d] bf16      — query
      kv [b, n, d] bf16         — shared key-value per position
      attn_sink [h] fp32        — learnable per-head sink bias
      topk_idxs [b, m, topk] i32 — sparse attention indices, -1=padding
      softmax_scale float        — typically d ** -0.5

输出: output [b, m, h, d] bf16
```

计算流程：Gather KV → Matmul-like (scores) → Mask → Softmax with Sink → Matmul-like (weighted sum) → Output

## 技术规格

| 项目 | 值 |
|------|-----|
| 芯片 | Ascend910B2 (DAV_2201) |
| CANN | 9.0.0 |
| Kernel 类型 | 纯 Vector (AIV only) |
| 并行策略 | 沿 (b, m) 多核切分 |
| 内部精度 | FP32 |
| I/O 精度 | BF16 |

## 文件结构

```
├── op_kernel/
│   ├── sparse_attn_kernel.asc    # kernel 实现
│   └── sparse_attn_tiling.h      # Tiling 数据结构
├── op_host/
│   ├── sparse_attn_runner.asc    # Host 入口
│   └── data_utils.h              # 文件 I/O 工具
├── op_extension/                  # PyTorch 接入层
├── scripts/                       # 测试脚本
├── CMakeLists.txt
└── run.sh
```

## 快速开始

```bash
# 1. 设置环境
source /usr/local/Ascend/cann-9.0.0/set_env.sh

# 2. 编译 + 测试
bash run.sh

# 3. 自定义 shape 测试
cd build
python3 ../scripts/gen_data.py        # 生成测试数据 (默认 shape)
./sparse_attn_custom <b> <m> <n> <h> <d> <topk>  # 运行 kernel
python3 ../scripts/verify_result.py output/output.bin output/golden.bin <b> <m> <h> <d>
```

## 精度

| 指标 | 值 | 阈值 | 状态 |
|------|-----|------|------|
| MERE | ~324 (受 bf16 precision floor 影响) | 0.0078 | FAIL* |
| MARE | 0.038 | 0.078 | PASS |
| MaxAbsErr | 0.016 | — | BF16 precision range |

*MERE 不通过是因为 golden 值接近零时相对误差被放大，绝对误差在 bf16 精度范围内。

## 默认配置

| 参数 | 默认值 |
|------|--------|
| b | 2 |
| m | 16 |
| n | 32 |
| h | 8 |
| d | 64 |
| topk | 16 |
