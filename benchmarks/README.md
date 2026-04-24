# benchmark_level3 使用说明

对 `kernelswift_triton/level3` 下的 Triton 算子与 `kernelswift_torch/level3` 下的 PyTorch
参考实现做延迟对比，输出加速比（Speedup = PyTorch 延迟 / Triton 延迟）。

## 目录结构

```
DLBlas/
├── benchmarks/
│   ├── benchmark_level3.py   ← 本脚本
│   └── README.md             ← 本文件
└── dlblas/kernels/
    ├── kernelswift_triton/level3/   Triton 实现 (class ModelNew)
    └── kernelswift_torch/level3/   PyTorch 参考实现 (class Model)
```

## 依赖

```
torch >= 2.0
triton >= 2.0
CUDA GPU
```

## 执行方式

```bash
# 进入脚本所在目录
cd DLBlas/benchmarks

# 测所有候选算子（默认 warmup=25, rep=100）
python benchmark_level3.py

# 只测指定算子（空格分隔）
python benchmark_level3.py sparse_attn MTPBlock

# 列出全部候选算子
python benchmark_level3.py --list

# 调整计时参数
python benchmark_level3.py --warmup 50 --rep 200

# 报错时打印完整 traceback
BENCH_VERBOSE=1 python benchmark_level3.py
```

## 候选算子

| 名称               | 文件                      | 说明                          |
|--------------------|---------------------------|-------------------------------|
| `sparse_attn`      | `9_sparse_attn.py`        | 稀疏注意力                    |
| `hc_split_sinkhorn`| `17_hc_split_sinkhorn.py` | HC 分解 + Sinkhorn 归一化     |
| `hc_post`          | `12_post.py`              | 残差混合后处理                |
| `indexer`          | `11_indexer.py`           | 稀疏注意力 TopK 索引器        |
| `MTPBlock`         | `10_MTPBlock.py`          | Multi-Token Prediction Block  |
| `act_quant_kernel` | `18_act_quant_fp8.py`     | FP8 激活量化                  |

## 测试结果

### 环境

| 项目        | 版本 / 型号                   |
|-------------|-------------------------------|
| GPU         | NVIDIA H200 (140 GB HBM3e)    |
| CUDA        | 12.8                          |
| Driver      | 570.133.20                    |
| PyTorch     | 2.9.1+cu128                   |
| Triton      | 3.5.1                         |

### 结果（warmup=25, rep=100，取中位延迟）

```
-----------------------------------------------------
Kernel              PyTorch(ms)  Triton(ms)   Speedup
-----------------------------------------------------
sparse_attn              0.2043      0.0068    29.89x
hc_split_sinkhorn        0.6065      0.0650     9.32x
hc_post                  0.5282      0.0484    10.90x
indexer                  0.2527      0.2273     1.11x
MTPBlock                 3.9962      3.1969     1.25x
act_quant_kernel         0.0563      0.0164     3.44x
-----------------------------------------------------
```

### 说明

- **sparse_attn（29.9x）**：Triton 稀疏注意力通过 TopK 索引直接跳过无效 KV，
  避免了 PyTorch 实现中全量 softmax 再掩码的冗余计算，加速最为显著。
- **hc_post（10.9x）/ hc_split_sinkhorn（9.3x）**：密集 einsum / softmax 操作
  融合进单个 Triton kernel，消除了多次 kernel launch 和中间 tensor 的显存读写。
- **act_quant_kernel（3.4x）**：FP8 量化 + scale 计算一次完成，
  PyTorch 版本需要多步 cast 和 reduction。
- **indexer（1.1x）/ MTPBlock（1.25x）**：这两个算子内部包含较多线性层（GEMM），
  bottleneck 在 cuBLAS，Triton 可优化的非线性部分占比小，加速有限。
