# Clike-AscendC

Clike-AscendC 是 KernelSwift Clike 华为昇腾赛道三个算子的 AscendC
实现，面向 Ascend 910B（Atlas A2），覆盖 SparseAttention、Indexer 和
Sinkhorn。本文档汇总其代码位置、优化方法及配套说明。

## 代码位置

下表中的路径均以 DLBlas 仓库根目录为基准。

| 算子 | 赛事参考实现 | 优化后的 Python 入口 |
|---|---|---|
| SparseAttention | `dlblas/kernels/ks_competition/torch/sparse_attention.py` | `dlblas/kernels/ks_competition/ascend/clike_sparse_attention.py` |
| Indexer | `dlblas/kernels/ks_competition/torch/indexer.py` | `dlblas/kernels/ks_competition/ascend/clike_indexer.py` |
| Sinkhorn | `dlblas/kernels/ks_competition/torch/sinkhorn.py` | `dlblas/kernels/ks_competition/ascend/clike_sinkhorn.py` |

三个优化入口均提供赛事评测工具要求的 `ModelNew`、`get_inputs` 和
`get_init_inputs`。公共 AscendC 实现及构建支持的位置如下：

| 内容 | 路径 |
|---|---|
| SparseAttention AscendC kernel | `dlblas/kernels/ks_competition/ascend/clike_910b/csrc/sparse_attention.asc` |
| Indexer AscendC kernel | `dlblas/kernels/ks_competition/ascend/clike_910b/csrc/indexer.asc` |
| Sinkhorn AscendC kernel | `dlblas/kernels/ks_competition/ascend/clike_910b/csrc/sinkhorn.asc` |
| PyTorch 动态库加载器 | `dlblas/kernels/ks_competition/ascend/clike_910b/loader.py` |
| CMake 配置 | `dlblas/kernels/ks_competition/ascend/clike_910b/CMakeLists.txt` |
| 构建脚本 | `dlblas/kernels/ks_competition/ascend/clike_910b/build.sh` |
| 性能测试脚本 | `dlblas/kernels/ks_competition/ascend/clike_910b/run_benchmarks.sh` |
| 赛事性能工具 | `benchmarks/ks/auto_bench.py` |

仓库中的 Task02 参考实现相比赛题文档将 `.cuda()` 改为 `.npu()`，将全局
`ModelArgs` 实例改为函数内按需构造，并直接使用 `torch.bfloat16` 表示默认
类型，以兼容赛事工具的安全 AST 加载。其 `Model`、`forward`、配置值、
数据类型、随机数调用、计算顺序、mask 和 TopK 写法均保持赛事 reference
的定义。

## 优化说明

### SparseAttention

- 该实现由每个 Vector core 将当前 batch 的 `32 x 128` KV 表驻留在 UB。
- 两级 `TQueBind` 流水用于重叠 gather 搬入与搬出。
- contraction 采用 FP16 Cube 路径，score 和 softmax 保留 FP32。

### Indexer

- 该实现将 650 个逻辑 key 补齐到 656，使 score task 保持 512B 对齐。
- 序列和 head 展平后通过一次 `bmm` 完成计算，以减少 broadcast matmul 开销。
- AscendC kernel 融合 ReLU、BF16 权重乘法、16-head reduction 和 causal
  mask，并使用双缓冲。

### Sinkhorn

- 该实现将 softmax 与 10 轮行列归一化融合为一次 kernel launch。
- 每个 `4 x 4` 矩阵在整个迭代期间驻留 UB。

## 文档索引

- [环境配置](ENVIRONMENT.md)
- [构建与运行脚本](RUNNING.md)
- [性能结果与测试精度](PERFORMANCE.md)

