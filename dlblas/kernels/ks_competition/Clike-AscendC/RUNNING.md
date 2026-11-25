# 构建与运行脚本

以下复现命令均以 DLBlas 仓库根目录为工作目录。

## 编译 AscendC 动态库

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
bash dlblas/kernels/ks_competition/ascend/clike_910b/build.sh
```

编译完成后生成以下动态库：

```text
dlblas/kernels/ks_competition/ascend/clike_910b/build/
libdlblas_ks_ascendc_ops.so
```

该构建脚本使用 CMake Release 模式、`-O3` 和
`--npu-arch=dav-2201`。

## 一键性能测试

```bash
bash dlblas/kernels/ks_competition/ascend/clike_910b/run_benchmarks.sh
```

性能脚本依次调用赛事提供的 `benchmarks/ks/auto_bench.py`：

- SparseAttention：预热 100 次，正式测量 1000 次。
- Indexer：预热 100 次，正式测量 1000 次。
- Sinkhorn：预热 100 次，正式测量 10000 次。

Sinkhorn 的单次执行时间较短，因此增加正式测量次数，以尽量降低不同测试
运行之间的波动，使中位数结果更稳定。

测试过程中，参考实现和优化实现分别连续测量，不交替执行。每个正式样本
执行一次 `forward`，随后同步 NPU，最终报告所有样本的中位数。
构建脚本和性能脚本使用相同的 Python 探测逻辑，只选择能够同时导入
`torch` 和 `torch_npu` 的解释器。

单独执行下列命令前，可通过公共 helper 解析同一个 Python：

```bash
source dlblas/kernels/ks_competition/ascend/clike_910b/python_env.sh
PYTHON_BIN="$(find_dlblas_python)"
```

## 单独测试 SparseAttention

```bash
"${PYTHON_BIN}" -u benchmarks/ks/auto_bench.py \
    --v0_file dlblas/kernels/ks_competition/torch/sparse_attention.py \
    --v1_file dlblas/kernels/ks_competition/ascend/clike_sparse_attention.py \
    --warmup 100 \
    --repeat 1000
```

## 单独测试 Indexer

```bash
"${PYTHON_BIN}" -u benchmarks/ks/auto_bench.py \
    --v0_file dlblas/kernels/ks_competition/torch/indexer.py \
    --v1_file dlblas/kernels/ks_competition/ascend/clike_indexer.py \
    --warmup 100 \
    --repeat 1000
```

## 单独测试 Sinkhorn

```bash
"${PYTHON_BIN}" -u benchmarks/ks/auto_bench.py \
    --v0_file dlblas/kernels/ks_competition/torch/sinkhorn.py \
    --v1_file dlblas/kernels/ks_competition/ascend/clike_sinkhorn.py \
    --warmup 100 \
    --repeat 10000
```

`auto_bench.py` 先加载由相同随机种子生成的模型和输入，将 reference 的
state dict 加载到 `ModelNew`，检查输出正确性，再分别计时。测试通过时的
输出格式如下：

```text
PASS accuracy; v0=<reference ms>, v1=<optimized ms>, speedup=<ratio>x
```

Task02 执行期间可能出现 NPU internal format 警告。该警告不影响输出正确性
或计时完成，测试状态以 `PASS accuracy` 为准。

`--warmup` 和 `--repeat` 是 `auto_bench.py` 提供的公开命令行参数，本地测试
可以按需要调整，不会修改赛事提供的 benchmark 文件。性能结果需要同时记录
这两个参数；正式成绩以赛事评测环境实际采用的参数为准。

