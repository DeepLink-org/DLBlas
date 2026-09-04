# 性能结果与测试精度

## 测试口径

以下性能数据采用统一测试口径记录。

- 硬件：Ascend 910B1 / Atlas A2。
- 软件：CANN 9.0.0、PyTorch 2.10.0、torch-npu 2.10.0。
- 测试工具：赛事提供的 `benchmarks/ks/auto_bench.py`。
- 统计方式：每个正式样本执行一次 `forward` 后同步 NPU，报告中位数。
- 预热次数：每个实现 100 次。
- 正式次数：SparseAttention 和 Indexer 为 1000 次，Sinkhorn 为 10000 次。
- Sinkhorn 单次执行时间较短，增加测量次数可尽量降低不同测试运行之间的
  波动，使中位数结果更稳定。
- 浮点正确性阈值：`atol=1e-2`、`rtol=1e-2`、`equal_nan=True`。
- 整数输出：使用 `torch.equal` 逐元素精确比较。

## 性能结果

| 算子 | PyTorch reference | AscendC 实现 | 加速比 |
|---|---:|---:|---:|
| SparseAttention | 12.805535 ms | 7.297500 ms | 1.755x |
| Indexer | 9.259980 ms | 5.142080 ms | 1.801x |
| Sinkhorn | 1.623945 ms | 0.329970 ms | 4.921x |

记录结果显示，三个算子均通过赛事 `auto_bench.py` 的 accuracy 检查。
表中数据为按当前次数配置完成的手动复测结果。

## 精度说明

### SparseAttention

| 阶段 | 精度 |
|---|---|
| `q`、`kv` 输入 | BF16 |
| `topk_idxs` 输入 | INT32 |
| `attn_sink` | FP32 |
| Reference 点积、softmax、加权求和 | FP32 |
| 优化实现 Cube contraction | FP16 |
| 优化实现 score 与 softmax | FP32 |
| 输出 | BF16 |

赛事文档明确标注了输入输出类型。该实现使用混合精度，最终 BF16 输出通过
`atol=rtol=1e-2` 检查。

### Indexer

| 阶段 | 精度 |
|---|---|
| `x`、`qr`、KV cache | BF16 |
| Linear 权重及主路径 | BF16 |
| RoPE 临时计算 | FP32 / complex64 |
| 优化实现 QK Cube `bmm` | BF16 |
| 融合 reduction 累加 | FP32，并保留 BF16 乘法舍入 |
| TopK 输出 | INT64 |

虽然 `ModelArgs.dtype` 的默认字符串是 `"fp8"`，该组测试实际使用
`default_dtype=torch.bfloat16`，输入、KV cache 和 Linear 权重均为 BF16，
没有执行 FP8 路径。TopK 输出是整数，`auto_bench.py` 要求逐元素完全一致。

### Sinkhorn

| 阶段 | 精度 |
|---|---|
| 输入 | FP32 |
| softmax、行归一化、列归一化 | FP32 |
| 输出 | FP32 |

赛事文档在 Task03 的 `forward` 注释中明确要求 FP32 输入和输出。该实现将
全部迭代融合到一个 AscendC kernel，但不降低计算精度。

## 文档要求的边界

根据赛事要求，优化实现需要保持与 reference 相同的 `Model` 初始化参数和
`forward` 参数，并通过 reference 正确性校验。赛事文档没有要求自定义
kernel 的每个内部阶段必须与 reference 使用完全相同的中间精度，因此允许
混合精度优化；最终输出仍须满足上述浮点容差或整数精确比较，并且实际执行
路径必须调用自定义算子。

