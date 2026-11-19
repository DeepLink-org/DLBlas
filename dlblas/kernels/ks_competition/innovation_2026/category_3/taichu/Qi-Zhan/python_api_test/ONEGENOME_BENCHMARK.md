# OneGenomeRice 算子级测试

`benchmark_onegenome_attention.py` 在运行完整 vLLM CI 前验证比赛的两个算子。

固定模型参数来自 OneGenomeRice 在 TP=4 下的单卡配置：

- Query heads: 4
- KV heads: 2
- Head size: 64
- KV cache block size: 32
- Dtype: FP16

## 推荐流程

```bash
# 1. 日常修改后的快速正确性门禁
python python_api_test/benchmark_onegenome_attention.py \
  --mode correctness --suite quick \
  --output benchmark_results/quick.json

# 2. 候选版本的 T1-T4 长上下文正确性
python python_api_test/benchmark_onegenome_attention.py \
  --mode correctness --suite official \
  --output benchmark_results/official-correctness.json

# 3. 正确性通过后运行 T1-T4 微基准
python python_api_test/benchmark_onegenome_attention.py \
  --mode benchmark --suite official \
  --warmup 1 --iterations 3 --decode-iterations 20 \
  --output benchmark_results/official-performance.json
```

开发中只跑部分 case 时可增加 `--cases T1` 或 `--cases T1,T2`；候选版本必须省略该参数，完整运行 T1-T4。

完整 vLLM CI 只在上述三步都通过且微基准没有明显回退后运行。

## 测试内容

快速正确性覆盖 RiceBenchmark 的 924-token prefill、8 路并发 prefill、128 路并发 decode 和 mixed prefill/decode。每个 case 都先用 `reshape_and_cache` 构造 paged KV cache，再调用 `flash_attn_varlen_func`。

Q/K/V 从宽度为 512 的 Mixtral packed QKV 张量切分，保留真实的非连续 token stride `(512, ..., 1)`。测试禁止先把输入简化成连续张量，因为这种简化会漏掉模型接入中最关键的 stride 契约。

T1-T4 使用官方 `speed.sh` 的实际输入长度：16384、32768、65536、102400，输出长度均为 100。长上下文参考实现只计算若干 query 行，避免创建平方级 attention score 矩阵。

性能报告分别记录整段 KV cache 写入、decode 单 token cache append、prefill Attention 和单步 decode Attention 的 min、median、mean、max 及原始样本，方便后续提交横向比较。
