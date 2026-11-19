# 太初赛道 · OneGenomeRice · Qi-Zhan

- `reshape_and_cache` 将 FP16 K/V 写入分页 KV cache，支持 packed QKV 切分后的步长及单 token append。
- `flash_attn_varlen_func` 在 SDAA 上执行分页注意力，包含 head size 64 的专用 prefill 和 decode kernel，以及与 KV cache append 的融合路径。

## 在 teco-ops 中构建

交付版本为参赛仓库的 `80a9a26954c001a326d75624d262f373f8d4cc93`。请在具备 SDAA SDK 和 PyTorch SDAA 扩展的环境中，从赛事 teco-ops 模板提交 `39b16e895ef3accdf8f5f0f3dd72fa924356fa0d` 开始。在模板根目录执行以下命令，将 `submission` 设置为本目录的绝对路径。

```bash
submission=/path/to/Qi-Zhan
git apply "$submission/teco-ops-integration.patch"
rsync -a "$submission/teco/" teco/
rsync -a "$submission/python_api_test/" python_api_test/
rsync -a "$submission/test/" test/
WITH_TORCH=ON WITH_INFERENCE_PLUGIN=OFF python setup.py build_ext --inplace
```

共享的 C API 头文件和 PyTorch 扩展通过补丁集成。算子实现和测例保留参赛仓库中的相对路径。

## 验证

在同一环境中运行 OneGenomeRice 正确性测试。

```bash
python python_api_test/benchmark_onegenome_attention.py --mode correctness --suite quick
```

长上下文正确性和微基准命令见 [OneGenomeRice 算子级测试](python_api_test/ONEGENOME_BENCHMARK.md)。
