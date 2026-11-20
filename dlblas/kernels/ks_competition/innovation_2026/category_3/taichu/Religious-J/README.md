# 太初赛道 · OneGenomeRice

提交人 `Religious-J`。算子为 SDAA FP16 `flash_attn_varlen_func` 和 `reshape_and_cache`。评分配置为 TP=4，每张卡有 4 个 query head、2 个 KV head，head dim 为 64，KV block 有 32 个 token。

- `teco/` 包含算子实现、设备入口和 dispatch。
- `api/` 包含 PyTorch 绑定。
- `python_api_test/`、`test/zoo/` 和 `tools/scoring/` 包含测试代码、评分用例生成器和运行脚本。

完整 K block 由 `reshape_and_cache` 转置，Flash Attention 直接读取。尾部未写满的 K block 保持 token-major 布局，由 Flash Attention 在片上转置。Decode 切分完整 KV 上下文并归并，各 prefill 路径均保留完整 causal 上下文。

## 在太初环境复现

在 `tecorigin-waic/teco-ops-baseline` 的独立检出中，从 `c30e4409` 的父提交开始，按原相对路径覆盖源码。旧的 `flash_attention.scpp` 已拆分为 prefill 和 decode，需在独立检出中移除旧文件以免 CMake 一并编译。将 `submission` 指向本目录。

```bash
submission=/path/to/DLBlas/dlblas/kernels/ks_competition/innovation_2026/category_3/taichu/Religious-J
git switch --detach cf7f8a97bc8a13de5cdc2bd3844e4ede040651a1
git rm teco/ual/kernel/flash_attention/flash_attention.scpp
for dir in api teco python_api_test test tools; do
    cp -R "$submission/$dir/." "$dir/"
done

source /opt/tecoai/setvars.sh
WITH_TORCH=ON WITH_INFERENCE_PLUGIN=OFF python3 setup.py build_ext --inplace
python3 tools/scoring/generate_scoring_cases.py
bash tools/scoring/run_scoring_baseline.sh
```

评分脚本可用 `reshape`、`decode`、`prefill` 或 `tail` 运行单组用例。默认检查 reshape 和 decode 的正确性，并测量 prefill 和 tail 的性能。日志输出到 `test/scoring_baseline/`。
