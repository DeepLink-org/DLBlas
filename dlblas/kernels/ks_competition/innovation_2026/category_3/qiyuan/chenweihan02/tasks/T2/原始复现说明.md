# KernelSwift T2 一键构建与评测说明

## 1. 适用范围

本文档用于复现 T2 Block-scaled FP8 矩阵乘的统一 A+B 作品：

- A 部分：`ntops` 中的 NineToothed 算子表达、数据布局、分块策略、缩放融合与 PyTorch 兼容入口；
- B 部分：`ninetoothed` 中的通用 SSA/Triton lowering、厂商工具链能力驱动的 FP8 dot 合法化，以及预验证低开销 launch；
- 正确性参考：FP32 分块反量化与矩阵乘的 PyTorch eager 实现；
- 性能基线：优先支持可执行的 PyTorch `_scaled_mm`，否则使用明确标注的 vLLM 路径。

PyTorch eager 只生成正确性参考值，不参与 latency 或 speedup。

统一复现版本为本 PR 最终 `ntops` HEAD 与 `ninetoothed@6b79203`。T2 使用的可移植 FP8 dot 合法化能力由 `4ea752e` 引入，最终正确性、回归和性能记录均使用 `6b79203`。

## 2. 基线选择与边界

一键脚本支持四种基线：

| 参数 | 含义 |
|---|---|
| `pytorch` | 调用当前镜像的 `torch._scaled_mm`，不修改其 timed kernel |
| `vllm_triton` | 调用当前镜像的 vLLM block-scaled FP8 Triton 算子 |
| `vllm_vendor` | 调用当前镜像由 vLLM 暴露的厂商算子 |
| `vllm_portable` | vLLM `fp8_utils.py` 的 Apache-2.0 兼容移植，默认值 |

当前两台目标服务器不能直接执行语义一致的前三条路径：

- 天数智芯 PyTorch `_scaled_mm` 报 `Invalid scaling configuration`；
- 海光 BW 的 PyTorch `_scaled_mm` 要求 CUDA CC 9.0+ 或 ROCm MI300+，而 BW 为 `gfx936`；
- 未修改的 vLLM Triton 路径在两套厂商 Triton 上分别出现数值错误或 FP8 dot 编译失败。

因此当前性能证据使用 `vllm_portable`。它保持 vLLM 的 kernel 结构、数据布局和默认 launch 配置，只加入厂商 Triton 缺少原生 FP8 `tl.dot` 时必需的 operand cast：CoreX 使用 BF16，DTK 使用 FP16。该文件只用于 benchmark，不被 `ntops` 算子导入。

这组结果应准确表述为“相对 vLLM-derived portable compatibility baseline 的 A+B speedup”。

## 3. 目录和环境要求

将两个仓库放在同一级目录：

```text
workspace/
├── ntops/
└── ninetoothed/
```

要求：

- Python 3.10 或更高版本；
- 厂商镜像自带且可用的 PyTorch 与 Triton；
- 使用 vLLM 基线时，镜像中应安装 vLLM；
- 一个可见的海光 BW 或天数智芯 GPU；
- `pytest`、`git` 和 `sha256sum`；
- 两个仓库均处于报告记录的可追溯版本。

脚本通过 `PYTHONPATH` 直接使用两个仓库的源码，不会用 `pip` 覆盖厂商 PyTorch、Triton 或 vLLM。

## 4. 一条命令完成构建与评测

NineToothed/Triton 在首次调用时 JIT 构建 kernel，因此首次正确性测试同时完成编译与执行验证。每次正式复现应使用新的输出目录；脚本为正确性、编译器回归和性能评测分别创建 NineToothed/Triton 缓存。

```bash
cd /path/to/workspace/ntops
bash scripts/run_kernelswift_t2.sh \
  --ninetoothed-dir /path/to/workspace/ninetoothed \
  --output-dir /path/to/results/t2_run_001 \
  --baseline vllm_portable \
  --mode all
```

以下命令均应在远程 Linux GPU 主机执行，`/path/to/remote/workspace` 仅表示该主机上的工作目录。

### 海光 BW/DTK 25.04 示例

```bash
cd /path/to/remote/workspace/validation/ntops
bash scripts/run_kernelswift_t2.sh \
  --ninetoothed-dir /path/to/remote/workspace/validation/ninetoothed \
  --output-dir /path/to/remote/workspace/results/t2_bw_dtk2504 \
  --baseline vllm_portable \
  --mode all
```

### 天数智芯天垓150示例

```bash
cd /path/to/remote/workspace/validation/ntops
bash scripts/run_kernelswift_t2.sh \
  --ninetoothed-dir /path/to/remote/workspace/validation/ninetoothed \
  --output-dir /path/to/remote/workspace/results/t2_tiangai150 \
  --baseline vllm_portable \
  --mode all
```

## 5. 可单独执行的阶段

```bash
# 仅记录环境、仓库 HEAD/dirty 状态和关键文件 SHA256
bash scripts/run_kernelswift_t2.sh --mode environment

# T2 算子接口、布局、边界、dtype、bias 与数值测试
bash scripts/run_kernelswift_t2.sh --mode correctness

# T2 依赖的 FP8 dot、SSA、静态布局和运行时相关编译器回归
bash scripts/run_kernelswift_t2.sh --mode regression

# 六个 case：M=1/16/128，N=K=4096，E4M3FN/E5M2
bash scripts/run_kernelswift_t2.sh \
  --mode benchmark \
  --baseline vllm_portable
```

如目标平台已经支持 PyTorch block scaling，可执行：

```bash
bash scripts/run_kernelswift_t2.sh \
  --mode benchmark \
  --baseline pytorch
```

也可以直接探测一个基线路径。以下命令先验证正确性，只有通过后才计时：

```bash
export PYTHONPATH=/path/to/ntops/src:/path/to/ninetoothed/src:${PYTHONPATH:-}
python benchmarks/probe_official_scaled_mm.py \
  --mode pytorch \
  --m 1 --n 4096 --k 4096 \
  --fp8-dtype e4m3fn
```

## 6. 输出文件

`--mode all` 生成：

```text
<output-dir>/
├── environment.log
├── correctness.log
├── compiler_regression.log
├── benchmark_summary.log
├── benchmark_raw/
├── cache_correctness/
├── cache_regression/
├── cache_benchmark/
├── triton_cache_correctness/
├── triton_cache_regression/
└── triton_cache_benchmark/
```

判定标准：

- `correctness.log` 中 T2 测试全部通过；
- `compiler_regression.log` 无失败；
- `benchmark_raw/` 中 submission 与 baseline 的每次调用均先通过正确性；
- `speedup = baseline_median_ms / submission_median_ms`；
- 每个性能 case 的 `speedup > 1`。

若选择的 PyTorch/vLLM 路径在目标硬件不可执行，脚本会保留错误日志并以非零状态退出，不会自动拿 eager latency 冒充 speedup。

## 7. 计时方法

- shape 为 `(M,N,K)=(1/16/128,4096,4096)`；
- 覆盖 `float8_e4m3fn` 和 `float8_e5m2`；
- scale layout 为 `A: 1×128`、`B: 128×128`，输出为 BF16；
- 每个 candidate 每轮先完成首次 JIT 与正确性检查，再预热 20 次、计时 100 次；
- 共 3 轮，奇数轮 submission 先测，偶数轮 baseline 先测；
- 每条路径在计时区间前后执行 GPU 同步；
- 报告三轮 latency 的中位数；
- submission 与 baseline 使用同一随机种子、shape、dtype、scale layout 和数学语义；
- FP32 eager reference 不计时。

可通过 `--warmup`、`--iterations` 和 `--rounds` 增大统计强度，但修改参数后应在报告中同时更新。

## 8. 当前已保存的结果

结果汇总见 `docs/KERNELSWIFT_T2_REPORT.md` 第 10 节，原始服务器日志保存在 `<results-root>/t2/logs/`。

关键日志：

```text
bw_t2_final_matrix_prevalidated_v1.log
tiangai_t2_final_matrix_prevalidated_v1.log
bw_t2_final_tests_v1.log
tiangai_launch_plan_tests_v1.log
bw_backend_fp8_fallback_final_v1.log
tiangai_backend_fp8_fallback_final_v1.log
```

本 PR 最终 `ntops` HEAD 与 `ninetoothed@6b79203` 的 12 个跨平台测试点全部通过计时前、baseline 和计时后正确性，且 speedup 全部大于 1。六个 case 在海光平台的总体均值为 **3.316458×**，天数平台总体均值为 **1.508378×**；两平台各 50% 合并为 **2.412418×**。

一键脚本已在天垓150的最终本地源码 validation 副本上做过独立 smoke 验证：shell 语法检查通过，T2 为 `30 passed in 23.50s`，脚本所列扩展编译器回归为 `117 passed in 9.59s`，benchmark 汇总输出成功生成。benchmark smoke 使用 1 轮、5 次预热、20 次计时，只验证编排与聚合逻辑，不替代上表三轮正式结果。
