# KernelSwift T3 一键构建与评测说明

## 1. 适用范围

本文档用于复现 T3 Gated RMSNorm 的统一 A+B 作品：

- A 部分：`ntops` 中的 NineToothed 算子表达、布局、并行与融合实现；
- B 部分：`ninetoothed` 中的通用 SSA lowering、静态布局分析、Triton 物化与低开销运行时路径；
- 正式性能基线：未修改的 vLLM `RMSNormGated` 调用链；
- PyTorch eager 只生成独立正确性参考值，不参与 latency 或 speedup。

统一复现版本为本 PR 最终 `ntops` HEAD 与 `ninetoothed@6b79203`。T3 复用 `18c50cd`、`5b6abe8` 引入的通用 SSA、静态布局和低开销运行时能力，最终正确性、回归和性能记录均使用 `6b79203`。

## 2. 目录和环境要求

将两个仓库放在同一级目录：

```text
workspace/
├── ntops/
└── ninetoothed/
```

要求：

- Python 3.10 或更高版本；
- 厂商镜像自带并能使用的 PyTorch、Triton 和 vLLM；
- 一个可见的海光 BW 或天数智芯 GPU；
- `pytest`；
- `ntops` 使用本 PR 最终 HEAD，`ninetoothed` 固定为 `6b79203`。

脚本通过 `PYTHONPATH` 直接使用源码，不会用 `pip` 覆盖厂商 PyTorch、Triton 或 vLLM。

## 3. 一条命令完成构建与评测

NineToothed/Triton 在第一次调用时 JIT 构建 kernel，因此首次正确性测试同时完成构建验证。每次正式复现都应指定全新的输出目录，脚本会为正确性、编译器回归、正式性能和消融实验分别创建独立缓存。

```bash
cd /path/to/workspace/ntops
bash scripts/run_kernelswift_t3.sh \
  --ninetoothed-dir /path/to/workspace/ninetoothed \
  --output-dir /path/to/results/t3_run_001 \
  --mode all
```

以下命令均应在远程 Linux GPU 主机执行，`/path/to/remote/workspace` 仅表示该主机上的工作目录。

### 海光 BW/DTK 25.04 示例

```bash
source /opt/dtk-25.04.2/env.sh
cd /path/to/remote/workspace/ntops
bash scripts/run_kernelswift_t3.sh \
  --ninetoothed-dir /path/to/remote/workspace/ninetoothed \
  --output-dir /path/to/remote/workspace/results/t3_final_bw \
  --mode all
```

### 天数智芯天垓150示例

```bash
export KS_ROOT=/path/to/remote/workspace
cd "${KS_ROOT}/ntops"
bash scripts/run_kernelswift_t3.sh \
  --ninetoothed-dir "${KS_ROOT}/ninetoothed" \
  --output-dir "${KS_ROOT}/results/t3_final_biv150" \
  --mode all
```

## 4. 可单独执行的阶段

```bash
# 仅记录环境、版本和关键文件 SHA256
bash scripts/run_kernelswift_t3.sh --mode environment

# 32 个 T3 语义组合（当前测试矩阵）
bash scripts/run_kernelswift_t3.sh --mode correctness

# T3 所依赖的通用 SSA、布局、运行时和后端回归
bash scripts/run_kernelswift_t3.sh --mode regression

# vLLM 基线对 A+B 统一实现的正式 speedup
bash scripts/run_kernelswift_t3.sh --mode benchmark

# 仅隔离已检查 launch 与预验证低开销 launch 的运行时微消融
bash scripts/run_kernelswift_t3.sh --mode ablation
```

也可以直接执行核心命令：

```bash
export PYTHONPATH=/path/to/ntops/src:/path/to/ninetoothed/src:${PYTHONPATH:-}
export NINETOOTHED_CACHE_DIR=/path/to/a/new/cache
python -m pytest /path/to/ntops/tests/test_rms_norm_gated.py -q
python /path/to/ntops/benchmarks/benchmark_rms_norm_gated.py
```

## 5. 输出文件

`--mode all` 生成：

```text
<output-dir>/
├── environment.log
├── correctness.log
├── compiler_regression.log
├── benchmark.log
├── runtime_ablation.log
├── cache_correctness/
├── cache_regression/
├── cache_benchmark/
└── cache_ablation/
```

判定标准：

- `correctness.log` 末尾为全部通过；
- `compiler_regression.log` 无失败；
- `speedup = baseline_ms / submission_ms`；
- 每个正式 case 的 `speedup > 1`。

`runtime_ablation.log` 是 B 部分运行时开销的微消融。

## 6. 计时方法

- 每个 shape 先交替预热基线和提交实现 50 次；
- 每轮各执行 500 次；
- 共 7 轮，基线与提交实现交替先后顺序；
- 每条路径在计时区间前后执行 GPU 同步；
- 报告 7 轮 wall-clock 均值样本的中位数；
- 两条路径使用同一进程、同一输入、同一 dtype 和同一数学语义；
- 计时前分别对基线、提交实现和独立 PyTorch 参考值执行正确性检查。

BW 的 vLLM 0.9.2 通过 `Mixer2RMSNormGated.forward_cuda` 调用未修改算子；天垓150的 vLLM 0.11.2 直接调用其公开 `rms_norm_gated` op。BW 镜像若同时暴露 ROCm SMI 和 NVML 兼容层，基准脚本只在 vLLM 平台发现阶段屏蔽伪 CUDA 设备计数，确保选择 ROCm；vLLM kernel 和计时路径本身不做修改。
