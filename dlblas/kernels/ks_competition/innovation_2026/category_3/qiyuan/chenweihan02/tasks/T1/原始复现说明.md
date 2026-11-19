# KernelSwift T1 一键构建与评测说明

## 1. 适用范围

本文档用于复现 T1 MXFP4 W4A16 分组专家矩阵乘的统一 A+B 作品：

- A 部分：`ntops` 中的 NineToothed MXFP4 解码、分组 GEMM、布局与并行实现；
- B 部分：`ninetoothed` 中的通用 SSA `bitcast`、`interleave`、`transpose` lowering，以及可安全绑定的 Triton 低开销 launch；
- 接口语义：对齐 PyTorch `scaled_grouped_mm` 的 grouped-M offsets，并采用 vLLM/OCP MXFP4 E2M1+E8M0 编码；
- 预评测 baseline：当前厂商环境无法执行原生 BF16×MXFP4 PyTorch/vLLM kernel，因此使用文档中明确披露的 vLLM 语义 fallback，并另外记录逐次解码的 PyTorch raw reference。

统一复现版本为本 PR 最终 `ntops` HEAD 与 `ninetoothed@6b79203`。其中 T1 使用的通用 SSA 基础来自 `18c50cd`，T1 编译与运行时能力来自 `5b6abe8`。

## 2. 目录与环境要求

将两个仓库放在同一级目录：

```text
workspace/
├── ntops/
└── ninetoothed/
```

要求：

- Python 3.10 或更高版本；
- 厂商镜像提供且可用的 PyTorch 与 Triton；
- 一个可见的海光 BW 或天数智芯 GPU；
- `pytest` 与 `ruff`；
- `ntops` 使用本 PR 最终 HEAD，`ninetoothed` 固定为 `6b79203`。

脚本通过 `PYTHONPATH` 直接导入源码，不覆盖厂商 PyTorch、Triton 或 vLLM。NineToothed/Triton 在首次调用时 JIT 构建 kernel，因此首次正确性测试同时完成 clean build 验证。

## 3. 一条命令完成验证

```bash
cd /path/to/workspace/ntops
bash scripts/run_kernelswift_t1.sh \
  --ninetoothed-dir /path/to/workspace/ninetoothed \
  --output-dir /path/to/results/t1_run_001 \
  --mode all
```

每次复现必须使用新的 `--output-dir`。脚本为 correctness、compiler regression 和 benchmark 分别创建独立的 NineToothed、Triton 与 XDG cache，避免跨阶段或跨平台共享生成产物。

以下命令均应在远程 Linux GPU 主机执行，`/path/to/remote/workspace` 仅表示该主机上的工作目录。

### 海光 BW/DTK 25.04 示例

```bash
bash -l
cd /path/to/remote/workspace/ntops
bash scripts/run_kernelswift_t1.sh \
  --ninetoothed-dir /path/to/remote/workspace/ninetoothed \
  --output-dir /path/to/remote/workspace/results/t1_bw_run_001 \
  --mode all
```

### 天数智芯天垓150示例

```bash
bash -l
cd /path/to/remote/workspace/ntops
bash scripts/run_kernelswift_t1.sh \
  --ninetoothed-dir /path/to/remote/workspace/ninetoothed \
  --output-dir /path/to/remote/workspace/results/t1_tiangai_run_001 \
  --mode all
```

厂商 GPU 库由登录 shell 初始化。若直接通过 SSH 执行非交互命令，应使用 `bash -lic`，否则可能出现 `libgalaxyhip.so` 或 `libcudart.so` 不在动态库路径中的假性失败。

## 4. 可单独执行的阶段

```bash
# 环境、仓库 HEAD、dirty 状态和关键文件 SHA256
bash scripts/run_kernelswift_t1.sh --mode environment

# Ruff 与 ninetoothed 贡献规范检查
bash scripts/run_kernelswift_t1.sh --mode lint

# 15 个算子正确性/缓存/stream/tile 选择测试
bash scripts/run_kernelswift_t1.sh --mode correctness

# 86 个相关 SSA、Triton runtime 与 emitter 回归
bash scripts/run_kernelswift_t1.sh --mode regression

# 5 个 BF16 代表 case，生成 benchmark.log
bash scripts/run_kernelswift_t1.sh --mode benchmark

# 扩展矩阵示例：FP16、大 N/K、更多 expert 和不均衡 routing
python benchmarks/benchmark_mxfp4_grouped_gemm_reference.py \
  --experts 4 --n 256 --k 1024 --counts 1,3,5,7 --dtype bfloat16 \
  --warmup 50 --iterations 500 --rounds 9
```

BW DTK25.04 的 FP16 性能测试使用相同的 5 个代表 case、`50/500/9` 协议和独立 cache。以下命令会生成远程实验日志；最终提交只在报告中汇报设备、软件版本、测试方法和汇总结果：

```bash
FP16_OUTPUT=/path/to/fresh/results/t1_bw_fp16
mkdir -p "$FP16_OUTPUT/cache_ninetoothed" "$FP16_OUTPUT/cache_triton" "$FP16_OUTPUT/cache_xdg"
export NINETOOTHED_CACHE_DIR="$FP16_OUTPUT/cache_ninetoothed"
export TRITON_CACHE_DIR="$FP16_OUTPUT/cache_triton"
export XDG_CACHE_HOME="$FP16_OUTPUT/cache_xdg"

{
  python benchmarks/benchmark_mxfp4_grouped_gemm_reference.py \
    --dtype float16 --experts 1 --n 16 --k 32 --counts 1 \
    --warmup 50 --iterations 500 --rounds 9
  python benchmarks/benchmark_mxfp4_grouped_gemm_reference.py \
    --dtype float16 --experts 2 --n 32 --k 64 --counts 1,3 \
    --warmup 50 --iterations 500 --rounds 9
  python benchmarks/benchmark_mxfp4_grouped_gemm_reference.py \
    --dtype float16 --experts 2 --n 64 --k 128 --counts 5,2 \
    --warmup 50 --iterations 500 --rounds 9
  python benchmarks/benchmark_mxfp4_grouped_gemm_reference.py \
    --dtype float16 --experts 4 --n 128 --k 256 --counts 1,0,17,5 \
    --warmup 50 --iterations 500 --rounds 9
  python benchmarks/benchmark_mxfp4_grouped_gemm_reference.py \
    --dtype float16 --experts 4 --n 256 --k 256 --counts 1,3,5,7 \
    --warmup 50 --iterations 500 --rounds 9
} 2>&1 | tee "$FP16_OUTPUT/benchmark.log"
```

也可以直接执行核心正确性测试：

```bash
export PYTHONPATH=/path/to/ntops/src:/path/to/ninetoothed/src:${PYTHONPATH:-}
export NINETOOTHED_CACHE_DIR=/path/to/new/cache/ninetoothed
export TRITON_CACHE_DIR=/path/to/new/cache/triton
cd /path/to/ntops
python -m pytest tests/test_mxfp4_grouped_gemm.py -q
```

## 5. 输出文件

`--mode all` 生成：

```text
<output-dir>/
├── environment.log
├── lint.log
├── correctness.log
├── compiler_regression.log
├── benchmark.log
├── cache_correctness/
├── cache_regression/
└── cache_benchmark/
```

判定标准：

- `correctness.log` 为 `15 passed`（当前 T1 版本）；
- `compiler_regression.log` 无失败；
- `speedup = baseline_ms / submission_ms`；
- 每个预评测 case 的 `speedup > 1`。

## 6. baseline 与计时口径

脚本先探测原生 `torch._scaled_grouped_mm`：

- 若当前 PyTorch 和硬件接受 BF16/FP16 activation、packed MXFP4 weight、E8M0 scale 与 grouped offsets，则单独报告原生路径可用；
- 当前已测 BW PyTorch 2.5.1没有该入口；天垓150 PyTorch 2.7.1存在入口，但拒绝 BF16 activation并要求 FP8，因此两者均不能作为当前可执行 T1 原生 baseline。

当前 benchmark 的主要 baseline 为 `vllm_semantic_fallback_predecoded`：严格按 vLLM/OCP MXFP4 编码预解码 weight 一次，然后逐 expert 执行 BF16/FP16 grouped GEMM。它是公开、可执行并明确披露的语义 fallback，不伪装成原生 PyTorch kernel。脚本同时记录每次重新解码 weight 的 PyTorch raw reference，但后者不作为主 speedup。

扩展矩阵只作为本地实验记录，不纳入提交文件。wrapper 对长归约、少量 expert token 且宽 N 的输入，根据 `K/N` 和实际最大 token 数选择较窄的 N tile；这是张量属性驱动的通用启发式，不是公开 case 分支。扩展结果中，长 K case 使用正式的 `50/500/9` 计时，其余扩展 case 使用明确记录的 `30/200/7` 预评测协议。

每个 case：

- 预热 50 次；
- 每轮 500 次；
- 共 9 轮；
- 每轮使用 GPU 同步包围 wall-clock 计时区间；
- 报告 9 个样本的中位数；
- baseline 与 A+B 使用相同输入、dtype、MXFP4 编码、expert offsets 和输出语义；
- 计时前执行独立 FP32 解码参考的正确性检查。
