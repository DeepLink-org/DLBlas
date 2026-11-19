# KernelSwift T4 一键构建与评测说明

## 1. 适用范围

本说明用于 T4“MLA RoPE 与压缩 KV Cache 写入融合”的统一 A+B 构建和评测。

正式性能基线为未修改 vLLM 的：

```text
RoPE.forward_cuda + concat_and_cache_mla
```

脚本不修改 vLLM、测试、计时逻辑或基线代码，不报告 eager 对比。

统一复现版本为本 PR 最终 `ntops` HEAD 与 `ninetoothed@6b79203`。T4 使用的 MLA cache 后端支持由 `0dbe0e9` 引入，最终正确性、回归和性能记录均使用 `6b79203`。

## 2. 目录和依赖

将两个仓库放在同一父目录：

```text
workspace/
├── ninetoothed/
└── ntops/
```

远程环境需要：

- 可用的海光或天数智芯 GPU；
- Python、PyTorch、vLLM、Triton、pytest 和 ruff；
- 本 PR 最终 `ntops` HEAD 与 `ninetoothed@6b79203`；
- 天数 CoreX 环境中的 `/usr/local/corex/bin/clang++`，或通过环境变量指定等价编译器。

仓库根目录的 `.gitattributes` 将 shell 脚本声明为 `eol=lf`。在 Windows 上编辑或打包后，应确认
`benchmarks/run_kernelswift_t4.sh` 仍为 LF 行尾，以保证 Linux Bash 能直接执行。

NineToothed 为 JIT 编译。首次正确性调用会从 SSA 生成并编译目标 kernel，因此不需要绕过九齿编译链调用外部闭源算子。

## 3. 一键执行

在远程 GPU 主机执行：

```bash
cd /path/to/workspace/ntops
bash benchmarks/run_kernelswift_t4.sh /path/to/fresh/t4-result-dir
```

如果两个仓库不是同级目录：

```bash
cd /path/to/ntops
NINETOOTHED_ROOT=/path/to/ninetoothed \
    bash benchmarks/run_kernelswift_t4.sh /path/to/fresh/t4-result-dir
```

脚本拒绝复用已存在的结果目录，以避免不同赛题、A/B 构建或不同平台共享生成文件和缓存。

### 天数 CoreX

脚本检测到 `torch.corex` 后默认配置：

```bash
NINETOOTHED_BACKEND=cuda
CUDA_HOME=/usr/local/corex
NINETOOTHED_CUDA_COMPILER=/usr/local/corex/bin/clang++
NINETOOTHED_CUDA_LANGUAGE=ivcore
```

如正式镜像路径不同，可在命令前覆盖这些变量。不要退回系统 NVIDIA `nvcc`；旧版天数镜像中的兼容 `nvcc 10.2` 可能返回成功但不生成 CoreX 可加载的 `.so`。

### 海光 BW

海光默认使用 NineToothed Triton 后端：

```bash
NINETOOTHED_BACKEND=triton \
    bash benchmarks/run_kernelswift_t4.sh /path/to/fresh/bw-t4-result-dir
```

## 4. 脚本执行内容

脚本依次完成：

1. GPU 分配与同步健康检查；
2. 记录 Python、PyTorch、Triton、vLLM、设备和平台信息；
3. 对 T4 直接代码运行 ruff；
4. 运行 T4 26 项正确性矩阵；
5. 运行 120 项 SSA、materializer、布局和运行时定向回归；
6. 对 32 和 128 query heads 各运行 3 个正式 case；
7. 保存统一代码 SHA-256 和正式 speedup 汇总。

关键输出：

```text
<result-dir>/
├── FORMAL_SPEEDUP_SUMMARY
├── SHA256SUMS
├── ninetoothed-cache/
└── logs/
    ├── environment.log
    ├── correctness.log
    ├── compiler_regression.log
    ├── ruff_ntops.log
    ├── ruff_ninetoothed.log
    ├── benchmark_heads32.log
    └── benchmark_heads128.log
```

## 5. 正确性判定

正式 benchmark 在计时前分别校验 vLLM 路径和 NineToothed 路径。测试矩阵额外覆盖：

- BF16 和 FP16；
- `positions`/`slot_mapping` 的 INT32 和 INT64；
- 1、16、128 tokens；
- 32、128 query heads；
- 16、64 block size；
- `slot_mapping=-1` 的 padding 语义；
- 重复调用时动态更新 position/slot，防止运行时索引被错误固化。

只有正确性、编译器回归和全部正式 case 均通过，结果才可写入报告。

## 6. 性能计算

每个 case 使用：

```text
speedup = vLLM_latency / NineToothed_latency
```

计时过程为交替预热 100 次、每轮 1000 次、共 9 轮，分别取两条路径 wall-clock 样本中位数。脚本运行的是默认统一分派路径，不强制实验性的 scalar-only 或 pair-only 实现。

## 7. 本轮天数复现结果

验证目录：

```text
<results-root>/t4/tiangai150
```

环境：Iluvatar BI-V150、IX-ML 4.4.0、PyTorch 2.7.1、Triton 3.1.0、vLLM 0.11.2、CoreX clang 18.1.8。

结果：

```text
T4 correctness: 26 passed
backend regression: 122 passed
ruff: passed
```

6 个正式 case 的最新完整复测均 `correctness=pass` 且 `formal_speedup > 1`。原始日志为：

```text
<results-root>/t4/tiangai150/FORMAL_SPEEDUP_SUMMARY
```

## 8. 本轮海光 BW DTK25.04 复测结果

验证目录：

```text
<results-root>/t4/bw
```

环境：BW/gfx936、Python 3.10.12、PyTorch 2.5.1、HIP 6.3.25405、Triton 3.1、vLLM 0.9.2。
本轮使用本 PR 最终统一验证锚点 `ntops` HEAD 与 `ninetoothed@6b79203`，baseline 仍为未修改的
vLLM `RoPE.forward_cuda + concat_and_cache_mla`。

```text
T4 correctness: 26 passed in 46.08s
compiler regression: 122 passed in 193.52s
ruff: passed
```

六个正式 case 均为 `correctness=pass`：

| Case | vLLM ms | NineToothed ms | formal speedup |
|---|---:|---:|---:|
| seq1 / 32 heads | 0.023896 | 0.010660 | 2.2417× |
| seq16 / 32 heads | 0.023658 | 0.011924 | 1.9842× |
| seq128 / 32 heads | 0.047728 | 0.010604 | 4.5007× |
| seq1 / 128 heads | 0.022710 | 0.015884 | 1.4298× |
| seq16 / 128 heads | 0.022176 | 0.015493 | 1.4313× |
| seq128 / 128 heads | 0.035463 | 0.022038 | 1.6092× |

BW 六个 case 的 speedup 算术平均为 `2.199483×`。

## 9. 双平台结果汇总

天数旧版天垓150和海光 BW DTK25.04 均完成同一 T4 测试矩阵。天数六个 case 的平均 speedup
为 `1.5766×`，BW 六个 case 的平均 speedup 为 `2.199483×`。
