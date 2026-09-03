# Ascend MoonEP Planning 独立实现

本目录提供一个不依赖 MoonEP 源码树的 Planning Torch/NPU 实现、测试工具和运行脚本。
当前迁移的实现包括 `planning_torch_ascend.py` 和
`planning_triton_ascend_reference.py`；测试通过两个真实 HCCL rank 执行每个 rank
的完整 Planning，并与独立的 Torch reference oracle 比较。

## 文件说明

| 文件 | 作用 |
|---|---|
| `planning_torch_ascend.py` | Torch/NPU `launch_planning` 实现。 |
| `planning_triton_ascend_reference.py` | Triton reference-semantic 实现。 |
| `planning_ascend_common.py` | `MoonEPCommPlan` 和公共 Planning 代码。 |
| `planning_test_utils_ascend.py` | case、fixture、oracle 和比较工具。 |
| `test_planning_triton_ascend_standalone.py` | 低依赖多卡功能测试。 |
| `run_reference_optimized_round.py` | correctness 和 E2E 性能测试。 |
| `run_planning_ascend_tests.sh` | 独立功能/性能统一入口。 |

## 运行依赖

代码目录不依赖 `MoonEP`、`MoonEP/tests` 或 `planning_triton_src_ascend`。运行时需要已有环境提供：

```text
Python 3
PyTorch
torch_npu
Triton（使用 `triton_reference_semantic` 时）
pytest
CANN / HCCL
```

不安装额外 Python 依赖，也不修改容器或系统环境。

## 使用方式

从 DLBlas 项目根目录执行，或确保项目根目录在 `PYTHONPATH` 中：

```bash
cd /path/to/DLBlas
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export MOONEP_NPU_IDS=<npu_id_a>,<npu_id_b>
export ASCEND_RT_VISIBLE_DEVICES=<npu_id_a>,<npu_id_b>
```

测试代码和统一脚本不会主动设置 `TASK_QUEUE_ENABLE` 或
`ASCEND_LAUNCH_BLOCKING`。需要同步调试时，再由调用方显式设置。

### 功能测试

默认执行 standalone case、`zero/current/src` 三种 dedup mode，并通过
`IMPLEMENTATION_MODULES` 只加载本目录的 Torch 和 Triton reference-semantic implementation：

```bash
./dlblas/kernels/ascend/moonep/run_planning_ascend_tests.sh \
  functional --npu-ids <npu_id_a>,<npu_id_b>
```

只测试一个 case：

```bash
./dlblas/kernels/ascend/moonep/run_planning_ascend_tests.sh \
  functional --npu-ids <npu_id_a>,<npu_id_b> --cases balanced_epn16
```

### `STANDALONE_CASES` 覆盖范围

standalone 测试使用低依赖的精选矩阵，不构造 MoonEP 的 VMM/meta ABI。当前
`STANDALONE_CASES` 包含：

| case | 覆盖重点 |
|---|---|
| `balanced_epn16` | 常规 balanced routing、较大 token/top-k 规模 |
| `tiny_s1_k1` | 最小输入、边界 shape |
| `typical_bias` | biased routing 和非均匀 expert 分布 |
| `all_remote_with_prefetch_slots` | 全 remote routing、prefetch slots 和 `B` 配置 |
| `duplicate_topk` | 同一 token 的重复 top-k expert |
| `single_expert_padding_tail` | single-expert routing 和 padding tail |

每个 case 都覆盖 `zero`、`current`、`src` 三种 dedup mode；测试会在每个真实
rank 上比较 Torch implementation 与独立 oracle 的 public outputs，并检查
Planning/dedup invariants。完整 `PLANNING_CASES` official 矩阵不属于这个低依赖
默认集合，需要时应单独运行 official 测试。

直接运行测试文件：

```bash
PYTHONPATH=/path/to/DLBlas \
torchrun --nnodes=1 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29631 \
  -m pytest -s -q \
  dlblas/kernels/ascend/moonep/test_planning_triton_ascend_standalone.py
```

### 性能测试

性能测试先做一次 correctness 对比，再执行 warmup 和正式测量。默认 case
`balanced_epn16` 使用 `num_sms=24`；可以用 `--num-sms` 覆盖：

```bash
./dlblas/kernels/ascend/moonep/run_planning_ascend_tests.sh \
  performance --npu-ids <npu_id_a>,<npu_id_b> \
  --dedup-mode src --warmup 10 --repeat 30
```

或直接运行：

```bash
PYTHONPATH=/path/to/DLBlas \
TRITON_CACHE_DIR=/tmp/moonep-triton-cache \
MOONEP_NPU_IDS=<npu_id_a>,<npu_id_b> \
ASCEND_RT_VISIBLE_DEVICES=<npu_id_a>,<npu_id_b> \
torchrun --nnodes=1 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29632 \
  /path/to/DLBlas/dlblas/kernels/ascend/moonep/run_reference_optimized_round.py \
  --module dlblas.kernels.ascend.moonep.planning_torch_ascend \
  --case balanced_epn16 --dedup-mode src \
  --warmup 10 --repeat 30
```

输出格式为：

```text
ROUND_RESULT rank=0 case=balanced_epn16 dedup_mode=src \
correctness=pass launches=1 median_ms=<value> min_ms=<value> peak_bytes=<value>
```

`median_ms` 是单 rank 的 E2E median；多卡比较时取 rank 中较慢者。首次运行可能
包含 PyTorch/CANN 初始化开销，应以 warmup 后的正式样本为准。

## 功能验证记录

验证环境为 Ascend 910B、两个真实 NPU、两个 distributed rank，以及已有的
PyTorch/torch_npu/Triton/CANN/HCCL；未设置 `TASK_QUEUE_ENABLE` 和
`ASCEND_LAUNCH_BLOCKING`。受当前机器可用资源限制，本次验证暂只覆盖双卡；脚本
仍按 `--npu-ids` 自动推导 rank 数，三卡及以上组合未纳入本次结果：

```text
Torch、Triton 和共享 oracle 的完整 standalone 矩阵：54 passed in 161.88s
```

## 性能记录

以下结果在 Ascend 910B 双卡（当前机器资源限制）上测得；本次没有测试三卡及以上
配置。软件环境为 Python 3.12.13、PyTorch 2.7.1、torch_npu 2.7.1.post8、
Triton 3.2.0、CANN 9.1.0-beta.1、npu-smi 25.2.3 和 HCCL。测试使用两个 distributed rank，
`balanced_epn16`、`num_sms=24`、warmup 10、repeat 30；未设置
`TASK_QUEUE_ENABLE` 或 `ASCEND_LAUNCH_BLOCKING`，每轮 correctness 均为 pass。
数值为两个 rank 中较慢 rank 的 median E2E，计时包含 output allocation、HCCL
input gather、Planning、同步和 rank-local copy，不包含 oracle、assertion 和
首次 Triton 编译。每个正式样本在计时开始前执行一次 rank barrier，使两个 rank
从同一采样边界开始；barrier 本身不计入 E2E。

| implementation | dedup mode | `num_sms` | max-rank median E2E |
|---|---|---:|---:|
| `torch_reference_port` | `zero` | 24 | 2349.266 ms |
| `torch_reference_port` | `current` | 24 | 3092.386 ms |
| `torch_reference_port` | `src` | 24 | 3072.568 ms |
| `triton_reference_semantic` | `zero` | 24 | 2.819 ms |
| `triton_reference_semantic` | `current` | 24 | 3.478 ms |
| `triton_reference_semantic` | `src` | 24 | 3.556 ms |

性能结果必须和以下条件一起解读：NPU 物理卡号、world size、case、dedup mode、
warmup/repeat、`num_sms` 和是否复用了编译 cache。
