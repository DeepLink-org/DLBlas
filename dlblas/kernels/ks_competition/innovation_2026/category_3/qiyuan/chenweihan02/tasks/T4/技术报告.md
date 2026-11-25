# KernelSwift T4：MLA RoPE 与压缩 KV Cache 写入融合算子—编译协同优化

**报告日期**：2026-08-31

**赛题**：T4 MLA RoPE 与压缩 KV Cache 写入融合

**作品形态**：A（NineToothed 算子）+ B（通用编译器/平台后端）统一构建与评测

**当前状态**：统一源码使用本 PR 的最终 `ntops` HEAD 与 `ninetoothed@6b79203`，已在海光 BW 与天数智芯天垓150通过 T4 正确性、相关编译器回归和 12 个 vLLM 性能 case；全部 `speedup > 1`

## 1. 摘要

T4 面向 MLA 解码和长上下文推理，将 query/key 的 RoPE 与压缩 KV Cache 写入融合为一个 NineToothed 算子。公开接口保持为：

```python
ntops.torch.mla_rope_cache(
    q_pe,
    k_pe,
    kv_c,
    positions,
    cos_sin_cache,
    slot_mapping,
    kv_cache,
)
```

`q_pe` 和 `kv_cache` 原地更新，数学语义、输入输出和原接口保持不变。实现没有读取 case 标识或隐藏数据，没有建立公开 shape 答案表，也没有绕过 NineToothed 编译链调用未申报闭源算子。

当前统一 A+B 结果为：

- 天数旧版天垓150：T4 正确性 `26 passed`、编译器/后端定向回归 `122 passed`，6 个正式 case 全部
  `speedup > 1`，平均 speedup `1.5766×`；
- 海光 BW DTK25.04：T4 正确性 `26 passed`、编译器/后端定向回归 `122 passed`，6 个正式 case 全部
  `speedup > 1`，平均 speedup `2.199483×`；
- 两个平台均使用最终统一提交完成独立结果目录和 SHA-256 记录；

### 1.1 代码版本与复现锚点

结果对应本次 PR 中的提交链。以下记录官方基线 commit、本题提交和统一验证锚点，便于从 PR 历史复现。

| 仓库 | 官方基线 commit | 本题提交/验证锚点 |
|---|---|---|
| ntops | `9ae4166` | 本 PR 的 T4 题目 commit；统一验证使用本 PR 最终 HEAD |
| ninetoothed | `b77f930` | T4 MLA cache 后端支持 `0dbe0e9`；统一验证锚点 `6b79203` |

## 2. 修改内容与动机

### 2.1 A 部分：NineToothed 算子实现

| 文件 | 修改 | 动机 |
|---|---|---|
| `src/ntops/kernels/mla_rope_cache.py` | 保留低启动成本的融合 scalar-lane 路径 | decode 和短 prefill 主要受 launch 开销影响 |
| `src/ntops/kernels/mla_rope_cache_pair.py` | 新增 one-thread-per-RoPE-pair 布局 | 大 query 域复用同一 pair 的 cos/sin，并减少线程数 |
| `src/ntops/torch/mla_rope_cache_pair.py` | 构造 stride-2 even/odd view，缓存稳定 launch | 避免完整中间张量和重复 ABI 准备 |
| `src/ntops/torch/mla_rope_cache.py` | 按后端能力和工作量统一分派 | 同时保护小 workload 启动延迟和大 workload 吞吐 |

pair 路径只在以下条件成立时启用：

```text
NINETOOTHED_BACKEND=cuda
q_pe.numel() >= 2^18
q_pairs_per_token >= cache_width
```

条件来自后端能力、query 工作量和 cache 行宽，不包含 case 编号、公开 shape 身份或预计算答案。其他情况自动回退到原始融合路径。

### 2.2 B 部分：通用 SSA 与目标后端

| 文件 | 层次 | 修改 |
|---|---|---|
| `ninetoothed/backends/emitters/ssa.py` | 通用 SSA emitter | 对 C-style masked store 做 producer sinking，并统一 mixed-dtype `where` 分支类型 |
| `ninetoothed/backends/materializers/cuda.py` | CUDA 目标后端 | 预校验后缓存 device pointer 和标量 ABI，每次调用只重新获取当前 PyTorch stream 并启动 kernel |
| `ninetoothed/backends/materializers/cuda.py` | CoreX 相关后端处理 | 检测 `torch.corex` 的外部 kernel 空 CUDA Graph 行为，回退到正确的快速直接 launch |
| `ninetoothed/backends/toolchain.py` | 通用工具链接口 | 允许通过环境变量配置 CUDA 兼容编译器和源语言，使 CoreX 使用 clang/ivcore |

通用分析放在 SSA emitter，硬件运行时差异放在 CUDA materializer，编译器路径通过通用环境接口配置。代码没有把某个 T4 shape 直接写进海光或天数后端。

## 3. 方案与实现

### 3.1 融合数据流

单次 kernel 完成：

1. 根据 `positions` 加载 token 对应的 cos/sin；
2. 对所有 query heads 执行 interleaved RoPE，并原地写回 `q_pe`；
3. 对单个 key head 执行 RoPE；
4. 将 `kv_c` 与旋转后的 key 拼成压缩 cache 行；
5. 根据 `slot_mapping` 写入目标 block/slot；
6. `slot_mapping=-1` 时屏蔽 cache 写入。

旋转后的 key 不落到独立全局中间张量，因此减少一次 kernel launch 和中间内存流量。

### 3.2 Pair layout 与别名安全

大 workload 将连续的 RoPE 通道拆成两个 stride-2 view：

```text
q_even = q[..., 0::2]
q_odd  = q[..., 1::2]
```

一个线程同时负责 even/odd 通道并复用 cos/sin。因为两个输出 view 与输入别名，必须在任一 store 前物化两个输入 SSA 值：

```python
original_even = q_even + 0.0
original_odd = q_odd + 0.0
```

否则第二个通道可能重新读取已经更新的第一个通道。动态索引测试和 BF16/FP16 全矩阵验证了该处理。

### 3.3 Masked producer sinking

原 C-style emitter 先生成 store value，再计算/应用 mask。融合 query 与窄 cache 行时，超出 cache 行的 lane 虽不能写入，仍会执行 key、cos/sin load 和算术。

修改后先物化 mask，并将纯 producer slice 与 store 一起放进同一条件块。该优化位于通用 SSA emitter，对所有具有同类 masked-store 结构的 C-style 后端生效，不依赖 T4 名称。

### 3.4 低开销 CUDA launch

推理循环通常重复使用同一批 tensor 对象。公开入口首次调用仍执行完整契约检查，随后绑定并缓存：

- tensor device pointer；
- 标量 ABI 转换；
- keepalive 对象；
- device index。

每次调用只查询当前 PyTorch stream 并调用生成的 host launcher，既保留 stream 语义，也消除 Python ABI 热路径开销。若后端不提供绑定能力，则回退到原 checked launch。

CoreX 4.4 对外部 CUDA kernel 的 graph capture 会报告成功但生成空图，因此不能把空 graph replay 当作正确执行。该平台通过 `torch.corex` 能力标记禁用外部 graph replay，改用上述预校验快速 launch。

### 3.5 CoreX 工具链

天数旧版镜像同时存在 NVIDIA 兼容 `nvcc 10.2` 和 CoreX 原生 clang。系统 `nvcc` 可能返回零但不产生可加载 `.so`，因此统一构建显式使用：

```text
CUDA_HOME=/usr/local/corex
NINETOOTHED_CUDA_COMPILER=/usr/local/corex/bin/clang++
NINETOOTHED_CUDA_LANGUAGE=ivcore
NINETOOTHED_BACKEND=cuda
```

代码只新增通用可配置接口，不把 `/usr/local/corex` 硬编码进编译器实现。

## 4. 实验与效果

### 4.1 天数环境

```text
GPU：Iluvatar BI-V150（32 GiB）
IX-ML：4.4.0
PyTorch：2.7.1
Triton：3.1.0
vLLM：0.11.2
CoreX clang：18.1.8（IX-ML 4.4.0）
```

### 4.2 正式基线与计时

未修改 vLLM 基线：

```text
rope.forward_cuda + ops.concat_and_cache_mla
```

NineToothed 提交路径：

```text
ntops.torch.mla_rope_cache
```

两条路径使用相同输入、BF16 dtype 和数学语义，并在计时前分别检查结果。计时交替预热 100 次，每轮 1000 次，共 9 轮，分别报告 wall-clock 样本中位数：

| Case | vLLM (ms) | NineToothed (ms) | 正式 speedup |
|---|---:|---:|---:|
| seq1 / 32 heads | 0.010055 | 0.005535 | 1.8165× |
| seq16 / 32 heads | 0.010477 | 0.006512 | 1.6090× |
| seq128 / 32 heads | 0.015783 | 0.008670 | 1.8204× |
| seq1 / 128 heads | 0.010707 | 0.005885 | 1.8194× |
| seq16 / 128 heads | 0.011055 | 0.009491 | 1.1648× |
| seq128 / 128 heads | 0.017988 | 0.014631 | 1.2295× |

以上均为相对 vLLM 的正式 speedup，不是 eager 对比。

### 4.3 分派消融

同一台天数主机上的设计空间对照：

| Case | 强制 scalar | 强制 pair | 统一分派 |
|---|---:|---:|---:|
| seq1 / 32 heads | 1.8174× | 0.8190× | 1.8482× |
| seq16 / 32 heads | 1.6044× | 0.8481× | 1.5850× |
| seq128 / 32 heads | 1.1917× | 1.2676× | 1.7932× |
| seq1 / 128 heads | 1.8336× | 0.8433× | 1.8000× |
| seq16 / 128 heads | 1.1458× | 0.9005× | 1.1668× |
| seq128 / 128 heads | 0.8693× | 1.2126× | 1.2228× |

强制 pair 会伤害小 workload，强制 scalar 在最大 case 低于 1。统一分派使 6 个 case 全部大于 1，说明双路径是必要的工程取舍。

### 4.4 Launch 消融

32-head 路径禁用可复用快速 launch 后，三个 case 的 speedup 分别为：

```text
0.2435× / 0.2453× / 0.3751×
```

这说明微秒级融合算子的主要瓶颈之一是运行时 launch/ABI 开销，B 部分的预校验绑定是端到端收益的关键来源。

### 4.5 海光 BW DTK25.04 最终统一复测

最终统一提交在 BW DTK25.04 的全新结果目录中完成复测：

```text
ntops@PR-final-HEAD
ninetoothed@6b79203
```

环境为 BW/gfx936、Python 3.10.12、PyTorch 2.5.1、HIP 6.3.25405、Triton 3.1、vLLM 0.9.2。
结果如下：

| Case | vLLM (ms) | NineToothed (ms) | 正式 speedup |
|---|---:|---:|---:|
| seq1 / 32 heads | 0.023896 | 0.010660 | 2.2417× |
| seq16 / 32 heads | 0.023658 | 0.011924 | 1.9842× |
| seq128 / 32 heads | 0.047728 | 0.010604 | 4.5007× |
| seq1 / 128 heads | 0.022710 | 0.015884 | 1.4298× |
| seq16 / 128 heads | 0.022176 | 0.015493 | 1.4313× |
| seq128 / 128 heads | 0.035463 | 0.022038 | 1.6092× |

本轮 T4 正确性为 `26 passed in 46.08s`，NineToothed 编译器回归为
`122 passed in 193.52s`，ntops 和 ninetoothed 的 Ruff 均通过。6 个正式 case 全部
`correctness=pass`，BW 平台 speedup 算术平均为 `2.199483×`。

完整证据位于：

```text
<results-root>/t4/bw/
```

### 4.6 双平台性能汇总

天数旧版天垓150的 6 个正式 case 平均 speedup 为 `1.5766×`，BW DTK25.04 的 6 个正式 case
平均 speedup 为 `2.199483×`。

## 5. 工程质量与适用边界

### 5.1 测试覆盖

T4 `26 passed` 覆盖：

- BF16/FP16；
- INT32/INT64 positions 和 slot mapping；
- 1/16/128 tokens；
- 32/128 query heads；
- padding slot；
- 重复调用和动态运行时索引。

NineToothed `122 passed` 覆盖 SSA lowering、masked producer sinking、混合 dtype `where`、运行时缓存、标量 ABI、静态布局/stride、backend registry 和 Triton runtime。T4 直接相关代码 ruff 全部通过。

### 5.2 接口与架构边界

- 公共算子接口和数学语义未修改；
- pair 路径是内部实现，不新增用户必传参数；
- producer sinking 位于通用 SSA 层；
- 可配置编译器/语言位于通用 toolchain；
- launch 优化和 CoreX graph 差异位于 CUDA materializer；
- 平台选择基于后端能力，不基于赛题 case 身份。

### 5.3 已知限制

- pair 路径要求 CUDA 兼容后端、连续输入、偶数 RoPE width 和 FP16/BF16；
- 非匹配 workload 回退原 scalar 融合路径；
- 当前 CoreX 阈值来自天数实测，换硬件后应通过 profile 重新验证；
- BW 最终复测使用 DTK25.04 的 Triton 路径；DTK26.04 的默认环境缺少 Triton/vLLM，未作为本次正式
  BW 结果环境；

### 5.4 第三方来源

实现未复制第三方 kernel 源码。vLLM 仅作为公开接口语义和正式性能基线，PyTorch 用于 tensor/runtime 接口；各自许可证随其原项目。

## 6. 复现与结果证据

一键入口：

```bash
bash benchmarks/run_kernelswift_t4.sh /path/to/fresh/result-dir
```

详细说明见 `docs/KERNELSWIFT_T4_REPRODUCE.md`。本轮天数原始日志位于：

```text
<results-root>/t4/tiangai150/
```

关键文件位于 `<results-root>/t4/tiangai150/`，包括 `FORMAL_SPEEDUP_SUMMARY`、`SHA256SUMS`
和 `logs/`。

BW DTK25.04 最终统一复测结果位于：

```text
<results-root>/t4/bw/
```

其中 `FORMAL_SPEEDUP_SUMMARY`、`SHA256SUMS` 和 `logs/` 分别记录正式 speedup、参与测试的
源码摘要以及环境、正确性、编译器回归和两个 head 配置的完整输出。复现时必须为每次运行
指定新的结果目录；脚本会拒绝复用已有目录和缓存。
