# KernelSwift T3：Gated RMSNorm 融合算子—编译协同优化

**报告日期**：2026-08-31

**赛题**：T3 Gated RMSNorm 融合算子

**作品形态**：A（NineToothed 算子）+ B（通用编译器/平台后端）统一构建与评测

**当前状态**：统一源码使用本 PR 的最终 `ntops` HEAD 与 `ninetoothed@6b79203`，已在两平台通过 T3 正确性、相关编译器回归和六个 vLLM 性能 case；全部 `speedup > 1`

## 1. 摘要

T3 面向 KDA、门控线性 Attention 和新型序列模型中的 Gated RMSNorm。朴素实现需要分别执行门控激活、逐元素乘、平方归约、倒平方根、归一化与 affine 缩放，容易产生多个 kernel launch 和中间张量流量。本作品使用 NineToothed 的张量元编程表达将上述步骤融合为一个 kernel，并通过静态布局、统一 SSA application-block lowering、能力驱动的 Triton 合法化和预验证低开销 launch 降低小算子的编译与运行时开销。

在同一份源码上，T3 正确性测试在海光 BW 与天数智芯天垓150均为 `32 passed`，T3 相关编译器回归均为 `94 passed`。相对未修改 vLLM 基线，海光三个 case 的平均 speedup 为 **1.2626×**，天数三个 case 的平均 speedup 为 **1.3436×**；六个 case 等权平均为 **1.3031×**。

## 2. 代码版本与复现锚点

结果对应本次 PR 中的提交链。以下记录官方基线 commit、本题提交和统一验证锚点，便于从 PR 历史复现。

| 仓库 | 官方基线 commit | 本题提交/验证锚点 |
|---|---|---|
| ntops | `9ae4166ad342e4745f0eed13a5a20d069e994fc0` | 本 PR 的 T3 题目 commit；统一验证使用本 PR 最终 HEAD |
| ninetoothed | `b77f930dc6c8b016e09adf33570d55a7bc8376c1` | 通用 SSA/运行时能力 `18c50cd`、`5b6abe8`；统一验证锚点 `6b79203` |

## 3. 数学语义与接口

设输入为 (x)，门控为 (g)，激活为 (a(g))，可选权重为 (w)，按隐藏维每 `group_size` 个元素分组。`norm_before_gate=False` 时：

$$
v=x\odot a(g),\qquad
y=\frac{v}{\sqrt{\operatorname{mean}_{group}(v^2)+\epsilon}}\odot w.
$$
`norm_before_gate=True` 时：

$$
y=\frac{x}{\sqrt{\operatorname{mean}_{group}(x^2)+\epsilon}}\odot a(g)\odot w.
$$

实现提供：

```python
ntops.torch.rms_norm_gated(
    input,
    gate,
    weight=None,
    eps=1e-5,
    group_size=None,
    norm_before_gate=False,
    activation="swish",
)
```

支持 `silu`/`swish` 与 `sigmoid`，支持有/无 affine weight、归一化前/后门控和多组归一化。输入、gate、weight 的形状、设备、dtype、末维连续性和 group 可整除性在 Python wrapper 中检查。公开接口只新增一个 T3 算子入口；NineToothed `Tensor` 的 `strides` 是可选参数，旧调用保持兼容。

## 4. 修改内容与动机

### 4.1 A 部分：NineToothed 算子实现

涉及文件：

- `src/ntops/kernels/rms_norm_gated.py`
- `src/ntops/torch/rms_norm_gated.py`
- `src/ntops/kernels/__init__.py`
- `src/ntops/torch/__init__.py`

主要修改：

1. 将 gate 激活、门控乘法、组内平方和归约、RMS 归一化和 affine 缩放表达为单个 NineToothed application，避免物化完整中间张量。
2. 输入先映射为 `[-1, group_size]` 的逻辑二维视图，一行对应一个独立归一化组；组间并行，组内沿隐藏维向量化和归约。
3. 输入、gate 和 weight 在寄存器计算前转为 FP32，平方和与归一化用 FP32 累加，最后写回输入 dtype。
4. 为 `activation × norm_before_gate × has_weight` 生成语义明确的静态变体，避免运行期在 kernel 内分支。
5. 对最常用的“二维输入 + 单组”路径保留原二维张量和一维 weight，不创建 stride-zero expanded view，也不做多余 reshape。
6. wrapper 提供必要的参数校验，并把实际 shape/stride 传给编译器建立静态契约。

### 4.2 B 部分：编译器与后端

下表只列 T3 实际依赖且有测试覆盖的能力，不把 `gather` 等 T4 能力归因于 T3。最终 ntops PR 的 T3 commit 只保留 RMSNorm 路径。

| 层次 | 修改 | T3 作用 | 架构归属 |
|---|---|---|---|
| 通用 Tensor/运行时 | 可选静态 source strides，并在 launch 时校验 | 删除重复 stride 参数和动态索引准备，同时防止错误复用 | 通用编译器层 |
| 通用布局分析 | 静态区间证明与 mask 简化 | 完整 tile 可去掉冗余边界 predicate，尾块仍保留 mask | 通用编译器层 |
| 统一 SSA emitter | 合法静态 application block、标量按值传递、静态循环策略 | 让 gate、归约、归一化和缩放在一个程序中生成；`eps` 不被误当指针 | 通用编译器层 |
| Triton materializer | 预验证 no-alias invocation plan 缓存 | wrapper 已检查契约后跳过重复 Python ABI、alias 和绑定准备 | 通用运行时能力，私有入口 |
| Triton 能力合法化 | 对 `backend=cuda && warp_size=64` 固定合法 `num_stages=1` | 适配 CoreX 现有软件流水能力 | 硬件能力驱动的后端逻辑 |
| HIP target 恢复 | 厂商 Triton ABI 不匹配时从 PyTorch 读取 gfx/warp target | 保证 BW 可编译，不改变数学语义 | HIP 后端兼容逻辑 |

预验证 launch 是私有能力 `_launch_prevalidated_noalias`。公开 kernel handle 仍执行完整参数、layout 和 alias 校验；只有 T3 wrapper 在每次调用已经验证 shape、stride、dtype、device，且 output 为新分配不别名张量时才使用快速路径。若后端不提供该能力，wrapper 自动回退到公开 handle。

## 5. 关键设计取舍

### 5.1 数据布局与并行策略

- 逻辑布局为“行/组 × group_size”；每行独立，天然无跨程序同步。
- 一次读取 input、gate 和可选 weight；中间值留在程序内，不写全局中间张量。
- weight 在单组常见路径保持一维广播；多组路径才生成对应的逻辑 group view。
- 静态 shape/stride 只在 wrapper 能证明契约时启用；不满足契约的输入在 launch 前报错，而不是静默产生错误索引。

### 5.2 融合与数值精度

所有非线性和 RMS 统计在 FP32 中执行，降低 BF16/FP16 平方和的误差风险。测试容差为 `rtol=2e-2, atol=2e-2`，与低精度推理算子的常见误差范围一致。最终输出保持输入 dtype。

### 5.3 通用优化与平台特化边界

静态布局证明、统一 SSA application block、标量 ABI 和预验证 runtime 都位于通用层，可被其他逐行归约/融合算子复用。平台相关逻辑不按赛题 shape 或 case ID 分支，而是依据 backend、warp size、gfx 架构和目标能力做合法化。实现没有公开 shape 答案表、隐藏 case 读取或外部闭源 kernel 绕行。

## 6. 实验环境与方法

| 项目 | 海光 | 天数智芯 |
|---|---|---|
| GPU | BW，64 GiB，gfx936，warp 64 | 天垓150 / BI-V150，32 GiB，warp 64 |
| Python | 3.10.12 | 3.12.3 |
| PyTorch | 2.5.1，HIP 6.3.25405 | 2.7.1，CUDA-compatible 10.2 |
| Triton | 3.1 | 3.1.0 |
| vLLM | 0.9.2 | 0.11.2 |
| 基线入口 | `Mixer2RMSNormGated.forward_cuda` | `rms_norm_gated` direct op |

T3 没有语义等价的公开 PyTorch 融合接口，且赛题对齐 vLLM `RMSNormGated`，因此按照“优先 PyTorch、其次 vLLM”采用 vLLM。PyTorch eager 只用于正确性 reference，不进入计时。

每个 case 交替预热 50 次，随后进行 7 轮配对计时，每轮每条路径执行 500 次；轮间交替基线与提交实现的先后顺序，使用 GPU 同步包围 wall-clock 区间，报告 7 个样本的中位数。

## 7. 正确性与性能结果

### 7.1 正确性与回归

| 平台 | T3 算子测试 | T3 相关编译器回归 | 结果 |
|---|---:|---:|---|
| 海光 BW | 32 passed | 94 passed | 通过 |
| 天垓150 | 32 passed | 94 passed | 通过 |

32 个 T3 case 来自 4 个 shape/group 组合 × 2 种 activation × 2 种门控顺序 × 2 种 affine 配置。相关编译器回归覆盖统一 SSA lowering、Triton 调优/运行时、标量参数、静态 bounds 和静态 strides。

### 7.2  A+B speedup

| 平台 | Shape | vLLM ms | A+B ms | speedup |
|---|---:|---:|---:|---:|
| 海光 BW | 64×128 | 0.084763 | 0.067287 | **1.2597×** |
| 海光 BW | 128×512 | 0.084719 | 0.068306 | **1.2403×** |
| 海光 BW | 16×4096 | 0.085418 | 0.066326 | **1.2878×** |
| 天垓150 | 64×128 | 0.094104 | 0.070783 | **1.3295×** |
| 天垓150 | 128×512 | 0.097470 | 0.071967 | **1.3544×** |
| 天垓150 | 16×4096 | 0.097530 | 0.072408 | **1.3469×** |

- 海光算术平均：**1.2626×**；
- 天数算术平均：**1.3436×**；
- 六个 case 等权平均：**1.3031×**；
- 最低 case：**1.2403×**，六个 case 均大于 1。

性能结果见上表。原始日志目录：

- 海光：`<results-root>/t3/bw/`
- 天数：`<results-root>/t3/tiangai150/`

## 8. 消融与优化来源

为隔离 B 部分的运行时贡献，保持同一生成 kernel、同一输入和固定 output，仅比较公开完整检查 launch 与 wrapper 可使用的预验证 no-alias launch。该实验不与 vLLM 比较，因此不作为赛事 speedup。

| 平台 | Shape | 完整检查 ms | 预验证 ms | 运行时微消融 speedup |
|---|---:|---:|---:|---:|
| 海光 BW | 64×128 | 0.043819 | 0.031003 | 1.4134× |
| 海光 BW | 128×512 | 0.044870 | 0.030885 | 1.4528× |
| 海光 BW | 16×4096 | 0.044710 | 0.031623 | 1.4138× |
| 天垓150 | 64×128 | 0.056039 | 0.043395 | 1.2914× |
| 天垓150 | 128×512 | 0.055744 | 0.043033 | 1.2954× |
| 天垓150 | 16×4096 | 0.056353 | 0.043113 | 1.3071× |

这说明对几十微秒级融合 kernel，重复 Python ABI/绑定/alias 检查是可观开销，预验证运行时能力对端到端结果有直接贡献。静态布局、融合表达和运行时优化之间存在耦合。

## 9. 工程质量、兼容性与适用边界

### 9.1 已完成

- T3 语义测试覆盖高维输入、分组、两种激活、两种门控顺序和有/无 weight；
- 统一 SSA、静态布局、标量 ABI、runtime cache 和 target 合法化有独立回归；
- 两个平台使用相同关键源码哈希、独立缓存和统一一键脚本；
- benchmark 同进程配对、交替顺序并记录完整样本；
- 公共接口变化最小，快速 launch 保持私有并有安全回退；
- 没有复制第三方 kernel；vLLM 只作为未修改 baseline，PyTorch 只用于张量与正确性参考。

### 9.2 已知限制

1. 当前正式性能预评测只有 3 个公开代表 shape，未知隐藏 case 可能包含不同 group_size、dtype、stride 或 batch 结构。
2. 正确性主矩阵使用 BF16，性能矩阵使用 BF16 + swish + `norm_before_gate=False` + affine weight；FP16/FP32及更广参数范围仍应扩测。
3. 相关编译器回归为 94 个定向测试。
4. 预验证快速路径要求 wrapper 每次重验 shape/stride/dtype/device，并保证新 output 不别名；不能由任意调用者无条件使用。
5. BW vLLM 0.9.2 镜像重启后可能同时暴露 ROCm 与 NVML 兼容发现。基准脚本只在平台解析阶段屏蔽伪 CUDA 计数，选择正确 ROCm 平台；该处理不修改或替换被计时的 vLLM kernel。

## 10. 一键复现

完整命令、目录约束、输出文件和判定方法见 `docs/KERNELSWIFT_T3_REPRODUCE.md`。核心命令为：

```bash
cd /path/to/ntops
bash scripts/run_kernelswift_t3.sh \
  --ninetoothed-dir /path/to/ninetoothed \
  --output-dir /path/to/fresh/results \
  --mode all
```
