# KernelSwift T2：Block-scaled FP8 矩阵乘算子—编译协同优化

**报告日期**：2026-08-31

**赛题**：T2 Block-scaled FP8 矩阵乘

**作品形态**：A（NineToothed 算子）+ B（通用编译器/平台后端）统一构建与评测

**当前状态**：统一源码使用本 PR 的最终 `ntops` HEAD 与 `ninetoothed@6b79203`，已在海光 BW 与天数智芯天垓150通过 T2 正确性、相关编译器回归和 12 个 A+B 性能点；全部 `speedup > 1`

## 1. 摘要

T2 面向 FP8 线性层、MoE 与 Attention 投影中的 block-scaled FP8 GEMM。其核心负载是按 K 维 128 元素分块，在每个分块内完成 FP8 矩阵乘，并将 activation scale 与 weight scale 融入 FP32 累加。直接反量化完整矩阵会增加显存流量和临时张量；直接把厂商兼容细节写入算子又会破坏 NineToothed 的跨平台抽象。

本作品在 A 部分使用 NineToothed 张量元编程表达 tiled GEMM、三种 scale layout、FP32 累加、可选 bias 与 BF16/FP16 输出；在 B 部分增加通用的 FP8 block-dot operand 合法化能力，由 Triton 工具链能力自动选择 native、FP16 或 BF16 lowering，同时复用静态布局分析与预验证 no-alias launch。公开接口不因平台改变，平台差异不按 case 或公开 shape 硬编码。

最终源码在海光 BW/DTK25.04 与天数智芯天垓150/IX-ML 4.4.0 上均得到 `30 passed` 的 T2 测试和 `120 passed` 的相关编译器回归。对 `(M,N,K)=(1/16/128,4096,4096)`、E4M3FN/E5M2 共六个 case，海光平均 speedup 为 **3.316458×**，天数平均为 **1.508378×**，两平台各 50% 合并为 **2.412418×**；12 个平台—dtype—shape 测试点全部大于 1。

## 2. 代码版本与复现锚点

结果对应本次 PR 中的提交链。以下记录官方基线 commit、本题提交和统一验证锚点，便于从 PR 历史复现。

| 仓库 | 官方基线 commit | 本题提交/验证锚点 |
|---|---|---|
| ntops | `9ae4166` | 本 PR 的 T2 题目 commit；统一验证使用本 PR 最终 HEAD |
| ninetoothed | `b77f930` | T2 可移植 FP8 dot 合法化 `4ea752e`；统一验证锚点 `6b79203` |

最终双平台数据只绑定本 PR 最终 `ntops` HEAD 与 `6b79203`。

## 3. 研究问题与设计推理

### 3.1 问题

目标是在同一份 NineToothed 算子源码上支持海光与天数智芯平台，并优化：

1. FP8 数据的分块加载与矩阵乘；
2. activation/weight scale 的索引、广播和融合；
3. 小 M 推理形状的 tile 与 launch 开销；
4. 厂商 Triton 对 FP8 `tl.dot` 能力不一致时的合法 lowering；
5. PyTorch `_scaled_mm` 风格接口的正确性、兼容性和失败行为。

### 3.2 关键观察

- scale block 固定为 128，scale 可以在每个 K block 的 dot 结果上融合，不需要生成完整反量化矩阵；
- M=1/16 的解码和小 batch 场景容易被过大的 M tile 与 Python launch 开销支配；
- CoreX 与 DTK 厂商 Triton 对原生 FP8 dot 的支持不同，算子层显式写死一种 cast 会把平台能力泄漏到算子源码；
- wrapper 已经完成 shape、dtype、stride、device 和输出别名条件校验，后端重复执行同一套绑定/alias 检查会产生纯运行时开销；
- PyTorch/vLLM 在目标平台提供了接口和编码规范参考，但不等于镜像中一定存在可执行的同语义二进制 baseline。

### 3.3 假设

如果将 K-block dot、两侧 scale 和 FP32 accumulation 表达为一个 tiled kernel，并把 FP8 dot 合法化放入能力驱动的后端层，那么可以同时：

- 保持算子源码与平台无关；
- 避免完整反量化与中间张量流量；
- 在小 M 形状减少无效工作和 launch 开销；
- 在两种国产 GPU 软件栈上通过同一数学语义的正确性测试；
- 相对可执行的开源兼容基线取得 `speedup > 1`。

### 3.4 可证伪判据

方案在出现任一情况时判为失败：

- 三种 scale layout、非 128 整除边界、bias 或两种输出 dtype 出现超容差误差；
- 后端 fallback 污染非 FP8 dot，或不同 fallback 共用错误的编译缓存；
- 任何已测正式 case 的 `baseline_latency / submission_latency <= 1`；
- 为特定公开 shape、case ID 或隐藏数据建立分支或答案表；
- baseline 与 submission 数学语义、输入或 dtype 不一致。

当前保存的测试与日志未触发以上失败条件。

## 4. 数学语义与接口

设输入 `A∈FP8^(M×K)`，列主序矩阵 `B∈FP8^(K×N)`，K block 大小为 128。对第 `q` 个 K block：

$$
P_q = \operatorname{dot}(A[:,q:q+128], B[q:q+128,:]),
$$

$$
Y = \operatorname{cast}_{out}\left(
    \sum_q P_q \odot S_A(q) \odot S_B(q) + bias
\right).
$$

`P_q` 与缩放后的累加值使用 FP32 accumulator，最终输出为 BF16 或 FP16。实现支持三种 scale layout：

| 变体 | `scale_a` | `scale_b` |
|---|---|---|
| 1×128 / 128×128 | `[M, ceil(K/128)]` | `[ceil(K/128), ceil(N/128)]` |
| 1×128 / 1×128 | `[M, ceil(K/128)]` | `[ceil(K/128), N]` |
| 128×128 / 1×128 | `[ceil(M/128), ceil(K/128)]` | `[ceil(K/128), N]` |

公开入口为：

```python
ntops.torch.scaled_mm(
    input,
    mat2,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
)
```

接口对齐 PyTorch `_scaled_mm` 的核心参数：

- `input` 必须二维、row-major contiguous；
- `mat2` 必须二维、`stride(0)==1` 的 column-major 视图；
- 两个矩阵使用 E4M3FN 或 E5M2；
- scale 使用 FP32 且连续；
- `out_dtype` 支持 BF16/FP16，默认 BF16；
- `bias` 可选，dtype 与输出一致；
- 高精度输出时，PyTorch 不应用 `scale_result`，实现只校验其兼容性而不修改结果；
- 实现始终 FP32 累加，因此 `use_fast_accum` 不降低当前数值保证。

FNUZ FP8 格式在当前两平台的软件栈与后端语义不一致，wrapper 明确拒绝，而不是静默按错误格式解释。

## 5. 修改内容与动机

### 5.1 A 部分：NineToothed 算子实现

涉及文件：

- `src/ntops/kernels/scaled_mm.py`
- `src/ntops/torch/scaled_mm.py`
- `src/ntops/kernels/__init__.py`
- `src/ntops/torch/__init__.py`
- `tests/test_scaled_mm.py`

主要修改：

1. **张量化分块表达**：output 按 `(BLOCK_M,BLOCK_N)` 切块；A 按 `(BLOCK_M,128)`、B 按 `(128,BLOCK_N)` 切块，K block 作为静态归约域。
2. **缩放融合**：每个 K block 执行一次 dot，再将对应 `scale_a × scale_b` 广播到输出 tile 并立即累加，不物化完整反量化矩阵。
3. **FP32 accumulation**：dot partial 的缩放结果累加到 FP32 accumulator，完成后才转为输出 dtype。
4. **三种布局统一表达**：通过 `ScalingVariant` 和 arrangement 选择 scale tile，application 的数学主体保持一致。
5. **shape-specialized tile**：tile 由矩阵 extent 和 scale layout 推导。M=1 使用 16 行 masked tile；普通 row-scaled 路径 M tile 最大为 64；weight row-scaled 变体可使用 128 行。不存在 case ID 或公开 shape 答案表。
6. **融合 bias**：bias 作为输出 tile 的广播输入，在 accumulator 写回前融合。
7. **静态 kernel cache**：wrapper 将 shape、dtype、scale layout、bias 和 tile 元参数作为编译契约，复用 `_cached_make`。
8. **低开销调用**：wrapper 完整校验输入后，为新分配且不别名的 output 使用 `_launch_prevalidated_noalias`；其他后端自动回退公开 kernel handle。

### 5.2 B 部分：通用编译器与 Triton 后端

T2 实际依赖的 B 部分能力如下：

| 层次 | 修改 | T2 作用 | 架构归属 |
|---|---|---|---|
| 通用 SSA emitter | block dot operand coercion hook | 在 lowering 阶段合法化 dot operands，而不是把厂商 cast 写入算子 | 通用编译器扩展点 |
| Triton backend options | `fp8_dot_fallback=auto/none/float16/bfloat16` | 用显式、可测试、可缓存的能力配置选择 lowering | 通用 Triton 后端 |
| 工具链能力检测 | 根据 Triton distribution metadata 选择有效 fallback | CoreX→BF16、DTK→FP16、支持 native 的上游 Triton→none | 平台工具链能力逻辑 |
| Triton emitter | 已知 FP8 dtype 在 `tl.dot` 前统一转换 | 避免 CoreX 错误 lowering 与 DTK 编译失败 | 后端合法化 |
| 编译缓存 | 有效 fallback 进入 cache key | 防止不同硬件/工具链误复用 kernel | 通用编译缓存语义 |
| 静态布局与 SSA application block | 静态 shape/stride、合法 tile 与 mask 分析 | 生成 tiled GEMM 和边界 mask | 通用编译器层 |
| 运行时 | 预验证 no-alias launch plan | 跳过 wrapper 已经完成的重复 ABI、绑定和 alias 检查 | 通用私有快速路径 |

`fp8_dot_fallback` 是 Triton 后端的通用能力，而不是 `scaled_mm` 专用 pass。测试使用独立的最小 block-dot kernel 验证 option normalization、source emission、非 FP8 保持、未知 dtype guard 和 cache key 隔离。

## 6. 关键设计取舍

### 6.1 为什么不在 A 部分直接写死 FP16 cast

早期 A 部分曾在 dot 前显式将两侧 FP8 转成 FP16。它能绕过部分厂商编译问题，但会把平台限制固化到算子表达，且 CoreX 的有效兼容类型与 DTK 不同。最终实现撤回算子内 cast，把合法化交给 B 部分；算子只表达“对 FP8 tile 做 dot”的数学意图。

### 6.2 为什么保留 FP32 accumulator

block scale 可能放大不同 K block 的 partial。若直接以 BF16/FP16 累加，误差会随 K block 数增长。FP32 accumulator 增加少量寄存器成本，但提高跨 dtype、scale layout 与非整除边界的稳定性，并使 `use_fast_accum=False` 的语义明确。

### 6.3 为什么对小 M 使用较小 tile

MoE、Attention decode 和小 batch 线性层常见 M=1/16。M tile 过大时，大部分行被 mask，造成无效 load、dot lane 和输出 predicate。当前 tile 由 extent 的 2 次幂可整除性与 layout 约束推导，而不是仅对六个 benchmark shape 建表。

### 6.4 为什么使用预验证 launch

公开 kernel handle 必须对一般调用执行完整 ABI、layout 和 alias 检查。T2 wrapper 每次已经验证 shape、stride、dtype、device，且 output 是新分配张量，因此可安全使用私有 no-alias launch。该优化不删除公开检查，也不改变外部接口；后端不提供私有入口时自动回退。

### 6.5 被否决或撤回的实验

开发历史保留了可追溯的负结果：

- 两 warp candidate 在目标 shape 上更慢，已撤回；
- row-scaled M tile autotune 没有形成跨平台稳定收益，已撤回；
- 强制 BF16 dot fallback 不适合两套平台，改为工具链能力选择；
- 静态 K loop lowering 的实验没有有效收益，已回退；
- 所有诊断性 benchmark 开关均未进入最终 timed kernel。

这些负结果说明最终配置不是只保留“成功数据”的 shape 查表，但当前没有为每个撤回实验保存统一格式的数值消融表。

## 7. 架构与工程规范符合性

### 7.1 接口改动最小

- 只新增 `ntops.torch.scaled_mm` / `_scaled_mm` 入口与导出；
- 参数保持 PyTorch 风格，不增加海光/天数专用公开参数；
- 不修改输入数学语义、官方测试或官方计时程序；
- 对不支持的 dtype/layout 显式报错，不静默回退到错误算法。

### 7.2 通用能力与平台特化边界

- scale layout、tile、dot-scale fusion 属于算子层；
- FP8 operand coercion hook、cache key、静态布局和 launch plan 属于通用编译器层；
- CoreX/DTK 差异只通过 Triton 工具链能力选择 fallback，不出现在 T2 shape 或语义分支；
- 没有在海光或天数后端复制同一套算子逻辑。


## 8. 实验环境与方法

### 8.1 环境

| 项目 | 海光 | 天数智芯 |
|---|---|---|
| GPU | BW，64 GiB，gfx936 | 天垓150 / BI-V150，32 GiB |
| 软件栈 | DTK 25.04 | IX-ML 4.4.0，Driver 4.4.0 |
| Python | 3.10.12 | 3.12.3 |
| PyTorch | 2.5.1 | 2.7.1 |
| Triton | 3.1 | 3.1.0 |
| vLLM | 0.9.2 | 0.11.2 |
| submission fallback | FP16 | BF16 |
| timed baseline | `vllm_portable`，FP16 fallback | `vllm_portable`，BF16 fallback |

### 8.2 正确性

独立 FP32 eager reference 对每个 K block 执行 FP32 matmul 和 scale，再累加并转为目标 dtype。T2 测试容差为 `rtol=0.05, atol=0.05`，覆盖：

- M=1、16 和非规则 `(33,257,385)`；
- 三种 scale layout；
- BF16/FP16 输出；
- bias 与高精度输出下的 `scale_result` 行为；
- M/N/K 非 128 整除边界 mask；
- tile 选择规则；
- FNUZ 明确拒绝；
- FP8 fallback option、source emission 与 cache key。

### 8.3 性能

- shape：`M=1/16/128，N=K=4096`；
- dtype：E4M3FN 与 E5M2；
- scale layout：1×128 / 128×128；
- output：BF16；
- 每轮在首次 JIT 和正确性通过后计时；
- 三轮交替 submission/baseline 的先后顺序；
- 每段计时前后 GPU synchronize；
- speedup 使用三轮中位 latency 计算；
- 计时后再次验证 submission 输出；
- eager reference 不计时。

## 9. 正确性与回归结果

| 平台 | T2 算子测试 | FP8 fallback 专项回归 | 性能 case 计时后正确性 |
|---|---:|---:|---:|
| 海光 BW/DTK25.04 | `30 passed in 78.98s` | `120 passed in 10.14s` | 6/6 pass |
| 天数智芯天垓150 | `30 passed in 23.59s` | `120 passed in 9.31s` | 6/6 pass |


上述数据来自全新目录的一键脚本正式复测，不是短时 smoke；原始日志位于同一结果目录。

## 10. 性能结果

### 10.1 海光 BW/DTK25.04

| FP8 dtype | M | submission (ms) | baseline (ms) | speedup |
|---|---:|---:|---:|---:|
| E4M3FN | 1 | 0.225511 | 0.809266 | **3.588588×** |
| E4M3FN | 16 | 0.227955 | 0.858246 | **3.764980×** |
| E4M3FN | 128 | 0.323871 | 0.938118 | **2.896579×** |
| E5M2 | 1 | 0.077159 | 0.274031 | **3.551511×** |
| E5M2 | 16 | 0.077034 | 0.332744 | **4.319443×** |
| E5M2 | 128 | 0.186894 | 0.332231 | **1.777644×** |

E4M3FN 平均 **3.416716×**，E5M2 平均 **3.216199×**，六个 case 平均 **3.316458×**。

### 10.2 天数智芯天垓150

| FP8 dtype | M | submission (ms) | baseline (ms) | speedup |
|---|---:|---:|---:|---:|
| E4M3FN | 1 | 0.290570 | 0.431914 | **1.486437×** |
| E4M3FN | 16 | 0.284554 | 0.525686 | **1.847403×** |
| E4M3FN | 128 | 0.504879 | 0.607184 | **1.202633×** |
| E5M2 | 1 | 0.289866 | 0.432237 | **1.491161×** |
| E5M2 | 16 | 0.290177 | 0.520649 | **1.794246×** |
| E5M2 | 128 | 0.506346 | 0.621988 | **1.228385×** |

E4M3FN 平均 **1.512158×**，E5M2 平均 **1.504597×**，六个 case 平均 **1.508378×**。

### 10.3 跨平台合并

按海光/天数各 50% 合并：

| 汇总项 | speedup |
|---|---:|
| E4M3FN | **2.464437×** |
| E5M2 | **2.360398×** |
| 12 个平台测试点总体算术平均 | **2.412418×** |


## 11. 基线可执行性说明

赛题写明“优先 PyTorch，其次 vLLM”。本作品遵循该优先级先探测目标镜像：

1. PyTorch `_scaled_mm` 接口存在，但两平台无法执行当前 block scaling 语义；
2. 上游 vLLM 的 block-scaled FP8 Triton 路径在厂商工具链上不能同时满足编译与正确性；
3. 因此使用 vLLM 源码结构的 portable compatibility baseline，且将兼容 cast 对 baseline 与 submission 按各自工具链语义独立应用；
4. eager reference 从不进入 timed 区间。

`vllm_portable` 的作用是获得可解释的性能证据。

## 12. 工程质量、兼容性与适用边界

### 12.1 已覆盖能力

- E4M3FN/E5M2 输入；
- BF16/FP16 输出；
- 三种 block scaling layout；
- 可选 bias；
- 非 128 整除边界；
- M=1 到较大 M 的静态 shape；
- CoreX/DTK fallback 和支持 native FP8 dot 的上游 Triton 保持路径。

### 12.2 已知限制

- 当前不支持 FNUZ FP8；
- 当前不输出 FP8，因此 `scale_result` 只做接口兼容校验；
- `input` 需 row-major contiguous，`mat2` 需 column-major；
- 当前性能表只覆盖 1×128/128×128 scale layout，其他两种布局已有正确性测试但尚无正式性能表；
- 两套目标工具链使用 FP16/BF16 dot fallback，尚未利用未来硬件/编译器可能提供的原生 FP8 dot 峰值；
- 当前 baseline 为 vLLM-derived portable compatibility baseline，不是未经修改的上游二进制；

### 12.3 第三方代码

`benchmarks/vllm_portable_scaled_mm.py` 参考 vLLM `fp8_utils.py` 中的 `w8a8_triton_block_scaled_mm`，遵循 Apache-2.0。来源：

- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM LICENSE](https://github.com/vllm-project/vllm/blob/main/LICENSE)

该第三方适配只存在于 benchmark 目录，submission 算子不调用它。

## 13. 复现与证据映射

一键命令、缓存隔离、输出文件和判定标准见 `docs/KERNELSWIFT_T2_REPRODUCE.md`。结果汇总见本报告第 10 节。服务器原始证据根目录：`<results-root>/t2/`。
