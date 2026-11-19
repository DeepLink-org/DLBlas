# KernelSwift T1：MXFP4 W4A16 分组专家矩阵乘算子—编译协同优化

**报告日期**：2026-08-31

**赛题**：T1 MXFP4 W4A16 分组专家矩阵乘

**作品形态**：A（NineToothed 算子）+ B（通用编译器/平台后端）统一构建与评测

**当前状态**：统一源码使用本 PR 的最终 `ntops` HEAD 与 `ninetoothed@6b79203`，已在 BW 与天垓150通过正确性和相关编译器回归；BF16 已完成 10 个双平台 A+B 代表性能点，FP16 已完成 BW 的 5 个代表性能点并保留天垓150的 3 个扩展性能点，全部 `speedup > 1`

## 1. 摘要

T1 面向 MoE 推理中的低精度专家计算。权重使用 OCP/vLLM MXFP4 编码：每字节包含两个 E2M1 值，每 32 个逻辑权重共享一个 E8M0 scale；activation 为 BF16/FP16，按 expert offsets 组成 ragged grouped-M 输入。朴素实现通常先完整反量化权重，再逐 expert 调用 GEMM，产生额外显存流量、临时张量和多个 Python/kernel launch。

本作品在 NineToothed application 中即时解码 E2M1 与 E8M0，将解码、缩放和 dot 融合，FP32 累加后直接写回输出；wrapper 根据实际 expert token 上界与 N 选择 tile，并缓存已验证的 stable-buffer launch。B 部分补齐可复用的 `bitcast`、`interleave`、`transpose` SSA lowering，并在严格 no-alias/稳定对象契约下将 Triton specialization 绑定为直接 compiled launch，同时保持 current-stream 语义和安全回退。

相同最终源码在天垓150和 BW 上都取得 `15 passed` 的 T1 测试和 `98 passed` 的相关编译器回归。BF16 的天垓150 5 个代表 case 平均 speedup 为 **2.970092×**，最低 **1.504281×**；BW 在同样的 50/500/9 协议下平均 **5.892741×**，最低 **2.659205×**。FP16 在 BW DTK25.04 的 5 个代表 case 平均 speedup 为 **6.183218×**，最低 **2.579908×**；天垓150的 3 个扩展 case 平均 speedup 为 **2.058291×**，最低 **1.250629×**。结果汇总见本报告第 7 节；BF16 和 FP16 的所有已记录性能点均大于 1。

## 2. 代码版本与复现锚点

结果对应本次 PR 中的提交链。以下记录官方基线 commit、本题提交和统一验证锚点，便于从 PR 历史复现。

| 仓库 | 官方基线 commit | 本题提交/验证锚点 |
|---|---|---|
| ntops | `9ae4166` | 本 PR 的 T1 题目 commit；统一验证使用本 PR 最终 HEAD |
| ninetoothed | `b77f930` | 通用 SSA 基础 `18c50cd`，T1 编译/运行时能力 `5b6abe8`；统一验证锚点 `6b79203` |

核心源码位于：

- `src/ntops/kernels/mxfp4_grouped_gemm.py`
- `src/ntops/torch/mxfp4_grouped_gemm.py`
- `tests/test_mxfp4_grouped_gemm.py`
- `benchmarks/benchmark_mxfp4_grouped_gemm_reference.py`
- `src/ninetoothed/backends/emitters/ssa.py`
- `src/ninetoothed/backends/emitters/triton.py`
- `src/ninetoothed/backends/materializers/triton.py`
- `src/ninetoothed/frontend/python.py`
- `src/ninetoothed/frontend/types.py`

## 3. 数学语义与接口

输入：

- activation `X`：`[sum(tokens_e), K]`，BF16 或 FP16；
- packed weight `W_p`：`[E, N, K/2]`，`uint8`，低/高 nibble 各存一个 E2M1；
- scale `S`：`[E, N, K/32]`，`uint8` E8M0；
- expert offsets：长度 `E` 的 exclusive end offsets，也兼容以 0 开头、长度 `E+1` 的边界向量；
- 输出 `Y`：`[sum(tokens_e), N]`，与 activation 同 dtype。

对 expert `e`：

$$
Y_e = X_e \cdot \operatorname{dequant}(W_{p,e}, S_e)^T,
$$

其中每个 E2M1 nibble 解码为集合
`{±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`，每 32 个值乘对应 E8M0 scale。实现用 FP16/BF16 dot operand 和 FP32 accumulator，最终转换为输出 dtype。

公开入口为：

```python
ntops.torch.mxfp4_grouped_gemm(
    input,
    weight_mxfp4,
    weight_scale,
    expert_offsets,
    num_experts=None,
    out=None,
)
```

`out` 用于 LLM 推理中稳定 buffer 复用，不提供时返回独立的新输出张量。

## 4. 修改内容与动机

### 4.1 A 部分：NineToothed 算子

1. **即时 MXFP4 解码**：E2M1 nibble 通过整数位运算构造幅值；E8M0 通过整数位模式和通用 `bitcast` 得到 FP32 scale，不物化完整 FP32 weight。
2. **解码—缩放—dot 融合**：偶/奇 nibble 解码后通过 `transpose + interleave` 恢复逻辑 K 顺序，在 K block 内完成 dot 与 scale 乘法。
3. **ragged expert 布局**：用 NineToothed jagged tensor 表达 expert 维，每个 expert 可有不同 token 数，支持零 token expert。
4. **两级 tile 选择**：M tile 由实际最大 expert token 数推导为 16/32/64；N tile 由输出宽度推导为 16/32/64/128。没有按公开 case ID 分支，也没有答案表。
5. **并行与调优**：expert、M tile、N tile 并行；允许 4/8 warps，由统一 auto-tuner 选择，pipeline stage 使用两个平台均合法的 1。
6. **稳定 buffer 热路径**：缓存 offsets padding/上界、layout state 和已绑定 launch；输入数据可原地变化，但 shape、stride、storage 与 routing contract 每次仍验证。
7. **安全输出语义**：显式 `out` 时允许复用；隐式分配时使用 workspace 计算后 clone，保证多次返回值互不覆盖。

### 4.2 B 部分：通用 SSA 与 Triton 后端

| 层次 | 修改 | T1作用 | 架构归属 |
|---|---|---|---|
| Python→SSA frontend | 函数式 `cast(..., bitcast=True)` | 表达 E8M0 位模式到 FP32 的无损解释 | 通用 frontend |
| 通用类型推导 | `interleave` 形状推导 | 两组 E2M1 nibble 恢复为连续 K | 通用 SSA 类型系统 |
| 统一 emitter | `bitcast`、`interleave`、block transpose | 将 T1 表达合法降到 Triton，而非在硬件后端写赛题模板 | 通用 emitter |
| Triton emitter | `tl.*` 合法拼写 | 同一 SSA 同时支持 BW 与 CoreX vendor Triton | 通用 Triton target |
| Triton materializer | 预验证 no-alias direct compiled launch | 去掉热路径重复 binder、specialization key、options 与 cache lookup | 通用运行时能力 |
| 兼容层 | `binder`/`device_caches` feature detection | 兼容不同厂商 Triton 3.1 fork | 能力探测，不按平台名硬编码 |
| 兼容层 | 二元嵌套 runtime stride guard | 避免旧 Triton 拒绝三项链式 `and` | 通用 emitter 兼容修复 |

直接 compiled launch 只通过私有 `_launch_prevalidated_noalias` 暴露。wrapper 必须先验证 dtype、device、shape、stride、offsets、output 不别名和对象稳定性；任何能力缺失或绑定失败都返回公开 checked handle。每次 launch 都重新读取 active device stream，非默认 stream 测试已覆盖。

## 5. 关键设计取舍

### 5.1 为什么不先完整反量化

完整反量化会把 4-bit weight 扩展为 BF16/FP16，额外读取 packed weight、写入大中间张量，再由 GEMM 重新读取。当前融合表达只读取 packed nibble 和 scale，在寄存器中构造 dot operand，降低显存流量并避免反量化 launch。

### 5.2 为什么 B 放在通用层

`bitcast`、`interleave`、transpose lowering 和 direct compiled binding 并非 T1 专属：量化解码、布局交织以及稳定 buffer 小算子都可复用。实现没有在海光或天数目录中放 shape 分支；两个平台当前都由同一 Triton target 和能力探测路径生成代码，符合“通用能力放通用层，硬件特性才放平台后端”的边界。

### 5.3 current-stream 与安全回退

绑定阶段只缓存 specialization、runtime arguments 和 grid，不缓存 stream；每次调用从 Triton driver 获取 current stream。若 vendor Triton 缺少底层 compiled ABI、输出可能 alias、对象/layout 变化或绑定异常，自动回退到完整 NineToothed checked launch。

## 6. 实验环境与 baseline

| 项目 | 海光 | 天数智芯 |
|---|---|---|
| GPU | BW 64 GiB，gfx936，warp 64 | 天垓150 / BI-V150 32 GiB，warp 64 |
| 软件栈 | DTK 25.04 | IX-ML 4.4.0 |
| Python | 3.10.12 | 3.12.3 |
| PyTorch | 2.5.1 | 2.7.1 |
| Triton | 3.1 | 3.1.0 |
| vLLM | 0.9.2 | 0.11.2 |

### 6.1 原生 PyTorch/vLLM 可执行性

- BW PyTorch 2.5.1没有 `torch._scaled_grouped_mm`；
- 天垓150 PyTorch 2.7.1存在该符号，但对 T1 BF16 activation 报错：`Expected mat_a to be Float8_e4m3 matrix got BFloat16`；
- 当前 vLLM 的直接 MXFP4 路径还依赖镜像中缺失的 `quark`。

因此当前不能得到原生 PyTorch/vLLM T1 latency。按照赛题“对齐 PyTorch 接口并参考 vLLM MXFP4 编码”的含义，预评测实现两条明确披露的 reference：

1. `vllm_semantic_fallback_predecoded`：按 vLLM/OCP 编码将 weight 预解码一次，然后逐 expert 执行 BF16/FP16 GEMM，作为主 baseline；
2. `pytorch_raw_reference`：每次都用 PyTorch tensor op 解码 E2M1/E8M0并执行 grouped GEMM，只作为开发正确性和额外性能参考。

这解决的是“当前环境如何获得可执行 baseline”；报告不会把语义 fallback 描述为厂商原生 kernel。

### 6.2 计时方法

两个平台的正式代表 case 均预热 50 次、每轮 500 次、共 9 轮，GPU 同步包围 wall-clock 区间并报告 9 轮中位数。每个平台都使用全新输出目录以及彼此隔离的 correctness、regression 和 benchmark JIT cache；长 K 性能测试也使用独立 cache。

## 7. 正确性、回归与性能

### 7.1 正确性与回归

| 平台 | T1算子测试 | 相关编译器回归 | 状态 |
|---|---:|---:|---|
| BW | 15 passed | 98 passed | 统一 HEAD 通过 |
| 天垓150 | 15 passed | 98 passed | 统一 HEAD 通过 |

15 个 T1 测试覆盖 BF16/FP16、ragged offsets、legacy boundaries、零 token expert、4组 shape、E2M1 编码契约、offsets 校验/缓存、动态输入、独立输出、非默认 CUDA stream，以及长归约 tile 选择契约。98 个相关回归同时覆盖 unified SSA lowering、Triton runtime/validation/emitter 和 jagged runtime stride 契约。

### 7.2 BF16 A+B 双平台 speedup

| 平台 | E | counts | N | K | baseline ms | A+B ms | speedup |
|---|---:|---|---:|---:|---:|---:|---:|
| BW | 1 | 1 | 16 | 32 | 0.062637 | 0.023555 | **2.659205×** |
| BW | 2 | 1,3 | 32 | 64 | 0.119773 | 0.024942 | **4.801980×** |
| BW | 2 | 5,2 | 64 | 128 | 0.122312 | 0.023991 | **5.098238×** |
| BW | 4 | 1,0,17,5 | 128 | 256 | 0.213102 | 0.027844 | **7.653477×** |
| BW | 4 | 1,3,5,7 | 256 | 256 | 0.233672 | 0.025260 | **9.250804×** |
| 天垓150 | 1 | 1 | 16 | 32 | 0.028474 | 0.018929 | **1.504281×** |
| 天垓150 | 2 | 1,3 | 32 | 64 | 0.055614 | 0.019579 | **2.840501×** |
| 天垓150 | 2 | 5,2 | 64 | 128 | 0.055897 | 0.019500 | **2.866542×** |
| 天垓150 | 4 | 1,0,17,5 | 128 | 256 | 0.102651 | 0.029008 | **3.538677×** |
| 天垓150 | 4 | 1,3,5,7 | 256 | 256 | 0.109238 | 0.026641 | **4.100457×** |

- BW 统一复测算术平均：**5.892741×**，最低 **2.659205×**；
- 天垓150统一复测算术平均：**2.970092×**，最低 **1.504281×**；
- 10/10 个双平台代表性能点均大于 1。

### 7.3 FP16 A+B 双平台 speedup

FP16 使用相同的 A+B 构建，只将 activation/output dtype 改为 FP16。BW 使用与 BF16 代表矩阵相同的 5 个 shape 和 `50/500/9` 协议；天垓150记录 3 个扩展 case，使用 `30/200/7` 协议。每个 case 的正确性均通过，speedup 按 `baseline_ms / submission_ms` 计算：

| 平台 | 协议 | E | counts | N | K | baseline ms | A+B ms | speedup |
|---|---|---:|---|---:|---:|---:|---:|---:|
| BW | 50/500/9 | 1 | 1 | 16 | 32 | 0.062881 | 0.024373 | **2.579908×** |
| BW | 50/500/9 | 2 | 1,3 | 32 | 64 | 0.125746 | 0.025228 | **4.984353×** |
| BW | 50/500/9 | 2 | 5,2 | 64 | 128 | 0.127347 | 0.024685 | **5.158904×** |
| BW | 50/500/9 | 4 | 1,0,17,5 | 128 | 256 | 0.205542 | 0.023939 | **8.586130×** |
| BW | 50/500/9 | 4 | 1,3,5,7 | 256 | 256 | 0.235877 | 0.024553 | **9.606796×** |
| 天垓150 | 30/200/7 | 1 | 1 | 16 | 32 | 0.029918 | 0.019345 | **1.546566×** |
| 天垓150 | 30/200/7 | 4 | 1,0,17,5 | 128 | 256 | 0.102779 | 0.030429 | **3.377677×** |
| 天垓150 | 30/200/7 | 4 | 1,3,5,7 | 512 | 1024 | 0.113628 | 0.090857 | **1.250629×** |

- BW FP16 性能算术平均：**6.183218×**，最低 **2.579908×**；
- 天垓150 FP16 扩展性能算术平均：**2.058291×**，最低 **1.250629×**；
- 8/8 个 FP16 性能点均大于 1；不同协议的结果分别统计，不合并计算平均值；
- 本次环境为 Python 3.10.12、PyTorch 2.5.1、Triton 3.1、vLLM 0.9.2，设备为 BW gfx936；原生 `torch._scaled_grouped_mm` 不存在，因此仍使用已披露的 vLLM 语义 fallback baseline。

统一复测原始证据目录为：

- BW：`<results-root>/t1/bw/{correctness,regression,benchmark}`；
- 天垓150：`<results-root>/t1/tiangai150/{correctness,regression,benchmark}`。

## 8. 优化来源与消融解释

性能来自 A+B 协同：

- A 消除完整反量化中间张量，并用 ragged tile 只计算实际 token；
- tile/warp 选择提高不同小M、N、K组合的利用率；
- offsets/layout缓存避免每次重建 jagged ABI；
- B 的 direct compiled launch 跳过重复 Python/Triton binder与 specialization cache lookup。

benchmark 同时记录 `submission_bound_ms`。天垓 final-clean 中 bound launch 为约 `0.0082–0.0278 ms`，公开 wrapper 为约 `0.0191–0.0285 ms`；两者差值反映参数校验和 Python wrapper 的剩余开销。

## 9. 工程质量与适用边界

### 9.1 已完成

- 接口新增最小且兼容；数学语义、测试和 baseline 口径均显式记录；
- 通用 frontend/SSA/emitter/runtime能力位于 ninetoothed 通用层；
- 没有按公开 shape、case ID或隐藏数据硬编码；
- 没有绕过 NineToothed 编译链调用未申报闭源算子；
- fast path 有 no-alias、对象身份、layout、offsets和 current-stream 测试；
- 两个平台使用同一源码逻辑，vendor差异通过 feature detection 安全回退；
- Ruff、贡献规范、T1测试和相关回归已在 BW 与天垓150通过。

### 9.2 已知限制

1. 原生 PyTorch/vLLM BF16×MXFP4 kernel在当前镜像不可执行；
2. 当前主要性能矩阵为 BF16；FP16 已通过正确性测试，并已形成 BW 与天垓150的性能记录。两平台 FP16 的 case/protocol 不完全对称，因此本报告分别统计，不将 FP16 汇总为双平台同矩阵 speedup；
3. 扩展矩阵已覆盖 FP16、大 N/K、更多 experts 和不均衡 routing，但仍不能代表隐藏 case；
4. 直接 compiled launch依赖 vendor Triton暴露稳定底层 ABI，能力缺失时会回退，性能可能下降但正确性不变；

## 10. 一键复现

完整命令、输出结构、baseline 与判定标准见 `docs/KERNELSWIFT_T1_REPRODUCE.md`。核心命令：

```bash
cd /path/to/ntops
bash scripts/run_kernelswift_t1.sh \
  --ninetoothed-dir /path/to/ninetoothed \
  --output-dir /path/to/fresh/results \
  --mode all
```
