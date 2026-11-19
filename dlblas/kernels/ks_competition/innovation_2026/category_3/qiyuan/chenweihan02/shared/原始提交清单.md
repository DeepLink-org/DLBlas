# KernelSwift 算子创新大赛统一提交清单

## 1. 队伍信息

| 字段 | 内容 |
|---|---|
| 参赛队伍 | `NULL` |
| 队长 | `陈伟汉` |
| 参赛成员 | `陈健勇` |
| 所选赛题 | T1、T2、T3、T4 |

## 2. 仓库与 commit 映射

### 2.1 `ntops` 算子提交

| 项目 | 内容 |
|---|---|
| upstream base | `9ae4166` |
| 验证代码锚点 | PR final HEAD |

| 赛题 | commit | 主要内容 |
|---|---|---|
| T1 | `ec8f4e1` | MXFP4 W4A16 分组专家 GEMM、Torch wrapper、测试、benchmark、报告与复现脚本 |
| T2 | `8a5be7a` | Block-scaled FP8 GEMM、Torch wrapper、测试、baseline 探测、benchmark、报告与复现脚本 |
| T3 | `7fc8074` | Gated RMSNorm 融合算子、Torch wrapper、测试、vLLM benchmark、报告与复现脚本 |
| T4 | 本 commit（PR final HEAD） | MLA RoPE 与压缩 KV Cache 融合算子、Torch wrapper、测试、vLLM benchmark、报告与复现脚本及统一 manifest |

`src/ntops/kernels/__init__.py` 与 `src/ntops/torch/__init__.py` 在四个题目 commit 中依次追加导出。算子数学接口、测试输入语义和基线实现未被修改。

### 2.2 `ninetoothed` 编译器/后端提交

| 项目 | 内容 |
|---|---|
| upstream base | `b77f930` |
| 验证代码锚点 | `6b79203` |

| commit | 主要关联赛题 | 能力边界 |
|---|---|---|
| `18c50cd` | T1/T3 及共享基础 | 通用 SSA tensor lowering、静态布局/stride、标量 ABI 与 Triton runtime 基础能力 |
| `5b6abe8` | T1/T3 | `bitcast`、`interleave`、transpose 与已验证 Triton 直接绑定能力 |
| `4ea752e` | T2 | 按工具链能力选择的可移植 FP8 dot 合法化及专项回归 |
| `0dbe0e9` | T4 | MLA cache 所需 masked-producer sinking、CUDA 低开销 launch 与 CoreX 工具链支持 |
| `6b79203` | T1-T4 统一验证 | 修正 jagged public value 的 runtime stride 校验并增加定向回归；dense Tensor 校验保持不变 |

该分支不包含按公开 case 编号、隐藏数据或答案表选择结果的代码。通用分析位于 SSA/frontend/runtime 层，平台后端只保留工具链、FP8 能力和 CUDA/CoreX launch 等硬件相关实现。

## 3. A/B 修改边界

| 赛题 | A：`ntops` 算子实现 | B：`ninetoothed` 通用编译/后端 |
|---|---|---|
| T1 | MXFP4 解码、分组专家映射、tile 与并行策略、融合累加 | `bitcast`、`interleave`、transpose lowering、稳定参数直接绑定、jagged runtime 契约 |
| T2 | block-scale 索引、FP8 tile、FP32 累加、输出转换 | FP8 dot operand 合法化、按 Triton 工具链能力选择 FP16/BF16 fallback |
| T3 | RMSNorm、门控、affine 与残差融合；静态语义变体 | 通用静态布局、application block、标量 ABI、预验证 no-alias launch |
| T4 | RoPE、KV 拼接与 cache 写入融合；scalar/pair 能力驱动分派 | gather/SSA lowering、masked producer sinking、CUDA 直接 launch、CoreX 编译器发现 |

benchmark 文件不被 submission kernel 导入；第三方实现仅作为正确性 reference 或报告中明确披露的性能 baseline。

## 4. 统一 HEAD 双平台最终复测

下表全部来自本 PR 最终 `ntops` HEAD 与 `6b79203` 的干净工作树。各题 correctness、题目脚本列出的编译器回归和 benchmark 使用彼此隔离的新缓存；所有 benchmark case 均通过正确性检查。

| 赛题 | 海光 BW DTK25.04 | 天数智芯天垓150 | timed baseline |
|---|---|---|---|
| T1 | `15 passed`；回归 `98 passed`；5 case 平均 `5.892741×`，最低 `2.659205×` | `15 passed`；回归 `98 passed`；5 case 平均 `2.970092×`，最低 `1.504281×` | `vllm_semantic_fallback_predecoded`，明确披露的 vLLM/OCP 语义兼容路径 |
| T2 | `30 passed`；回归 `120 passed`；6 case 平均 `3.316458×`，最低 `1.777644×` | `30 passed`；回归 `120 passed`；6 case 平均 `1.508378×`，最低 `1.202633×` | `vllm_portable`，明确披露的 vLLM-derived portable compatibility baseline |
| T3 | `32 passed`；回归 `94 passed`；3 case 平均 `1.2626×`，最低 `1.2403×` | `32 passed`；回归 `94 passed`；3 case 平均 `1.3436×`，最低 `1.3295×` | 未修改 vLLM `RMSNormGated` 调用链 |
| T4 | `26 passed`；回归 `122 passed`；6 case 平均 `2.199483×`，最低 `1.4298×` | `26 passed`；回归 `122 passed`；6 case 平均 `1.5766×`，最低 `1.1648×` | 未修改 vLLM `RoPE.forward_cuda + concat_and_cache_mla` 调用链 |

所有 speedup 均按 `baseline_latency / submission_latency` 计算，不使用 PyTorch eager latency。T1/T2 因目标镜像中的原生 PyTorch 接口不能执行同语义工作负载，使用报告中明确披露的兼容 baseline；它们不得表述为原生 PyTorch 或未经修改的上游 vLLM binary。

远程原始结果目录（路径为示意，实际由运行时的 `--output-dir` 指定）：

- BW：`<results-root>/bw/`；
- 天垓150：`<results-root>/tiangai150/`。

## 5. 一键联合复测

以下命令只能在远程 GPU 主机执行。两个仓库、结果和缓存必须位于远程主机的指定工作目录下；每题、每平台必须使用全新输出目录。

```bash
export KS_ROOT=/path/to/remote/workspace
export NTOPS_DIR=${KS_ROOT}/ntops
export NINETOOTHED_DIR=${KS_ROOT}/ninetoothed

cd "${NTOPS_DIR}"

bash scripts/run_kernelswift_t1.sh \
  --ninetoothed-dir "${NINETOOTHED_DIR}" \
  --output-dir "${KS_ROOT}/results/t1-$(date +%Y%m%d-%H%M%S)" \
  --mode all

bash scripts/run_kernelswift_t2.sh \
  --ninetoothed-dir "${NINETOOTHED_DIR}" \
  --output-dir "${KS_ROOT}/results/t2-$(date +%Y%m%d-%H%M%S)" \
  --baseline vllm_portable \
  --mode all

bash scripts/run_kernelswift_t3.sh \
  --ninetoothed-dir "${NINETOOTHED_DIR}" \
  --output-dir "${KS_ROOT}/results/t3-$(date +%Y%m%d-%H%M%S)" \
  --mode all

NINETOOTHED_ROOT="${NINETOOTHED_DIR}" \
  bash benchmarks/run_kernelswift_t4.sh \
  "${KS_ROOT}/results/t4-$(date +%Y%m%d-%H%M%S)"
```

## 6. 文档索引

| 赛题 | 技术报告 | 一键复现 |
|---|---|---|
| T1 | `docs/KERNELSWIFT_T1_REPORT.md` | `docs/KERNELSWIFT_T1_REPRODUCE.md` |
| T2 | `docs/KERNELSWIFT_T2_REPORT.md` | `docs/KERNELSWIFT_T2_REPRODUCE.md` |
| T3 | `docs/KERNELSWIFT_T3_REPORT.md` | `docs/KERNELSWIFT_T3_REPRODUCE.md` |
| T4 | `docs/KERNELSWIFT_T4_REPORT.md` | `docs/KERNELSWIFT_T4_REPRODUCE.md` |
