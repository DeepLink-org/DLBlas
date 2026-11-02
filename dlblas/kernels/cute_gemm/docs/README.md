# 文档索引

cute_gemm 是用 CuTe DSL 编写的 CUDA kernel 库,覆盖 LLM 推理与训练的核心算子。
目标硬件:SM90(H100/H200)、SM100/SM110(B200/B300)、SM120(GeForce RTX 50)。

## 算子族

| 族 | 内容 | 入口 |
|---|---|---|
| GEMM | `gemm`/`gemm_add`/`gemm_rms` 等 20 余个变体,融合 epilogue | `cute_gemm/gemm_interface.py` |
| 激活 GEMM | 激活前向与梯度,gated 与非 gated 共用入口 | `GemmBase.gemm_with_activation(_grad)` |
| MoE | SM90 grouped GEMM 两个原语:上投影加激活、down 投影梯度 | `cute_gemm/fused_grouped_gemm/interface.py` |
| 归约 | rmsnorm/softmax 前反向、cross-entropy 前向、topk 反向 | `cute_gemm/rmsnorm.py` 等 |
| 量化 | MXFP8/MXFP4/NVFP4 操作数容器与量化器 | `cute_gemm/blockscaled/` |
| 其他 | rotary、hadamard、complex64、AllGather+GEMM | `cute_gemm/` 各域模块 |

## 文档地图

文档分三类:**架构**(系统分层与组合)、**模块**(单个子系统的机制参考,
与 architecture.md「九层职责表」的模块一一对应)、**笔记**(硬件/接口的
硬性限制与实测数据;行内以"约束/依据"标注原类别)。
写法约定:结论前置、代码路径可点、不收开发记录。

### 架构

| 文档 | 回答什么 |
|---|---|
| [architecture.md](architecture.md) 架构参考 | 系统分几层、入口分几类、一次调用怎么走、绑定与身份怎么运作;「九层职责表」末列是全部模块文档的索引 |

### 模块(modules/,按调用链自顶向下)

| 文档 | 回答什么 |
|---|---|
| [mlp_activation_recompute.md](modules/mlp_activation_recompute.md) MLP 激活重计算 | 激活怎么重算、为什么不能用 torch.compile SAC |
| [gemm_variant_api.md](modules/gemm_variant_api.md) 变体 API | 变体入口与四个变体族、typed out-op 注册面、VariantSpec 声明式样板 |
| [gemm_dispatch.md](modules/gemm_dispatch.md) 低层 dispatch | `gemm.py` 的 plan cache、`sm_to_cls` 选择、split-K 第二 kernel、launch 边界 |
| [gemm_runtime.md](modules/gemm_runtime.md) 通用宿主管线 | digest/registry 跨进程身份、`GemmEpiPlan`、`dlkernel::gemm_epi` 算子、autotune 接入点 |
| [epilogue.md](modules/epilogue.md) epilogue 子系统 | EpiOp 协议、`@gemm_epilogue` 铸类、library 工厂、Complex64 域件 |
| [operand_transform.md](modules/operand_transform.md) A 操作数变换 | `TransformA` 契约、`make_copy_block` 挂载点、W4/W8 打包解码格式 |
| [sm_device_layer.md](modules/sm_device_layer.md) SM 设备实现层 | 继承树与覆写面、四代 mainloop 差异、mixin 菱形组合、SM 选择入口 |
| [fused_grouped_gemm.md](modules/fused_grouped_gemm.md) MoE grouped GEMM | token-row gather、concat W13、dH/A′/dS 两个独立入口与调用方合同 |
| [ptr_array_grouped_gemm.md](modules/ptr_array_grouped_gemm.md) 指针数组 grouped GEMM | 多权重池单 launch:B_ptrs、tensormap 槽、窄 fork、零传输预算与性能边界 |
| [tile_scheduler.md](modules/tile_scheduler.md) tile 调度 | 四种持久化模式、三类调度器的覆写点、K 维分解 |
| [support_layer.md](modules/support_layer.md) 支撑层 | copy atom 原语、pipeline 类交换、dsl 进程级补丁、归约代数、varlen 参数族 |
| [reduction_family.md](modules/reduction_family.md) reduction 家族 | `ReductionBase` 形态、rmsnorm/softmax/cross_entropy/topk 的 host 链路 |
| [all_gather_gemm.md](modules/all_gather_gemm.md) AllGather+GEMM | `ag_args` 接入任意变体、三次流水线、blockscaled 变体 |
| [blockscaled_api.md](modules/blockscaled_api.md) Block-scaled API | block-scaled 操作数 API、fp4/6/8 格式、SF 生成 |
| [varlen_blockscaled_sf_layout.md](modules/varlen_blockscaled_sf_layout.md) varlen blockscaled SF 布局 | varlen 下 SF 的布局规则 |
| [tvm_ffi_compilation.md](modules/tvm_ffi_compilation.md) TVM-FFI 编译 | 符号形状/NamedTuple/ParamsBase 参数怎么编译与缓存 |
| [caching_and_parallel_compilation.md](modules/caching_and_parallel_compilation.md) 编译缓存与并行编译 | 三层编译栈怎么工作、--async-compile 怎么用、冷启动时间去哪了 |
| [autotuning_subsystem.md](modules/autotuning_subsystem.md) Autotune 子系统 | tune 怎么选配置、缓存怎么复用、环境变量怎么切 |
| [static_selection.md](modules/static_selection.md) TuneMode 与静态选择 | 三模式契约、F/W 双身份、解析兜底、离线 study/production/prefill |

### 笔记(notes/)

| 文档 | 回答什么 |
|---|---|
| [ragged_tma_ptr_shift_finding.md](notes/ragged_tma_ptr_shift_finding.md)【约束】 ragged TMA ptr_shift | ptr_shift 只能用于 store、不能用于 load,为什么 |
| [register_spilling_sm100.md](notes/register_spilling_sm100.md)【约束】 SM100 寄存器溢出 | 怎么诊断、setmaxregister 怎么用、epi_load 死锁坑 |
| [sm120_hardware.md](notes/sm120_hardware.md)【约束】 SM120 硬件能力 | SM120 有/没有什么(TMA multicast、TMEM、WGMMA、CLC) |
| [epilogue_transform_reorg.md](notes/epilogue_transform_reorg.md)【依据】 epilogue/transform 重组 | epilogue 与 transform 目录结构如何形成 |
| [clc_spurious_invalid_investigation.md](notes/clc_spurious_invalid_investigation.md)【依据】 CLC 假 INVALID 根因 | try_cancel 为什么会假 INVALID、修复是什么 |
| [gather_a_2cta_cluster_fence_soundness.md](notes/gather_a_2cta_cluster_fence_soundness.md)【依据】 gather_A fence 论证 | 2-CTA relay fence 为什么是正确的 |
| [hadamard_roofline_analysis.md](notes/hadamard_roofline_analysis.md)【依据】 Hadamard roofline | 分阶段 SMEM Hadamard vs FHT 什么时候谁快 |
| [global_memory_coalescing_notes.md](notes/global_memory_coalescing_notes.md)【依据】 全局内存合并访问 | H100 上合并访问的实测行为 |
| [gpu_pipe_microbench_notes.md](notes/gpu_pipe_microbench_notes.md)【依据】 GPU pipe 微基准 | SMEM/SHFL/REDUX 吞吐数据与复现命令 |
| [sm120_ncu_profiling.md](notes/sm120_ncu_profiling.md)【依据】 SM120 Nsight Compute 指南 | 怎么在 RTX 50 上采集与对比 kernel 时序 |

## 阅读路径

1. 读 [architecture.md](architecture.md)。先看分层图与入口分类,建立九层地图。
2. 按其"阅读顺序"表逐层读源码。每一步给出文件、行号与理解目标。
3. 按模块查 modules/ 文档(顺序即调用链自顶向下)。硬件行为存疑时查笔记与 AGENTS.md 的 Measured facts 节。
