# 支撑层:设备代数、host 基础设施与 DSL 基础件

支撑层是 GEMM 与 reduction 两个 kernel 家族之下共用的部件,分三类(architecture.md:281):纯设备代数、host 基础设施、DSL 基础件;另有一个跨家族共用的 `pipeline.py`。本文按文件给出原语清单、交换机制与消费关系。

约定:`路径:行号` 相对仓库根(模块内文件省略 cute_gemm/ 前缀)。对照 2026-10-13 工作区源码核实,行号随代码演进漂移。标【推断】的是分析结论,其余为源码可见事实。

## 三分总览

| 类别 | 文件 | 内容 | 主要消费者 |
|---|---|---|---|
| 纯设备代数 | `copy_utils.py`(1840 行) | copy atom / tiled copy / TMA / TMEM / cp.async / gather 原语 | GEMM 主循环与 epilogue;reduction 家族的 g2s 装载 |
| 纯设备代数 | `layout_utils.py`(393 行) | 布局代数与寄存器重排 | GEMM epilogue、vector 广播分区 |
| 纯设备代数 | `reduce.py`(480 行) | warp/block/cluster 三层归约 | rmsnorm、softmax、cross_entropy、topk、epilogue 归约 op |
| host 基础设施 | `cache/`、`autotuner.py` | 编译缓存与扫参框架 | 见 [caching_and_parallel_compilation.md](./caching_and_parallel_compilation.md)、[autotuning_subsystem.md](./autotuning_subsystem.md) |
| host 基础设施 | `compile_utils.py`(58 行) | 无张量编译用的 fake 张量工厂 | 全部 `_compile_*` 函数 |
| DSL 基础件 | `dsl/`(6 个模块) | 进程级 CuTe 补丁、`cute_op` 算子封装 | 导入即生效;`cute_op` 被各 host 入口使用 |
| DSL 基础件 | `cute_dsl_utils.py`(281 行) | dtype 映射、设备查询、`mlir_namedtuple` | 所有带 NamedTuple 参数的模块 |
| 跨家族共用 | `pipeline.py`(811 行,13 个类) | 包装并热替换 cutlass 原始 pipeline | GEMM 全部流水线;rmsnorm 反向 |

## copy_utils.py:拷贝原语分类

八个功能组:

| 组 | 代表原语(定义行) | 用途 |
|---|---|---|
| tiled copy 构造 | `get_copy_atom` :308、`tiled_copy_1d` :332、`tiled_copy_2d` :364、`vectorized_thread_partition` :343 | 从 dtype/线程数/向量宽推导 tiled copy;`tiled_copy_2d` 是 reduction 家族的标准入口(reduction_base.py:49) |
| 通用执行 | `copy` :317、`cvt_copy` :121、`cvt_copy_pair_xor_sts32` :170、`predicate_k` :384 | 带可选转换与 K 尾谓词的拷贝;pair_xor 是 r2s 的 32 位 XOR 配对版 |
| smem atom 工厂 | `sm90_get_smem_load_op` :561、`get_smem_store_atom` :593、`get_smem_load_atom` :617、`get_smem_store_C` :641、`get_smem_load_C` :675、`epilog_smem_copy_atom` :701、`get_smem_store_epi` :720、`get_smem_store_A` :750、`get_smem_load_A` :769 | 按 dtype/架构选 LDS/STS/STSM/LDSM atom;A 侧两个工厂目前只有 `spec/` 原型消费 |
| TMA | `get_tma_desc_addr` :908、`tma_get_copy_fn` :1174、`tma_get_block_copy_fn` :1223、`tma_gather4_load` :929、`tma_producer_copy_fn` :1452、`chain_tma_producer_copy_fns` :1464 | 把 TMA atom 包成 `(src_idx, dst_idx)` 闭包;block 版走 block_copy;producer 版内嵌 pipeline arrive |
| TMEM(SM100) | `tmem_store_atom_from_load_atom` :50、`load_t2r` :287、`s2t_get_copy_fn` :1288、`tmem_reg_frag` :1340、`coord_frag` :1366、`r2s_partition_from_t2r` :1375、`s2r_partition_from_t2r` :1412 | tcgen05 TMEM↔寄存器:由 load atom 反推 store atom(:55-66 对齐 C++ `tmem_load_to_store`) |
| cp.async | `cpasync_bulk_get_copy_fn` :1036、`cpasync_bulk_get_store_or_add_fn` :1111、`cpasync_bulk_s2g` :850、`cpasync_reduce_bulk_add_f32` :888 | smem→gmem 批量拷贝与原子累加 |
| gather | `gather_m_get_copy_fn` :1477、`gather_k_get_copy_fn` :1545、`gather_m_get_tma_copy_fn` :1655、`gather_k_get_tma_copy_fn` :1696 | 按 `A_idx` 逐行/逐块 gather,带 gmem 预取闭包 |
| varlen/TMA 形状 | `create_ragged_tensor_for_tma` :458、`offset_ragged_tensor` :492 | 为 ragged 维补 1-2 个额外维,让 TMA 描述子越界安全(varlen_utils.py:132-153 消费) |
| gmem store / 寄存器 | `store` :1766、`store_v2` :1791、`store_v4` :1817、`load_s2r` :255、`load_s2r_retile` :269、`contiguous` :262、`sr_cvt_copy` :220 | 向量化 gmem 写、smem→寄存器装载、位置无关 swizzle 视图(`as_position_independent_swizzle_tensor` :532、`swizzle_ptr` :526) |

### 消费子集

grep 实测(调用点计数):

| 家族 | 文件 | 用到的子集 |
|---|---|---|
| GEMM | gemm_sm90.py(9 处)、gemm_sm100.py(16 处) | TMA 闭包(`tma_get_block_copy_fn`、`tma_producer_copy_fn`、`chain_tma_producer_copy_fns`)、gather 四件套、`s2t_get_copy_fn`、`sm90_get_smem_load_op`、`tiled_copy_1d`、`create_ragged_tensor_for_tma` |
| GEMM | gemm_base.py(5 个不同符号) | `tma_get_copy_fn`、`cvt_copy_pair_xor_sts32`、`store`、`vectorized_thread_partition`、`create_ragged_tensor_for_tma` |
| reduction | rmsnorm.py / softmax.py(各 4 处) | `copy`、`predicate_k`(g2s 装载 + K 尾谓词) |
| reduction | cross_entropy.py(5 处)、reduction_base.py:49 | 上者加 `tiled_copy_2d` |
| 旁支 | varlen_utils.py、rotary.py、transform/hadamard.py、sm90_utils.py | 各取 1-2 个符号(ragged 张量、通用拷贝) |

【推断】分组即依赖边界:GEMM 吃 TMA/gather/TMEM 这组重原语,reduction 家族只吃 tiled copy + 谓词执行这组轻原语;两组几乎不重叠,所以 copy_utils 的改动风险集中在对应家族。

## layout_utils.py:布局代数

两块内容。布局变换:`transpose_view` :10、`select` :17、`concat_to_interleave` :21(gate/up 交错)、`expand` :34、`concat_layout` :197、`convert_layout_acc_mn` :204(MMA 累加器布局 ↔ (M,N) 视图)、`convert_layout_acc_frgA` :244(back-to-back GEMM)、`convert_layout_zero_stride` :290、SF 布局 `tile_atom_to_shape_SF_strided` :356(对齐 blockscaled 布局但尊重实际 stride)。寄存器重排:`permute_gated_Cregs_b16` :77 与 f32 版 :41、`permute_Cregs_b32_for_stsm` :110 / `for_ldsm` :153——4 线程内 prmt+shuffle,让 C 寄存器无 bank conflict 地走 STSM/LDSM。vector 分区辅助:`mma_partition_C_vec` :316、`mma_partition_A_vec` :326、`copy_partition_S_vec` :336、`copy_partition_D_vec` :346。

## reduce.py:三层归约

归约分三层,调用方通常只碰最高层 `row_reduce`:

| 层 | 函数 | 机制 |
|---|---|---|
| warp 内 | `warp_reduce` :22 | 硬件有 `redux.sync` 时单条指令(Int32 全架构;Float32 min/max 仅 SM100 家族,:30-34),否则 shuffle-butterfly;`abs`/`nan` 折进 REDUX.MAXABS(:36-40) |
| CTA 内 | `block_reduce` :93 | 每 warp lane0 写 smem buffer(`(warps/row, warps_per_row)`),`barrier` 后一个 warp 再归约(:100-110) |
| cluster 内 | `cluster_reduce` :114 | warp0 的单线程按总字节数武装事务 mbarrier(:138-142),每 warp 前 `cluster_n` 个 lane `store_shared_remote` STAS 到各 peer(:143-150),`mbarrier_wait` 后读回归约(:151-160) |

`row_reduce`(:202)是标准入口:TensorSSA 先按 `reduction_profile=0` 线程内归约(:223),`warp_reduce` 以 `threads_in_group=min(threads_per_row, 32)`(:235-240);仅当 `warps_per_row > 1 或 cluster_n > 1` 才进 `block_or_cluster_reduce`(:165,:251)。多值 tuple(如 (max, sum))共用一个 mbarrier、一次 cluster 往返(:125-128),sync 槽与 buffer 槽解耦。两个变体:`online_softmax_reduce`(:271)融合 max+sumexp,跨 warp 状态打包成单个 Int64(`f32x2_to_i64`,:310);`swap_shuffle_reduce`(:371)是分布式所有权归约——结果分散在各 lane 而非广播,比 butterfly 少 `size·log(lanes)` 次 shuffle,被 epilogue 的 OnlineLSE 消费(epilogue/ops.py:44)。

消费者:rmsnorm(前向 :265 起,反向 :1096)、softmax、cross_entropy、topk、rms_final_reduce、epilogue/ops、epilogue/quantize_out。

## pipeline.py:类交换与一台自研流水线

13 个类分三部分:状态(`PipelineStateWAdvance` :111,给 cutlass 的 `PipelineState` 补 `advance_iters` :113)、mixin(`_PipelineIndexPhaseMixin` :143,提供按 index/phase 直呼的 `_w_index_phase` 方法,:147-180)、11 个 pipeline/Barrier 包装类。

类交换机制:`_override_create`(:27)生成一个静态工厂——先调 cutlass 原类的 `create`,再 `object.__setattr__(obj, "__class__", child_cls)` 换类(frozen dataclass 不能直接赋 `__class__`,:33-34)。`NamedBarrier` 在 :213 用它热替换;其余类手写 `create`(:241、:294、:333),在换类之外再注入 `elect_one_*` 配置字段。效果:kernel 代码里写 `PipelineTmaAsync.create(...)`,拿到的是 cute_gemm 版本,而类型检查与 DSL 协议仍对得上原类。另有两个手工 mbarrier 原语补 cutlass 缺口:`mbarrier_arrive_release_cluster`(:41,release-arrive 拆成 fence + relaxed arrive)与 `mbarrier_acquire_cluster`(:64,把已完成的 wait 升级为 cluster-scope acquire)。

| 类 | 基类 | 增量 | 主要调用点 |
|---|---|---|---|
| `PipelineAsync` :220 | PipelineAsyncOg | producer_commit/consumer_release 可选 elect_one | 调度流水线 gemm_sm90.py:1964;SM100 acc 流水线 gemm_sm100.py:2633 |
| `PipelineCpAsync` :289 | PipelineCpAsyncOg | elect_one release | gemm_sm100.py:2670 |
| `PipelineTmaAsync` :326 | PipelineTmaAsyncOg | `extra_tx_count` 增量事务 + elect_one release | AB/epi 流水线 gemm_base.py:1565-1566(非 gather);rmsnorm :910 |
| `PipelineTmaCpAsync` :720 | PipelineTmaAsyncOg | TMA+cp.async 混合 producer | gather-A 的 AB 流水线 gemm_base.py:1565-1566 |
| `PipelineTmaUmma` :553 / `PipelineUmmaAsync` :606 / `PipelineAsyncUmma` :663 | 同名 Og | UMMA(tcgen05)变体 | SM100:gemm_sm100.py:2544、:2584 |
| `PipelineTmaCpAsyncUmma` :766 | PipelineTmaUmmaOg | UMMA + cp.async | SM100 gather-A:gemm_sm100.py:2553 |
| `PipelineTmaStore` :528 | PipelineTmaStoreOg | read 模式开关 | TMA store 流水线 gemm_base.py:1607 |
| `PipelineStasAsync` :387 | PipelineAsyncOg(直接) | 全新类,见下 | rmsnorm 反向 :873;reduction_base.py:74 注释 |

`PipelineStasAsync` 是唯一的纯自研流水线:cutlass 没有对称 all-to-all 的对应物(:416-418 注释自比"事务武装移进数据路径的 PipelineTmaAsync")。full barrier 是事务 barrier,由数据路径(`reduce.cluster_reduce` 的 `arrive_and_expect_tx` + STAS)武装,`producer_commit` 是 no-op(:398-402);一组 (full, empty) barrier 可守护多个 buffer 槽(:404-409);empty barrier 按 `num_warps × cluster_n` 次到达初始化(:441-444)。

## varlen_utils.py:参数族从签名到设备侧

参数族四张量:`VarlenArguments`(:16)= `mCuSeqlensM` / `mCuSeqlensK` / `mAIdx` / `mCuTilesM`,对应 `gemm` 签名的 `cu_seqlens_m/k`、`A_idx`(gemm_interface.py:922-924;`batch_idx_permute` 不属于它,走调度器参数)。流经三站:

1. host 组包:`make_varlen_args`(gemm_tvm_ffi_utils.py:311)把 torch 张量装进 NamedTuple;`make_fake_varlen_args`(:334)为无 GPU 编译造 sym 张量;调用点 gemm.py:653(真)/gemm.py:183(fake)。
2. 参数化:`VarlenManager.to_underlying_arguments`(:70)进 `Params`(:30),`varlen_m` 与 `varlen_k` 互斥有断言(:72-74)。
3. 设备构造:kernel 里 `VarlenManager.create(params, len_m_static, len_k_static, len_n_static)`(gemm_sm90.py:1013、gemm_sm100.py:1256),静态长度由操作数形状补齐(:1016-1026);布尔标志 `varlen_m`/`varlen_k`/`gather_A` 在构造期定为 const(:64-66)。

设备侧方法族:

| 方法 | 行 | 语义 |
|---|---|---|
| `len_m` / `len_k` / `len_n` | :95 / :101 / :107 | 当前 batch 的序列长度;非 varlen 回落静态长度;N 永不变长 |
| `tile_m_offset` / `len_m_tiles` | :112 / :116 | `cu_tiles_m` 前缀:M-fold reduce sink 的每序列 tile 索引(:20-24 注释) |
| `offset_batch_A` | :120 | varlen 时取本 batch 的 A 切片:rank-2 走 `domain_offset`,rank≥3 走 ragged TMA 张量(:124-154) |
| `offset_batch_AIdx` | :159 | gather 索引按 cu_seqlens 偏移 |
| `offset_batch_SFA` / `_SFB` | :169 / :190 | SF 缓冲按 128 对齐 tile 偏移(`cu_seqlens[b]//128 + b`,:180-186) |
| `offset_batch_B` / `_epi` | :206 / :227 | varlen_k 的 B 切片 / varlen_m 的输出切片 |

同一 `VarlenManager` 实例传进 epilogue 钩子(`epi_begin` 的形参,gemm_base.py:1206),所以 epilogue op 的分组列统计与 mainloop 用同一套偏移。调度器另走一条路:直接消费 `cu_seqlens_m`,见 [tile_scheduler.md](./tile_scheduler.md)。

## dsl/:进程级补丁与 import 顺序

`dsl/` 六个模块,前三个是 import 即生效的进程级 monkey-patch,由 `dsl/__init__.py:5-7` 拉起:

| 模块 | 补丁内容 |
|---|---|
| `cute_tensor_indexing.py` | 教 CuTe 张量 `A[i, :]` 与 `...`(展开为正确个数的 `None`,:5-6);原方法保留为 `_dlkernel_original_getitem/setitem`(:9-10) |
| `cute_tensor.py` | `tensor.to(dtype)` / `.clone()` / `.contiguous()`(docstring :2-22) |
| `mixed_constexpr_if.py` | lowering 前把 `if const_expr(S) and D` 改写成嵌套分支,使短路求值在 DSL 里与 Python 语义一致 |
| `smem_struct.py` | `Reserved[...]` 注解 + `@partitioned_struct`:单个 `cute.struct` 无法混排 RESERVED/USER 分区,装饰器在 trace 时拆成两个 struct(:6-10) |
| `torch_library_op.py` | `cute_op`(:29):`torch.library.custom_op` + 注册 no-op fake(`@op.register_fake` :54),省掉每个 op 手写 `_*_fake` |
| `cute_dsl_ptxas.py` | 系统替换 ptxas;由环境变量 `CUTE_DSL_PTXAS_PATH` 门控,在 `cute_gemm/__init__.py:7-13` 于任何 CuTeDSL 实例化之前打补丁 |

import 顺序约束:三个 monkey-patch 必须先于任何 CuTe DSL 使用者执行。`cute_gemm/__init__.py:5` 把 `import cute_gemm.dsl` 放在全部子模块 import 之前(:17-19 注释明示),而 Python 导入任何 `cute_gemm.xxx` 子模块都会先执行包 `__init__`,所以库内库外入口都拿到补丁后的语义。【推断】直接以文件路径单独执行某模块(绕过包导入)会拿到未打补丁的 CuTe 张量类,`:` 索引会退回报错。

## cute_dsl_utils 与 compile_utils

`cute_dsl_utils.py` 四件事:

- 进程级补丁第三处:TVM-FFI 参数转换器(:34-53),`cutlass.Constexpr[T]` 字段发射为 `ConstNone`(:39-41),带 `_fields` 的 tuple 重定向到自身类型注解(:44-50)。
- dtype 映射 `torch2cute_dtype_map`(:56),含 packed fp4 的 dlpack 逻辑 K(:58-59)。
- 设备查询:`get_device_multiprocessor_count` :74、`get_max_active_clusters` :128、`get_device_capacity` :178(`DLKERNEL_ARCH` 可覆盖)、`get_compile_target_capacity` :196(编译目标与调度架构分离,CI 代理腿依赖它)。
- SM 预算(`set_num_sms` :89 / `get_num_sms` :111,DeepGEMM DeviceRuntime 语义):进程级 cap,`0` 恢复默认;有设备时校验 `0 < n ≤ 物理 SM 数`。生效点两处——启发式 `sm_count_for` 的波数分母、持久 kernel 的 launch grid(`get_max_active_clusters` 按整 cluster 截断;occupancy 查询在 `_max_active_clusters_hw` :117 单独缓存,cap 不进缓存键,恢复默认后立即生效)。
- DSL 协议胶水:`mlir_namedtuple`(:248)给 NamedTuple 补全 JitArgument 四协议(:263-266),`ParamsBase`(:270)给 dataclass 同款;`TileSchedulerArguments`、`VarlenArguments` 等全部参数类型靠它们进出 jit 边界。

`compile_utils.py` 是无张量编译的 fake 工厂:`make_fake_tensor`(:8,sym stride + divisibility 折算对齐,:17-22)、`div_for_dtype`(:36,16 B 对齐换算)、`fake_batched`(:41,batch-first 约定)、`make_fake_stream`(:56)。每个 `_compile_*` 函数用它构造 fake 输入再 `cute.compile`,见 [caching_and_parallel_compilation.md](./caching_and_parallel_compilation.md)。

## 孤立与超前件

`tensormap_manager.py` 零引用(grep 全库无消费者);`blockscaled/quantize_utils.py` 自述服务于"fused RMSNorm/LayerNorm quantized forward"(:6-8),该 kernel 尚不存在,现有消费者只有 epilogue 的 `quantize_out.py:41`。`spec/` 原型零消费者。详见 architecture.md:283。

## 参考

- [architecture.md](../architecture.md)「reduction 家族与支撑层」(:277)
- [reduction_family.md](./reduction_family.md)——reduce.py 三层归约在各个 reduction kernel 中的用法
- [tile_scheduler.md](./tile_scheduler.md)——varlen 参数族的调度器侧消费
- [caching_and_parallel_compilation.md](./caching_and_parallel_compilation.md)——compile_utils fake 张量的使用场景
