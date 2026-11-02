# 低层 dispatch(gemm / split_k_reduce)

`gemm.py` 是线性 GEMM 的编译与启动层:`gemm()`(`gemm.py:378`)以张量元数据组键查 `_gemm_plan_cache`(`:261`),命中只剩换指针 launch;未命中走校验、major/dtype 推导、`@jit_cache` 编译(`:49`)。SEPARATE split-K 在 GEMM 之后接 `split_k_reduce.py` 的第二 kernel 做定序求和。launch 出口只有一处:`launch_gemm`(`gemm_tvm_ffi_utils.py:497`),TVM-FFI 从真实张量抽指针并按编译期签名校验。

约定:`路径:行号` 相对仓库根。对照 2026-10-13 工作区源码核实,行号随代码演进漂移。标【推断】的是分析结论,其余为源码可见事实。

## 服务对象

| 调用方 | 路径 |
|---|---|
| 线性族变体 | `gemm_interface.py:377`(`gemm_tuned`)→ `:513`(`_gemm_execute`)→ `:662`(`gemm_dispatch`,即本层 `gemm`);eager 快路径旁路直达 `run_gemm_plan`(`gemm_interface.py:633`) |
| 对称变体 | `gemm_symmetric.py` 持有同构的另一份 plan cache(`:217`)与 `run_gemm_symmetric_plan`(`:312`) |
| epilogue mod | 不走本层。`EpiMod.gemm` 经 `gemm_runtime/host.py` 的 `build_gemm_epi_plan`/`run_gemm_epi_plan`(`:314`/`:509`) |

epilogue mod 旁路本层的原因:那条管线由 EpiOp schema 驱动,plan 里装的是 `call_ops`/`arg_template`,编译语义由 fn digest 决定。线性族的 epilogue 是手写 mixin `GemmDefaultEpiMixin`(`gemm_default_epi.py:70`),编译签名由 `gemm.py` 的平铺参数列表直接决定,无 fn、无 digest、无 EpiOp schema。这个分叉是记录在案的范围决策,不是遗留事故:把 `gemm_interface.py` 的手写 custom op 迁上 `dlkernel::gemm_epi`、统一 eager 旁路的三种写法,列在 [../notes/epilogue_transform_reorg.md](../notes/epilogue_transform_reorg.md) 的后续战役(`:278-281`);镜像 plan cache 模式四处并存的代价记在 [../architecture.md](../architecture.md)「潜在问题」第 4 条。

两条 host 管线的对照(同一个模式,两份实现):

| 维度 | 本层(`gemm.py`) | 通用宿主管线(`gemm_runtime/host.py`) |
|---|---|---|
| 服务对象 | 线性族变体 | `@gemm_epilogue` fn 与 epilogue mod |
| plan 类型 | `_GemmPlan`(`gemm.py:225`,19 字段) | `GemmEpiPlan`(`host.py:267`,18 字段) |
| plan cache | `gemm.py:261`,键平铺 | mod 自己的 `_plan_cache`,键含 op 的 `host_arg_key` |
| 编译入口 | `_compile_gemm`(`gemm.py:49`,`@jit_cache`) | `_compile_gemm_epi`(`host.py:84`,`@jit_cache`) |
| epilogue 形态 | `GemmDefaultEpiMixin` 手写钩子 | EpiOp 组合,`call_ops` 逐 op 编排 |
| SM 选择 | `sm_to_cls`(`gemm.py:93`) | `GemmBase.select_gemm_class`(`gemm_base.py:167`) |
| 身份机制 | 无 digest(jit 键即语义) | `semantic_digest`/`GemmClassRef`(`identity.py`) |
| launch | `launch_gemm`(`gemm_tvm_ffi_utils.py:497`) | 同一函数 |
| torch 算子面 | 14 个变体 out-op | `dlkernel::gemm_epi` 一个(`torch_op.py:93`) |

## gemm() 与 plan cache 键

`gemm()` 签名覆盖全部公共参数族:varlen(`cu_seqlens_m/k`)、gather(`A_idx`)、blockscaled(`SFA/SFB` + 格式名)、量化输出(`SFD`/`SFDCol`)、split-K、AllGather(`ag_args`,`gemm.py:442`),返回 `_GemmPlan`(`:443`)。返回 plan 是刻意的:外层(变体 API 的 iface plan)捕获它后直接调 `run_gemm_plan`,warm 调用只付一次键(`gemm_tvm_ffi_utils.py:390-394` 设计注释)。

键在 `gemm.py:459-501`,两部分:

- 每个张量参数的 `tensor_key`(`gemm_tvm_ffi_utils.py:428`,dtype + shape + stride,不含指针);
- 标量旋钮与存在性布尔:tile/cluster/pingpong/persistent、`alpha_mode`/`beta_mode`/`sr_seed_mode`(`scalar_mode`,`gemm_tvm_ffi_utils.py:434`:0 缺席 / 1 host 常量 / 2 设备指针)、`A_idx is not None` 等。

键的正确性论证在 `:450-453` 注释:shapes/strides 蕴含 majors、校验断言与 fp4/SF 形状检查的结论,键命中即"换数据指针重放一次已校验调用"。标量值不进键,mode 进键——mode 选择编译出的 epilogue 形态,值是逐调用的(`gemm_tvm_ffi_utils.py:386-388`)。

`_gemm_plan_cache`(`:261`)是进程级 dict,随 (shape, stride, dtype, flag) 组合增长,无淘汰;编译去重在下一层由 `_compile_gemm` 的 `@jit_cache`(`:48`)承担(`:258-260` 注释)。查询与回填在 `:502`/:546,launch 在 `:547`,返回 plan 在 `:570`。

`_GemmPlan`(`:225`)19 个字段:

| 组 | 字段 | 说明 |
|---|---|---|
| 编译产物 | `compiled_fn` | TVM-FFI 函数,唯一启动入口 |
| 架构路由 | `is_sm100_family` | SM100/110 用 2-CTA MMA,split-K buffer 计算用它(`:306`) |
| 架构路由 | `use_d_as_c` | SM8x 把 add_to_output 降为 C=D(`:864-869`),launch 前 `C = D`(`:604-605`) |
| 标量 mode | `alpha_mode`/`beta_mode`/`sr_seed_mode`/`sfd_norm_const_mode` | 逐调用 `scalar_arg` 的依据(`:633`/:639/:647) |
| 静态模板 | `epi_static`/`scheduler_static` | 无逐调用值时缓存的整体参数模板 |
| 调度 | `max_active_clusters`/`max_swizzle_size`/`scheduler_uses_semaphore` | 持久调度的三个预计算量 |
| 外层启动 | `split_k`/`split_k_mode`/`add_to_output`/`tile_M`/`tile_N`/`cluster_M`/`cluster_N` | 外层持有 plan 即可 launch,无需再查本层键(`:247-248` 注释) |

静态模板有严格条件:epilogue 全缺席(标量 mode 全 0、无 bias 向量、split-K 为 1 或 SEPARATE、无 SFD)才缓存 `epi_static`(`:970-989`);scheduler 模板要求无 semaphore 消费、无 permute、无 AG(`:990-992`)。任一条件不满足则每次调用重建 `EpilogueArguments`(`:631-649`)——它携带指针,不可缓存(`gemm_tvm_ffi_utils.py:412-413`)。

plan 不可变(`:228-233` docstring),跨调用复用安全。split-K workspace 与 semaphore 是逐调用分配,永不进 plan(`:606-609` 注释);流正确性由 caching allocator 的 stream 感知免费获得(`gemm_tvm_ffi_utils.py:414-417`)。

`run_gemm_plan`(`:573`)的契约写在 docstring(`:598-603`):只收逐调用指针与标量;张量必须匹配建 plan 的元数据——`gemm()` 用键保证,外层持有 plan 时由外层自己的键保证。

tile-count semaphore 的生命周期贯穿三层:变体层按 `(dynamic_scheduler, capacity==9)` 决定是否分配(`gemm_interface.py:606-610`);存在性进 plan 键(`gemm.py:472`);launch 时只有 `scheduler_uses_semaphore`(SM8x/SM90 动态持久,`:964`)才转发,SM100+ 的 CLC 调度不需要它。

## 冷路径 _build_gemm_plan

`_build_gemm_plan`(`:671`)四步:

| 步 | 行 | 内容 |
|---|---|---|
| 校验 | `:714-889` | 组合约束:varlen_m/k 互斥(`:719`)、gather_A 要求 varlen 且 cluster_N=1(`:720-722`)、AG 要求稠密持久且无 split-K(`:723-726`)、split-K 要求稠密且无随机舍入(`:737-744`)、capacity 白名单(`:753`)、blockscaled SF 布局(`:778-789`)、SFD/SFDCol 覆盖检查(`:805-852`) |
| 推导 | `:891-896` | `get_majors`/`get_dtypes`(`gemm_tvm_ffi_utils.py:237`/`:245`);b_kn 下 B 的 major 按逻辑 (n, k) 重标(`:892-895`) |
| SEPARATE 重定向 | `:898-910` | `d_dtype=Float32`,c/alpha/beta/bias 全清零(`:906-910`),编译签名只见 f32 D 与裸 epilogue |
| 编译与预计算 | `:915-963` | `_compile_gemm`;`max_active_clusters`(`:960-963`);semaphore 消费判定 `:964`(仅 SM8x/SM90 动态持久) |

## 编译与 SM 版本选择

`_compile_gemm`(`:49`,挂 `@jit_cache`:48):

- **sm_to_cls 静态映射**(`:93-99`):`8→GemmDefaultSm80`、`9→GemmDefaultSm90`、`10/11→GemmDefaultSm100`、`12→GemmDefaultSm120`,`:100` 查表。这些类是 `GemmDefaultEpiMixin` 混入各 SM 基类的组合类。未知 major 在 `:753` 的 assert 已挡住;查表 KeyError 不会成为静默回退。epilogue 族的同构映射在 `GemmBase.select_gemm_class`(见 [../architecture.md](../architecture.md)「目标硬件差异」)。
- **fake 张量**:`make_fake_gemm_tensors`(`gemm_tvm_ffi_utils.py:508`)以共享 `sym_int` 造 mA/mB/mD/mC(`gemm.py:101-117`),调用方形状即 FFI 签名:`batched=False` 造 2D fake(kernel 在 trace 时补平凡 batch 维)、`b_kn` 造 (k, n) 朝向(trace 时转置,docstring `:537-549`);swap_ab 直接造交换后的维度(`:618-630`)。SEPARATE 时 mD 单独造 (l·split_k) 批(`gemm.py:118-123`);epilogue 标量按 mode 造 None/常量/零地址指针(`:125-131`);scheduler/varlen args 用 fake 版(`:176-183`)。
- **编译**:`compile_gemm_kernel`(`gemm_tvm_ffi_utils.py:668`)实例化 kernel 类,按 capacity 传构造参数(SM8x persistent/warps、SM90/120 pingpong+split_k、SM100 CLC+blockscaled,`:704-746`),`cute.compile(..., options="--enable-tvm-ffi")`(`:770-782`)。fake/符号张量机制见 [./tvm_ffi_compilation.md](./tvm_ffi_compilation.md)。

## SEPARATE split-K:两 kernel 协议

三种模式(`gemm_config.py:26`/`:29`/`:33`):

| 模式 | 部分和提交 | 最终化 | 逐位确定 |
|---|---|---|---|
| SERIAL | turnstile 按 split 序 | 最后一个 split 在 GEMM 内 | 是 |
| PARALLEL | 到达序 red.add | arrival counter 满后在 GEMM 内 | 否 |
| SEPARATE | 各 split 写独立 workspace 切片 | 第二 kernel | 是 |

共同原则:完整 epilogue(alpha、β·C、bias)恰好执行一次,由持有完成和的一方执行;非最终化的 split 只出裸 f32 partials(`gemm.py:264-267` 注释)。

**SERIAL/PARALLEL**:semaphore + workspace 经 epilogue args 进 kernel。buffer 由 `_split_k_buffers`(`:292`)每次 launch 现分配(`:619-627`):每输出 tile 一个 Int32 完成标志,f32 workspace 每 tile 一条平坦片段带;tile 数向 cluster 倍数取整,SM100 2-CTA 把 cta_tile_m 减半(`:305-311`)。

**SEPARATE**(`staged_split_k`,`:902`):

1. 编译签名重定向(冷路径 `:898-910`):GEMM 不带任何 epilogue 操作数。
2. 每次 launch 造 workspace(`:616` → `_staged_split_k_workspace`:270):batch 维是合并的 l·split_k,contiguous 维 pad 到 4 个 f32(TMA 16 字节,`:281-289`);GEMM 的 D/C 槽位换成 workspace 与 None(`:617`)。
3. GEMM launch 后跑归约(`:657-668` → `_reduce_staged_split_k`:330):m-major 输出转置视图并交换 row/col 向量角色(`:336-341`);C 的 majorness 与 D 不同时物化一次(`:342-346`,GEMM epilogue 本身允许 c_major 独立)。
4. `split_k_reduce`(`split_k_reduce.py:356`)经 `@cute_op`(`dlkernel::split_k_reduce_out`,`:297`)进 torch 算子面,torch.compile 看到的是注册算子(`:373-374` 注释);`_compile_split_k_reduce`(`:236`)按 (out_dtype, split_k, vec_elems, 标量 mode…) 编译。
5. `SplitKReduce.kernel`(`split_k_reduce.py:132`,host 入口 `:35`):每线程一行 vec_elems 连续列,列维按向量谓词、行维 if 守卫(`:156-162`);按 split 升序求和(`:206-208`),顺序固定所以逐位可复现(`:6-8`);完整 linear epilogue 恰好一次,复用 `apply_linear_epilogue`(`:32` import,`:225` 调用)——与 GEMM mixin 的 in-GEMM 最终化是同一份数学(`:30-32` 注释)。`vec_elems = gcd(N, 4)`(`:319`),C 非连续时降为 1(`:320-321`)。

【推断】SEPARATE 的定位:GEMM 签名极简(f32 D、零 epilogue 参数),最终化挪进一个逐元素小 kernel,归约侧拿到固定求和顺序的确定性;代价是每次 launch 一次 l·split_k 倍的 f32 workspace 分配。blockscaled 与 SEPARATE 互斥(`:741-744`)。

## launch 边界(TVM-FFI 抽指针)

`run_gemm_plan`(`:573`)组装逐调用参数:

- epi args:静态模板或现造(`:630-649`);split-K semaphore/workspace permute 成 kernel 朝向 (ntile_m, ntile_n, L)(`:640-645`)。
- scheduler args:`plan_scheduler_args`(`gemm_tvm_ffi_utils.py:471`)。静态模板命中直接返回(`:486-487`);semaphore 只在 `scheduler_uses_semaphore` 为真时转发(`:491`)——SM100+ 用 CLC,编译签名里是 None,调用方传了也丢弃(`:473-477` 注释)。AG 分片几何每次 launch 校验(`:483-485`)。
- varlen args(`:653`)。

`launch_gemm`(`gemm_tvm_ffi_utils.py:497`)是唯一 launch 出口:`plan.compiled_fn(A, B, D, C, epi_args, scheduler_args, varlen_args, SFA, SFB)`(`:505`)。签名各架构统一,尾部 (SFA, SFB) 恒传(None 除非 blockscaled,`:498-500`);blockscaled 的 U4/U6 TMA unpack 对齐在每次 launch 前检查(`:58-67`,缓存 plan 的键不含指针)。TVM-FFI 从真实张量抽数据指针、按编译期 arg spec 校验 shape/stride/对齐——机制见 [./tvm_ffi_compilation.md](./tvm_ffi_compilation.md)。

三层绑定(compile / plan / call)的设计注释在 `gemm_tvm_ffi_utils.py:357-425`:每个事实在最早能知道它的层处理;热路径 = 一次键 + dict 命中 + `run_*_plan`,其中约 8us 是 FFI + cudaLaunch 地板,其余是 1-2us 级条目(`:401-404`)。执行助手每调用重推导路由标志(b_kn/dense_2d/swap,约 0.5us)而非存进 plan,是记录在案的取舍:冷热两路共用同一路由体,永不漂移(`:406-410`)。

## 速查

| 问题 | 答案 |
|---|---|
| 谁建 plan | `_build_gemm_plan`(`gemm.py:671`),仅键未命中时 |
| 谁编译 | `_compile_gemm`(`:49`),`@jit_cache` 去重,`sm_to_cls`:93 选类 |
| 谁启动 | `launch_gemm`(`gemm_tvm_ffi_utils.py:497`),唯一出口 |
| 谁做 SEPARATE 归约 | `split_k_reduce.py` 的 `SplitKReduce.kernel`(`:132`),`@cute_op` 算子 `:297` |
| plan 里有什么 | `gemm.py:225` 的 19 字段;指针永不进 plan |
| 键里有什么 | `gemm.py:459-501`;元数据 + 旋钮 + 标量 mode,无指针无标量值 |
