# epilogue 子系统

GEMM 的非 mainloop 数学——激活、norm、梯度、归约统计、量化输出、位置编码——全部经 `cute_gemm/epilogue/` 挂载。四层分工:ops.py 定义 EpiOp 词表,mixin.py 把词表编排成设备钩子,frontend.py/visit.py 提供 fn 前端与铸类,library.py 与域模块给出现成 mod。本文件展开 `../architecture.md` 的「epilogue 子系统」一节;import 方向、绑定与身份机制、EpiOp 覆写矩阵见该文的对应小节。

约定:`路径:行号` 相对仓库根(模块路径省略 cute_gemm/epilogue/ 前缀)。对照 2026-10-13 工作区源码核实,行号随代码演进漂移。标【推断】的是分析结论,其余为源码可见事实。

## 四层结构与依赖方向

| 层 | 文件 | 职责 | 依赖 |
|---|---|---|---|
| op 词表 | ops.py(3088 行) | EpiOp 协议(:298):资源生命周期、值端口、host schema、cache 身份;14 个 op 类 | cute_dsl_utils、copy_utils、reduce 等支撑层,不引本包其余模块 |
| 编排 | mixin.py(381 行) | ComposableEpiMixin(:60):把类属性 `_epi_ops` 元组展开成 `epi_*` 钩子族、smem 结构体与 EpilogueParams | ops |
| fn 前端 | frontend.py(1996 行)+ visit.py(383 行) | EpiMod(:289):装饰 fn、铸 kernel 类、三层调用面;visit.py 是铸出类的设备侧循环 | ops、mixin、legality、math、visit、gemm_runtime.host |
| 库与域件 | library.py(989 行)+ rotary / scaled_exp / head_rmsnorm / quantize_out / math | 现成 mod 与参数化工厂;域件各解决一个业务域 | frontend、ops、cute_gemm.activation、blockscaled、gemm_tvm_ffi_utils |

math.py 是叶子,只依赖 cutlass:Pair(:16)、F2(:76)、F16Lanes(:114)、pexp/pexp2(:159/:166)。legality.py 也是叶子,只引 gemm_config 与 cute_dsl_utils,被 frontend 反向消费。依赖单向:library 与域模块 → frontend → visit → mixin → ops;域模块直接 import `gemm_epilogue` 装饰器写 mod(rotary.py:12)。`__init__.py` 对 frontend 符号走 PEP 562 懒加载(:34-44),kernel 类 import ops/mixin 时不必拉起 frontend(它要 import 全部 per-arch kernel 类)【推断:懒加载的动机,`__init__.py:18-21` 自述】。

## EpiOp 协议

EpiOp(ops.py:298)是协议基类:一个 op 声明一类张量资源怎么用,组合、host 管线、缓存从 schema 继承。方法分五组:

| 组 | 方法(基类定义行) | 作用 |
|---|---|---|
| 值端口 | `fn_port`:321、`fn_prepare`:323、`fn_apply`:328、`fn_sink_flush`:331 | 声明 op 如何进 fn 的逐元素数据流 |
| host 三件套 | `host_arg_key`:373、`host_fake_arg`:381、`host_call_arg`:387 | 编译键描述符 / fake 追踪参数 / 每次调用的运行时参数 |
| param 桥 | `config_key`:340、`cache_key`:354、`arg_spec_type`:391、`host_arg_form`:398、`param_fields`:404、`to_params`:409、`epi_m_major_score`:414 | cache 身份、EpilogueArguments 字段类型、args→params 自动生成 |
| smem / TMA | `smem_bytes`:419、`smem_struct_field`:423、`get_smem_tensor`:428、`tma_atoms`:433、`is_tile_load`:437、`is_tile_store`:441、`load_g2s_copy_fn`:445 | smem 预算、struct 字段、TMA atom、tile 级装载 |
| 设备生命周期 | `begin`:459、`begin_loop`:463、`load_s2r`:468、`end_loop_stage`:472、`end_loop_finish`:493、`needs_async_fence`:498、`end`:502 | 每 tile / 每 subtile 的执行协议 |

要点:

- 值端口是 fn 前端唯一的分发依据。`fn_port` 注释明言 frontend 只派发该属性,绝不 isinstance(ops.py:301-304);`_pinned_visit_kind`(frontend.py:1806)同样只读 `fn_port`。取值:`row`/`col`/`tile`/`scalar`(内建 fragment kind,装载类 op 声明)、`value`(自定义值源,`fn_prepare` 把 begin_loop 态转成稠密 fragment)、`apply`(fn 收到可调用,`y = rope(acc)`)、`sink`(fn 返回该名,前端收集成 fragment 后每 subtile 调一次 `fn_sink_flush`)、`None`(fn 前端不可用,仅手写 mixin)。可作操作数 pin 的端口集合在 `_OPERAND_PORTS`(frontend.py:1803)。
- flush 两段式:`end_loop_stage` 各 op 先把条带 stage 进互不相交的 smem,驱动器每 subtile 只发一次共享 barrier,再 `end_loop_finish` 合并写 gmem(协议注释 ops.py:483-490;实现 mixin.py:331-356)。多 sink 的 epilogue 每次 flush 只付一次 `arrive_and_wait`。
- `config_key`(:340)fail-closed:实例带静态配置却不实现该方法直接抛 `NotImplementedError`,防止两个语义不同的 epilogue 在持久 jit cache 里共用一键。`cache_key`(:354)= 模块 + 类名 + name + config_key。
- `host_arg_key` 返回 None 表示该 op 从编译产物中滤除(:374-375);`arg_spec_type(const=True)` 产 Constexpr 注解,FFI 转换器不发出运行时参数(:391-396)。
- `host_call_converter(key)` 在计划构建时选择逐次调用的转换函数。identity 参数选择
  None,热路径不调用 Python converter。rank-1 的 8/16-bit `ColVecLoad` 选择契约检查:
  stride=1、基址 16 字节对齐。真实冷调用在编译/调参前检查,每次 launch 再检查实际
  指针;FakeTensor 阶段只检查元数据。该列向量使用 16-byte cp.async 对齐窗口,
  shared 多留 16 字节、偏移消费视图,并按当前分段有效行补零。rank-2 与 FP32 保留原路径。
- `EpiContext`(ops.py:121)把 epi_tile、tiled copy、tile 坐标、varlen、barrier、tidx 打包传给设备钩子;`EpiSmemBytes`(ops.py:273)三分量(unstaged / D-stage / C-stage)逐 op 相加成 smem 预算。

### op 词表

ops.py 的 14 个类,按资源分组:

| 组 | 类(定义行) | 角色 |
|---|---|---|
| 标量与向量装载 | Scalar:518;VecLoad:604 → RowVecLoad:720 / ColVecLoad:727 | scalar / row / col 端口;标量或设备指针、每行 / 每列广播量 |
| tile 装载 | TileLoad:1195 | tile 端口;走 C 流水的 tile 级输入(如保存的 preact) |
| tile 存储 | TileStore:796 | aux 输出的整条 TMA 存储路径:r2s 拷贝、逐 op rounding、gated 半 tile、store 谓词;混合 dtype 的多个 TileStore 可组合 |
| D 存储 | DStore:1134 | D 本体的 convert 与 r2s 路径;host 管线留在 kernel,不进 `_epi_ops` |
| 向量归约 | VecReduce:1461 → ColVecReduce:1664 / RowVecReduce:1884 | sink;combine=add/max/max_abs,`scaled=True` 让积归约走单条 fma |
| 在线 LSE | OnlineLSEReduce:2614 | sink;耦合 (max, sum) 累加器,combine 标志表达不了 |
| 分组统计 | GroupedColStatsBase:2137;GroupedColStatsOut:2503 | prepass 统计折叠与一次性写出 |
| 选择 | ColVecSelect:2837 | sink;逐行沿 N 取第 idx[m] 列,直接写 (l, m) colvec |

域件追加的 op:RotaryCosSinLoad(rotary.py:15,gmem→rmem 交错 cos/sin 表,head 广播用 stride-0 repeat 表达)、RotaryCosSinTMALoad(rotary.py:192,TMA 版)、HeadRstd(head_rmsnorm.py:12,per-(row, head) rstd,`stat_value`:32)、MaxLog2(scaled_exp.py:35,prepass max 统计,`stat_value`:64)、BlockScaleFactorStore(quantize_out.py:135,mxfp8/mxfp4/nvfp4 D 的 blocked SF 写出,SM90/SM100/SM120 三路几何)。各 op 类如何覆写协议方法,见 `../architecture.md`「多态方法速查」的 EpiOp 表。

## fn 授权线:@gemm_epilogue → EpiMod → _mint

三层授权:

1. `@gemm_epilogue`(frontend.py:1818)装饰一个逐元素 fn。fn 签名即 operand 清单:第一参数必须叫 `acc`(:409-413 校验),其余参数名在 plan 期从张量元数据推 kind;`ops=` 在推歧义(m==n)或自定义 op 时显式 pin。声明项:outputs(aux tile 存储,裸名配默认 TileStore)、reduces/outs(sink)、mode、paired、prepass/prepass_outs、extra_ops(fn 看不见但驱动消费的 op)、vectorize。
2. EpiMod(:289)构造期做全部校验:保留名(:414)、sink 必须 `fn_port=="sink"` 且同名(:399-403)、scaled sink 与 acc_pair 互斥(:404-408)、prepass 与 prepass_outs 成对(:387-388)。然后算 `semantic_key`(:423-440:fn 指纹 + prepass 指纹 + outputs + mode + 各 op `cache_key` + vectorize + 绑定 Gemm 类身份),sha256 成 `semantic_digest`(:441),写 `TORCH_OP_EPI_MODS` 注册表(:446)供 `dlkernel::gemm_epi` 解析。原始构造参数存进 `_ctor_recipe`(:323),供绑定重建;生成器参数先归一化成元组(:312-315),防二次消费。
3. `_mint`(:541)按 (kind_sig, sm, paired_acc, packed_c, prepass_sig, rounding, arg_forms, add_to_output) 八元组键铸类、memo 进 `_minted`。铸 = `type()`(:620):基类对 `(_EpiModMixinBase, 绑定类.select_gemm_class(sm))`(:618-622,SM 选择权在 Gemm 类,EpiMod 不持 SM 策略),fn、op 元组、模式开关全部以类属性注入(:623-645)。`EpilogueArguments` 用 typing.NamedTuple 现场生成(:586),字段类型来自各 op 的 `arg_spec_type`——DSL 的 TVM-FFI 转换器读 NamedTuple 注解产出 launch 参数;`add_to_output` 是类属性不是字段(:589-593),不进 FFI 字段遍历。类名编码 ident 与全部 mint 开关(:594-603),防重铸冲突(:605-610);注册进 module globals 仅供进程内复用,jit cache 拿到的是 `GemmClassRef`(:646-648)。

无独立 IR:fn 在 DSL tracing 时逐元素内联,组合次序就是用户代码文本——`rope(acc) * alpha` 与 `rope(acc * alpha)` 是两个不同的 fn。frontend.py:126-150 记录了不做 epilogue 级 IR 的理由:组合点是 fn 文本(可评审),扩展点是 op(一份资源生命周期加一个端口方法即接入全部 fn),EVT 式图拓扑会藏掉 packed intrinsic 与寄存器布局控制,而 MLIR 已共享公共子表达式。

绑定:`GemmBase.bind_epilogue`(gemm_base.py:274)→ `EpiMod._bind_gemm_class`(:489)。generic 数学 mod(库工厂默认 `_gemm_cls=None`)绑到 GemmBase 根返回自身(:513-514);绑到具体类按 `_ctor_recipe` 重建,校验与 digest 注册重跑,全部运行缓存从空开始,memo `_bound_epilogues` 以实际 Gemm 类为键(:515-539)。digest、`GemmClassRef` 五元组、worker 重铸、`__getstate__` 清缓存(:448-459)——见 `../architecture.md`「绑定与身份机制」。

### 三层调用面

| 入口 | 行 | 语义 |
|---|---|---|
| `gemm` | frontend.py:660 | 底层:tile/cluster/persistent/varlen/transform_a/split_k/swap_ab 等全参数显式,组合限制逐条抛 ValueError |
| `__call__` | frontend.py:1371 | eager torch 面:解析 config(未钉 config 时走 autotune;任意模式命中实测冠军即用,miss 落 `-1` 兜底——rank 解析 config 后按该 config 切片 reduce sink 并回传视图)、分配缺失输出、sink 经 `host_finalize` 返回终值;`torch.compiler.is_compiling()` 时转单条 `dlkernel::gemm_epi`(:1465-1470) |
| `plan` | frontend.py:1705 | 只解析+编译不 launch,返回 EpiPlan(:1876);`run()` 零 host 决策,decode / CUDA-graph 入口 |

`StaticEpi`(frontend.py:1935)与 `epilogue_from_class`(:1994)给手写 GEMM 类同一 plan/run 接口,见下文逃生舱一节。

## visit.py:铸出类的设备侧循环

`_EpiModMixinBase`(visit.py:20)继承 ComposableEpiMixin,铸出类供 fn 与操作数/输出签名为类属性,这里对它们通用。四个覆写:

| 方法 | 行 | 职责 |
|---|---|---|
| `epi_to_underlying_arguments` | visit.py:38 | 设 rounding / prepass 驱动标志(:39-40),packed_cd 断言 16 位 lane(:41-44),走 `_epi_ops_to_params_dict` |
| `epi_prepass_subtile` | visit.py:82 | prepass fn 扫本 subtile 的原始累加器,返回值 flush 给 prepass sink(标量展开循环,统计扫描而非存储路径) |
| `epi_prepass_end` | visit.py:115 | barrier 定序各线程的 raw partial 平面,再跑寄存器解析(`fn_prepass_resolve`,:126-137) |
| `epi_visit_subtile` | visit.py:140 | 主循环:fn 逐元素(或逐对)调用,收集 outputs 与 sinks |

prepass 是独立于存储路径的统计扫描(驱动标志 `epi_needs_acc_prepass`,visit.py:40 置位;GemmBase 以同名类属性声明该阶段)。典型流:prepass 求 sq-sum → HeadRstd 出 rstd → 主 fn 把 rstd 当 value 操作数逐元素乘回(qknorm_epi,library.py:353)。

三种 fn 模式(`_EPI_MODES`,frontend.py:202)对应三条循环:

| 模式 | 循环形态 | fn 收到的值 |
|---|---|---|
| element | SM100 且偶数长度走 f32x2 packed(visit.py:336-362),否则标量循环(:364-381) | Float32 标量 / F2 打包对 |
| acc_pair | flat_divide 相邻 N 对视图建于循环外(:248-322),标量调用 + SM100 vectorize | `Pair(acc0, acc1)`,fn 内 unpack/pack |
| packed_cd_b16x2 | C/D 各两 16 位 lane 打进一个 f32(:185-246),结构对应手写 GemmDGatedMixin | `unpack(c)` 出 Pair,D 返回 `pack(dx, dy)` |

操作数按 kind 准备 fragment(:160-184):`apply` 端口取 `fn_prepare` 端口态,fn 收到可调用;`c` 在 SM100 全 tile 16 位输入下保持不展宽、以 F16Lanes 走混合精度加(`mixed_lanes_ok`,:151-157);`row`/`col` 已是累加器 dtype,`scalar` 是值。sink 值收集进寄存器 fragment(scaled sink 拿 (val, scale) 双 fragment,:56-69),循环后一次性 `fn_sink_flush`(:332-382,`_flush_sinks`:71)。

速度规则(frontend.py:90-103,每条都踩过):`cutlass.range` 体内静态分支必须 `const_expr`;vectorize 要求纯 loop-index 寻址(用 flat_divide 对视图,不用 `2*i`)与非零 stride(广播 fragment 先用标量展开稠密化,`_dense` 辅助 visit.py:201/:273);sink 必须在 fragment 级折叠,逐元素写零 stride 别名累加器在 SM100 packed 路径会双计;max/LSE 对 OOB lane 逐元素置 -inf(add/max_abs 的 0 恒等不适用)。

## 手写 mixin:同机制的逃生舱

GemmDefaultEpiMixin(gemm_default_epi.py:70)不经 frontend:直接声明 `_epi_ops` 元组(:71-83,alpha/beta/sr_seed Scalar、行列向量 load、两个 BlockScaleFactorStore)与手写 `EpilogueArguments`(:96),`epi_visit_subtile`(:128)内联 `apply_linear_epilogue`。它继承的正是 ComposableEpiMixin,组合出 GemmDefaultSm80/90/100/120(:155-168),服务四个架构。fn 形式是同一套 EpiOp 机制的捷径,不是第二框架(frontend.py:146-147 自述)。

逃生梯度是连续的(frontend.py:148-150):`ops=` pin 一个操作数 → 给 op 加一个端口方法 → 整体落手写 mixin,每级保留其余组合。第三级的 `StaticEpi`(frontend.py:1935)把手写类包进 plan/run 接口:无 operand 推断、无分配,要求全部带参 op 实现 host 三件套(docstring :1937-1941)。对称调度这类 fn 表达不了的数据流(scheduler 覆写)留在 mixin。

## library 工厂为何 exec 生成源码

gemm_interface 的激活族入口(gemm_with_activation / gemm_norm_act / gemm_rms)骑在参数化工厂上:linear_act_mod(library.py:528)、norm_act_mod(:581)、dact_mod(:634)、dgated_mod(:662)、rstd_gated_mod(:694)、sq_reduce_mod(:767)、ln_affine_act_mod(:841)、dact_ln_stats_mod(:889)、ln_bwd_apply_mod(:931)、dact_dbias_mod(:975),全部 `@functools.lru_cache` 按 (activation, gated, has_c, has_rowvec, has_colvec, sr, has_alpha) 布尔组合产 mod。

组合手段是 `_gen_epi_fn`(:504):拼 def 源码文本,`compile`(:509)+ `exec`(:511)。不用 closure 的原因在 library.py:485-501 注释:frontend 从 fn 签名推 operand,签名参数表就是前端接口。closure 是一个带全部可能参数的函数——缺席的 `bias_n` 参数会被推成真实 RowVecLoad(smem、cp.async、进向量循环);传中性值(bias=0.0)则成为活的运行时项而非编译掉;伪造 `__signature__` 无济于事,operand kind、trace 期调用、基于 co_varnames 的指纹读的都是真 code object。只有生成 def,「操作数缺席」才等于「该项不在 kernel 里」。指纹同时键在每组合的 bytecode 与被引激活 fn 自身源码上(getclosurevars,:488-490)。

非工厂的现成 mod 直接 `@gemm_epilogue` 声明:linear_epi(:74)、residual_epi(:257)、swiglu_mod(:197)、rope_epi(:263)、lse_epi(:280)、lse_target_epi(:289)、amax_epi(:319)、rms 族(:411-476)、ln 族(:826 起)等约 30 个;rotary.py 另有 5 个 rope mod(:324-476)。所有 mod 的数值由 tests/test_gemm_epilogue.py 对 torch reference 钉死(frontend.py:152-153)。工厂产的 op 名(mAuxOut、mRowVecBroadcast、mColVecBroadcast、mColVecReduce、sr_seed)是与 `run_gemm_epi_plan` epi_values 和 concat_layout 共用的线名(library.py:481-483)。

## legality.py 与 autotune

legality.py(115 行)是纯 host 规则,不触 benchmark、缓存身份与 CUDA launch(:1-8 自述)。三个函数:

| 函数 | 行 | 用途 |
|---|---|---|
| `sink_config_is_legal` | legality.py:22 | 布尔版:`check_oob=False` 的 sink 要求 reduce 维整除 tile(varlen 下 M 维另加禁令) |
| `validate_sink_config` | legality.py:33 | 同规则的报错版;frontend.py:1073 在 plan 期调用 |
| `prune_epilogue_configs` | legality.py:54 | 对候选 GemmConfig 逐条剪:device_capacity、gather/varlen 支持、transform_a 兼容与 padded_n 整除、blockscaled、swap_ab(仅 element 且无 sink)、acc_pair 的 tile_n 偶数与 SM90 `tile_n % 32`、sink 整除 |

消费方是扫参层:GemmEpilogueTunePolicy.prepare_candidates(gemm_epilogue_tune_policy.py:66)先走 varlen policy 的结构剪枝,再调 `prune_epilogue_configs`。边界划分:epilogue 合法性只动候选集;sink 缓冲容量是调用边界契约,由 tuned 包装逐调用校验,不改候选集与胜者(legality.py:4-7)。EpiMod 进扫参的通道是 `tuned_mod_gemm`(gemm_runtime/autotune.py:375),经 `EpiMod.gemm_tuned`(frontend.py:652,函数内 import :656 断环)或 `__call__(tune_mode=TUNED/RUNNING)`。候选枚举、评测协议、缓存键见 ./autotuning_subsystem.md。

## Complex64 域件

epilogue 数学假定实数 lane;复数输出(RoPE 后的 QK 等)由 cute_gemm/complex.py 的 Complex64 支撑。它不是 EpiOp,是元素类型件;epilogue 是它的主要消费场景。

**根本约束**:`cute.MemRefType.get` 的 C++ MLIR 绑定对元素类型白名单(int/float/ptr/sparse_elem),`complex<f32>`、`tuple<f32, f32>`、`vector<2xf32>` 一律被拒,检查在编译后的 `_cute_ir` 里,Python 层绕不过(complex.py:3-7 自述约束;详见 docs/modules/complex64.md)。可行存储只剩既有 Numeric 占用的类型,实现选 f64:re 低 32 位、im 高 32 位按位打包,算术拆两个 Float32 lane 计算后重打包,位转换由 ptxas 折叠。

**Python 类型身份**:存储是 f64,而 JIT 侧 `Numeric.from_mlir_type` 把 f64 一律映回 Float64。两条构造路径结果不同——显式传 dtype 的构造保留 `_dtype=Complex64`;从 MLIR 值构造的 smem 分配、`recast_tensor`、指针转 tensor 丢标记,需 `allocate_smem_complex`(complex.py:226)或 `recast_to_complex64`(:246)重打标(内部一行 `t._dtype = Complex64`)。不打标的后果:`smem[i] = c` 经 `_cvt_to_dest` 把 f64 打包位当实数做 cvt 截断到 f32,高 32 位清零,静默损坏。

**为何继承 Float32**(complex.py:38):混合算术 `Float32 * Complex64` 时,Python 子类优先规则让 `Complex64.__rmul__` 先于 `Float32.__mul__` 触发,分发权在本实现手里,无需 monkey-patch。对比:继承 Float64 拿到同 kind 同宽度的提升,`arith.mulf` 直接作用在打包位上,静默损坏;继承 Int64 报错响亮但不可用。结构代价:`isinstance(c, Float32)` 为 True,cutlass 仅 nvvm_wrappers.py 的 warp-reduce fmax 辅助检查它,无实际影响。同类型初始化必须短路:`__init__` 顶部对 Complex64 实例直接 `Numeric.__init__(x, .value)`,否则泛型分支把已有打包位再当实数处理。

**已验证**(tests/test_complex.py,44 用例):构造全分支、两方向算术矩阵、conj/real/imag、类不变量(width=64、`mlir_type == f64`)、三个包装函数、`__c_pointers__`(动态 SSA 抛错,与 Float32 一致)、特殊值位级往返、Float32/Float64 写入自动提升落 `(v, 0)`、smem 往返。**未验证**:TMA 携带 Complex64、copy atom、MMA、Float32 为 tensor(非标量)时的混合算术、typed smem 布局收益。

**使用注意**:

1. tvm-ffi 边界按 f64 ABI;调用处用 `complex_storage(t)`(complex.py:257)把 torch.complex64 转 float64 视图,其他 dtype 抛 TypeError。
2. 凡从 MLIR 值构造 JIT tensor 的路径都丢 Complex64 标记——用两个包装,或手动 `_retag_as_complex64`(:213)。
3. `is_same_kind(Complex64, Float32)` 为 True:标量 f32 写入自动提升为 `(v, 0)`,想存「复数位」而误传 f32 会被静默包装;要加固就在 `__setitem__` 加显式检查。
4. 不要破坏 Float32 继承——`Float32 OP Complex64` 依赖反射算符优先。
5. `_register_with_tvm_ffi`(complex.py:278,import 时执行,:292)注册 ffi 边界;为 f64→Complex64 映射打补丁时不得扰动既有 Float64 映射(当前实现不打补丁,按 tensor 重打标)。

## 相关文档

- `../architecture.md` — import 方向、「绑定与身份机制」(digest、GemmClassRef、worker 重铸)、「多态方法速查」(EpiOp 覆写矩阵)
- ./autotuning_subsystem.md — 候选枚举、评测协议、policy 钩子、缓存键
- ./gemm_runtime.md — plan/compile/launch 管线与 `dlkernel::gemm_epi`
- ../notes/epilogue_transform_reorg.md — epilogue 与 operand_transform 分包的设计记录
- docs/modules/complex64.md — Complex64 的完整设计取舍与验证清单
