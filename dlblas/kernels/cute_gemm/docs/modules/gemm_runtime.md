# gemm_runtime:通用宿主管线与身份机制

`cute_gemm/gemm_runtime/` 把每个 epilogue/transform GEMM 变体都要写的宿主侧三步——建 plan、编译、启动——收成一份 EpiOp-schema 驱动的实现,并配一层让动态铸出类跨进程复活的身份机制。四个文件一条依赖链:`identity.py`(叶子,491 行)→ `host.py`(plan 层)→ `torch_op.py`(torch.compile 边界)与 `autotune.py`(扫参接入)。变体文件不再各留一份约 400 行的近似拷贝,只剩 mixin、per-SM 类 stamping、校验断言和薄包装(host.py:1-19)。

约定:`路径:行号` 相对仓库根(模块内文件省略 cute_gemm/ 前缀)。对照 2026-10-13 工作区源码核实,行号随代码演进漂移。标【推断】的是分析结论,其余为源码可见事实。

## 谁共用这条管线

管线是三个函数:`build_gemm_epi_plan`(host.py:314,冷路径)→ `GemmEpiPlan`(host.py:267,缓存对象)→ `run_gemm_epi_plan`(host.py:509,热路径)。

| 消费者 | 进入点 | 说明 |
|---|---|---|
| fn epilogue | `EpiMod.gemm` epilogue/frontend.py:660、`__call__` :1371、`plan` :1705 | `gemm` 冷路径 build(:1166)、热路径 run(:855/:1207);`__call__` run(:1557);`plan` run(:1920) |
| 预解析 plan 层 | `EpiPlan` epilogue/frontend.py:1876、`StaticEpi` :1935 | `EpiPlan.plan`(:1946)build(:1963);`run()` 按构造零 host 决策 |
| 手写变体 | gemm_iface.py:267、:290 | 带 `warm_slots` 的变体 replay 直接调 `run_gemm_epi_plan` |
| W4 包装 | gemm_w4.py:17、:129、:238 | `gemm_w4a16` 是 `EpiMod.gemm(transform_a=...)` 薄糖 |
| autotune | gemm_runtime/autotune.py:375 | 候选经 `mod.gemm` 重入同一 plan cache |

reduction 家族不在这条管线上,它有自己的 `@cute_op` / `@jit_cache` / `@autotune` 三层(../architecture.md)。全库执行时序见 ../architecture.md;fn 前端与 EpiOp 词表见 ./epilogue.md,变体 API 见 ./gemm_variant_api.md。

## identity.py:身份层(叶子)

identity.py 回答一个问题:动态铸出的类只存活于创建进程,编译缓存、异步 worker、torch.compile 图如何重新找到它。

### 四张 digest→mod 注册表

| 注册表 | 定义 | 键与消费方式 |
|---|---|---|
| `TORCH_OP_EPI_MODS` | identity.py:234 | epilogue digest;`dlkernel::gemm_epi` 解析用 |
| `TORCH_OP_TRANSFORM_MODS` | identity.py:235 | transform digest;同一 op 的 transform 侧 |
| `LOCAL_EPI_MODS` | identity.py:283 | 无 import 锚点的 EpiMod;`consume=True`,resolve 即弹出 |
| `LOCAL_TRANSFORM_MODS` | identity.py:284 | 小型格式/fn mod 包;`consume=False`,可多次 resolve |

写入只发生在 mod 构造期,绝不在 trace 期:Dynamo 把全局 dict 变异缓冲为延迟副作用,trace 期写入对 fake-tensor pass 不可见(identity.py:229-233 注释)。构造期注册共三处:epilogue/frontend.py:446、operand_transform/frontend.py:345 与 :459、:503(dropout handle)。`consume` 差别的理由:worker payload 可能闭包大块 Python 状态且 worker 存活整个会话,弹出即释放;transform 包小,一个 plan 可能 resolve 多次(identity.py:240-244)。

### fail-closed 指纹

`function_semantic_key`(identity.py:160)= (module, qualname, 源码 sha256, 默认参数, 闭包依赖)。闭包依赖取 globals 加 nonlocals,按名排序,跳过 dunder 名(:177-188)。`semantic_value_key`(:40)递归指纹值:原语、容器、enum、模块与类(按 qualname)、函数/方法/builtin/partial、dataclass、torch.dtype,以及实现 `__dlkernel_semantic_key__` 的对象(:54-65);其余一律 TypeError(:121-128)。理由在 docstring:过粗的键会静默复用错误的 kernel(:47-50)。

两处宽松都有明确的边界理由:

- 外部函数(stdlib/site-packages)只按 qualname 指纹(identity.py:136-157)。递归进去会把运行时可变的库全局拖进 digest——例如任何碰 cutlass `dsl_user_op` 的函数会到达懒加载的 `_DSL_PACKAGE_ROOT`,digest 就取决于本进程是否已编译过东西,异步 worker 会把 ref 误判为"已变更"(:141-147)。
- `module_locator`(:199)给 digest 键对象找 (module, global_name) import 锚;`__main__` 与 notebook 对象返回 None。调用期才绑定的模块全局能骗过扫描、在 fresh import 里却不存在(:207-213),消费端因此在 resolve 时容忍失败并用按值 payload 兜底(:391-410)。

### 两个 pickle ref 与令牌注册表

| 机制 | 定义 | 行为 |
|---|---|---|
| `GemmClassRef` | identity.py:307 | 五元组 `(kind, module, qualname, mint_key, semantic_digest)`,kind ∈ static / epi_mod / epi_mod_local |
| `resolve_gemm_class` | identity.py:376 | static 直接 import 加 qualname;epi_mod import 后校验 digest、不符走 payload;最后 `obj._mint(*mint_key)` 重铸等价类 |
| `__dlkernel_pool_payload__` | identity.py:331 | mod 以 cloudpickle 按值发往异步 worker;通道不进缓存键,序列化失败则池拒绝该键、退回进程内编译(:318-322) |
| `TransformARef` | identity.py:458 | `w4_name` 按名从 W4_FORMATS 重铸(:477-482),或 `mod_local` 走 `LOCAL_TRANSFORM_MODS` |
| `KERNEL_VARIANT_CLASSES` | identity.py:421 | kernel_variant 令牌 → 静态 Gemm 类,`GemmBase.register_kernel_variant`/`resolve_kernel_variant` 的线侧 |

令牌注册表的三条规则:注册只经类方法在 role 模块 import 期发生,不走 `__init_subclass__`,所以 type() 铸出的 kernel 类进不了表(:414-420 注释);同 token 换类抛 ValueError,表保持不变(:432-440);未知 token 抛 ValueError,解析绝不回退到泛型类(:443-455)。

identity.py 模块级零 cute_gemm import(:25-37 全 stdlib);三处函数内 import(cloudpickle 与 PoolPayload :264-266、:355-357,operand_transform.host :480)是延迟的,不破坏叶子性。上层五个模块依赖它而它不依赖任何人(../architecture.md「信息隐藏排序」)。【推断】按删除测试衡量:删掉它,跨进程编译缓存、torch.compile 支持、异步编译池同时失效,它是全库分量最重的叶子。

## host.py:plan/compile/launch

| 函数/对象 | 行号 | 职责 |
|---|---|---|
| `_compile_gemm_epi` | host.py:84(`@jit_cache` :83) | 编译入口,全部参数是可 pickle 原语 |
| `GemmEpiPlan` | host.py:267 | 18 字段启动配方,按元数据键缓存 |
| `build_gemm_epi_plan` | host.py:314 | 张量元数据 → epi_keys → 编译 → plan |
| `run_gemm_epi_plan` | host.py:509 | 热路径,只传每调用指针与标量 |

冷路径:推导 majors(`_get_major` :310)、batched(:383)、四个 dtype(:407-410;layout-owning transform 时 a_dtype 取格式的 mma_dtype、batched 强制 False :401-405)。epi_keys 由每个活跃 op 的 `host_arg_key` 产出,按名排序(:414-419)。`_compile_gemm_epi` 解析 `GemmClassRef` 回类(:135)、transform ref 回 mod(:138-140),为活跃 op 构造 fake 参数(:195-198),split-K 追加 semaphore/workspace fake(:199-210),交给 `compile_gemm_kernel`(:232-264)。每个参数都是可 pickle 原语,jit_cache 用它做磁盘键并把冷 miss 投递给异步 worker(:130-133)。

GemmEpiPlan 的 18 个字段按用途分四组:

| 组 | 字段 |
|---|---|
| 编译产物与 schema | `compiled_fn`、`gemm_cls`(携带 op schema 与 EpilogueArguments 类型)、`epi_arg_keys` |
| 调度器 | `is_sm100_family`(:488 置位)、`max_active_clusters`、`max_swizzle_size`、`scheduler_uses_semaphore`(:464 置位)、`scheduler_static` |
| 热路径预计算 | `call_ops`((name, converter, key) 三元组,:473-482)、`arg_template`(全 None 字段模板,:487) |
| 几何与 split-K | `tile_M`/`cluster_M`/`tile_N`/`cluster_N`、`split_k`/`split_k_mode`、`d_transposed`、`cu_tiles_tile_m` |

`call_ops` 与 `arg_template` 是启动开销优化的关键:warm 调用做一次 dict `.copy()` 加按位 `_make`,不重建字典、不解析 kwargs(:287-294 注释,:533-541)。split-K 缓冲经函数内 import 从 `cute_gemm.gemm._split_k_buffers` 取(:546),保持本模块在 gemm.py 之下的 import 位置。

两个调度布尔只在此处判定,plan 构造后不再有 host 分支:`scheduler_uses_semaphore = is_dynamic_persistent and capacity[0] == 9`(:464,只有 SM90 动态持久调度器消费 semaphore,注释 :462-463);`is_sm100_family = capacity[0] in [10, 11]`(:488,SM100/110 用 2-CTA MMA)。

transform_a 是 build 的一等参数,但本层不枚举变换种类:`as_transform_mod` 归一后只问三个问题——`compile_ref()` 取 pickle ref(:358-364),`owned_fmt` 非空则 A 换成 blob、取静态几何(:365-371),`needs_operands` 则断言 A 是 `TransformAOperand`(:372-382)。trace 侧对应两条 fake 路径:layout-owning 用静态几何 `fake_operands`(host.py:148),带运行时操作数的 value 变换用 `fake_bundle`(:189-191)。`FakeArgCtx`(host.py:52)携带 swap-at-trace 标记:swap 时 m/n 是 kernel 维,tile 形状参数按 caller 取向跨界、trace 期转置。变换模块细节见 ./operand_transform.md。

## torch_op.py:torch.compile 边界

一个 mutating custom op 覆盖全部 epilogue-GEMM,包括用户 `@gemm_epilogue` fn——它们原本没有 torch.compile 故事,先例是 torch.compile 自己的 triton_kernel_wrapper_mutation(torch_op.py:2-11)。

| 入口 | 行号 | 形态 |
|---|---|---|
| `dlkernel::gemm_epi` | torch_op.py:92-93 | mutating:`outs` 携带 D + 声明输出 + reduce partial,全部图拥有 |
| 其 register_fake | torch_op.py:140-144 | 纯 no-op,编译归 jit_cache 与异步池在真实执行时负责 |
| `dlkernel::gemm_epi_f` | torch_op.py:178-179 | functional:输出在 op 内分配,一图一节点 |
| `compile_call` | torch_op.py:212 | `EpiMod.__call__` 的 compile-path body;派发经 epilogue/frontend.py:1470 的函数内 import |

边界规则:

- host 常量(config、flags、标量操作数)塞进 `meta` repr 字符串,Dynamo guard 落在字符串上,正好钉住编译行为(:13-19)。`_config_meta`(:198)逐字段 getattr 而不 `repr(cfg.__dict__)`:`__dict__` 对 Dynamo 是 dict 变量,items 停在 LazyVariableTracker,repr 无法常量折叠,fullgraph 会挂(:201-206)。
- `default_config` 被 `torch._dynamo.assume_constant_result` 标记(:56):结果由 device guard 钉死,trace 期直接求值,不让 Dynamo 追进 lru_cache 帮助函数。
- digest 解析 `_resolve`(:72):先查注册表,未命中按 locator import,import 到的 mod digest 不符即抛 RuntimeError(:79-83),并回填注册表。
- caller 持有 out= 或 partial 缓冲时走 mutating 形态(缓冲不能图拥有,:253、:303);否则走 functional(:342)。带 sink 时 config 被钉死——partial 缓冲必须在 op 运行前按精确形状图分配,op 内 autotune 不允许换 tiling(:248-252)。varlen M-fold sink 在 functional 路径直接 NotImplementedError(:331-340)。
- 序列化一侧:`EpiMod.__getstate__`(epilogue/frontend.py:448)清四个运行缓存——`_minted`、`_plan_cache`、`_call_cache`(:455-457)与 `_bound_epilogues`(:458)。worker 按配方重建出的 digest 与提交一致,缓存键稳定。

## autotune.py:mod 级扫参接入点

`tuned_mod_gemm`(:375)让任意 EpiMod(含 transform)直接复用 Autotuner 机器:CUDA-graph L2-cold bench、异步编译池重叠、磁盘缓存。使通用路径成立的关键机制:

- 每个 `(mod.semantic_digest, epi_names, has_c, capacity, transform digest, ta_names)` 一个 tuner(:338-346)。`mod_digest` 顶进 key,改 epilogue fn 即同时失效内存与磁盘 tuning 缓存(:17-19)。
- `epi_args` 摊平成具名 kwargs,wrapper 合成 `__signature__`(:305-331):Autotuner 从顶层张量 kwargs 派生缓存键与 L2 轮换 clone 集,dict 值既不能 key 也不能 clone,否则每次 bench replay 共享一个缓冲(:12-16)。
- 失败候选计 inf 而不中止 sweep:`mod.gemm` 的 ValueError/TypeError/AssertionError 重包成 RuntimeError(:300-303),bench 循环只把 RuntimeError/MemoryError 转成 inf(:27-28)。
- reduce sink 全缓冲契约:`sink_arg_shapes`(:114)按候选空间最坏 tile 给缓冲上界,undersized 调用方缓冲在查缓存前拒绝(:142-178),一个缓冲按 config 切片服务整个扫描(:181-200)。

候选枚举、评测协议、与 `GemmEpilogueTunePolicy` 的衔接见 ./autotuning_subsystem.md,本文不重复。

## 相关文档

- ../architecture.md:执行时序、绑定与身份机制、信息隐藏排序。
- ./epilogue.md:EpiOp 词表、fn 前端、EpiMod 三层入口。
- ./gemm_variant_api.md:变体 API 与手写变体的规范化。
- ./autotuning_subsystem.md:tune 子系统全貌。
- ../notes/epilogue_transform_reorg.md:宿主管线与 epilogue/transform 分层重组的记录。
