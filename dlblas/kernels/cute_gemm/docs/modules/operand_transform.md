# operand_transform:A 操作数变换

`cute_gemm/operand_transform/` 把"A 操作数在 mainloop 里如何产生"抽成一个 seam:SM90 RS(WGMMA)与 SM120(warp MMA)两个 mainloop 都按 k16 块调 `copy_block(stage_idx, b, k_tile)` 产出 A fragment。默认产出是 canonical ldmatrix s2r load;一个变换用自己的产出替换它——打包权重解码、逐块数学、dropout 掩码——mainloop 继续持有 MMA issue、commit-group 纪律与 pipeline 等待(transform.py:1-21)。kernel 对变换算什么一无所知,只消费一份声明式契约(transform.py:86-149)。

约定:`路径:行号` 相对仓库根(模块内文件省略 cute_gemm/ 前缀)。对照 2026-10-13 工作区源码核实,行号随代码演进漂移。标【推断】的是分析结论,其余为源码可见事实。

## 分层

| 文件 | 位置 | 职责 |
|---|---|---|
| transform.py、kinds.py | kernel 侧,被 GemmSm90/GemmSm120 import | TransformA 契约、三个实现、操作数种类表 |
| frontend.py | fn 前端,位于 host 层之上 | `@a_transform` 铸 mod、digest、torch 表面 |
| host.py | host 侧 | handle 归一化、W4 blob/strip 几何、测量出的 config 规则 |
| formats/ | host 与 device 各半 | DecodeFormat 契约与 15 个打包格式 |
| rng.py | device | dropout 的 philox 与 PRMT/set.ge 原语 |

frontend import gemm_runtime.identity(指纹与 ref 机器),transform.py 不 import 它:kernel 侧与身份机制解耦(__init__.py:16-20)。frontend 再导出走 PEP 562 惰性(__init__.py:49-58),kernel import 本包不会拉起前端。通用宿主层(gemm_runtime.host、EpiMod.gemm/__call__/plan、torch_op)从不枚举变换种类,一律经 `as_transform_mod`(host.py:50)归一后调 TransformModBase 协议(frontend.py:147-154)。

## 三族变换

| 族 | handle | 设备类 | A 形态 | 语义 |
|---|---|---|---|---|
| value fn | `@a_transform` frontend.py:520 | TransformAValue transform.py:435 | 普通 (M,K) 16 位 | canonical load 后,fn 逐 vec_size 块原位应用于 fragment(frontend.py:9-24) |
| packed 解码 | `w4_transform(fmt)` frontend.py:477 | TransformAW4 transform.py:187 | 离线重打包 blob(`owns_a_layout`) | `copy_block(stage, 0)` LDS 本 tile 原始字并解码块 0,后续块从同一组寄存器解码(transform.py:194-197) |
| dropout | `dropout_a(p)` frontend.py:514 | TransformADropout transform.py:528 | 普通 (M,K) 16 位 | philox keep-mask AND 上 fragment;mask-only,1/(1-p) 折进 epilogue(:529-548) |

## TransformA 声明式契约

基类只有类属性,没有方法(transform.py:86-149)。mainloop 按开关调用:

| 开关 | 行号 | 含义 | 置位处 |
|---|---|---|---|
| `owns_a_layout` | transform.py:145 | mA 不是 (M,K) 操作数而是 blob;变换自持 smem 布局、TMA、gmem 切片 | TransformAW4 :211 |
| `promote` | transform.py:149 | 慢累加:每 k-tile 零初始化 wave 累加器,drain 后 fp32 折叠 | TransformAW4 构造器按 `fmt.promote` 置位 :223 |
| `uses_work_tile` | transform.py:148 | mainloop 每个 work tile 起始调 `on_work_tile` | TransformADropout :551 |
| `aux` | transform.py:146 | 安装 AuxOperandA:A 侧额外 TMA 操作数,与 A/B 同 mbarrier 到达 | W4 的 `AuxKTileStrip` :254;strip kind(:465-466) |
| `aux_raw` | transform.py:147 | bundle 的 sf 槽是裸 gmem 小张量(如 dropout seed),无 TMA/smem | TransformADropout :550 |
| `tile_k`、`a_major_mode` | transform.py:144、:143 | 要求的 tile_K;layout-owning 声明的 fragment major(K canonical) | 按格式 |

`make_copy_block(tiled_mma, sA, tCrA, tidx, warp_group_idx, sAux, mAux)`(契约 :112-122)在 kernel 内被每个 MMA warpgroup 调一次,返回闭包 `copy_block(stage_idx, b, k_tile)`:`b` 是静态 Python int(寄存器索引),`k_tile` 是全局 k-tile 序号(split-K 正确,只有坐标相关变换如 dropout 使用)。fragment atom `((2,2,2), MMA_M, MMA_K)` 两架构同构(transform.py:12-16),逐块变换体与架构无关;只有 layout-owning 解码的 m64 块/线程槽映射按 `gemm.arch` 分支(transform.py:354-366)。

mainloop 侧(RS 交叠调度,`mma_rs_interleaved` gemm_sm90.py:1702):

- `make_copy_block` 调用点:gemm_sm90.py:1221(RS)、gemm_sm120.py:1262(warp MMA,同一 seam)。
- 调度:produce(b+1) 发在 WGMMA(b) 与 WGMMA(b+1) 之间,每 k16 块一个 commit group,wait_group(mma_k-2) 是最深安全等待;下一 tile 的 slot-0 预载在当前 tile 最后一块发出(gemm_sm90.py:1716-1727 docstring,循环体 :1756-1767)。
- schedule 永远属于框架,不属于变换(transform.py:119-122)。

### 可选钩子何时被调

- `promote_acc(acc_slow, acc, zero_init)`(transform.py:412):mainloop 在每 k-tile 的块 0 零初始化 WGMMA 累加器,最后一块后 wait_group(0) 排空,再调它把 wave 折进持久累加器(`acc += scale_row * wave`)。排空有意杀死跨 tile WGMMA 重叠:协作双 warpgroup 下另一 WG 的 WGMMAs 填满 tensor pipe(transform.py:134-140)。调用链:promote_fn 在 gemm_sm90.py:1234 绑定,零初始化在 :1759/:1789,drain 加调用在 :1772-1773 与 :1806-1807。SM90 独有——SM120 无 per-k-tile promote seam,int4sm 在 SM120 被断言拒绝(transform.py:245-248)。
- `on_work_tile(tile_coord_mnkl)`(transform.py:563):每个 work tile 起始、MMA warps 进 k-tile 循环前调用(gemm_sm90.py:1303-1305,sm120 :1355-1357),刷新每 tile 寄存器状态(dropout 的每行 RNG 坐标,:564-567)。`mma_rs_interleaved` 每 tile 跑一次,两次钩子之间的所有 copy_block——含下一 k-tile 的 slot-0 预载——都属于该 tile(transform.py:126-131)。

## 操作数种类与 bundle

运行时操作数是对变换 (M,K) 索引空间的一个种类表 `ARG_KINDS`(kinds.py:263-270),是 EpiOp 操作数种类在 (M,N) 上的 mainloop 镜像(kinds.py:1-4)。一个 kind 一个对象,own 几何、device staging、host view、trace fake、fn_facing 五件事(kinds.py:5-17):

| kind | 定义 | 粒度 | 典型用途 |
|---|---|---|---|
| `colvec_ktile` | kinds.py:264 | 每 (行, k-tile) | 线性 CE dx 的 pow2 rescale |
| `colvec_k64/32/16` | kinds.py:265-267 | 每 (行, 64/32/16 元素) | dense blockscaled-SF 粒度 |
| `kvec_m64` | kinds.py:268 | 每 (m64 行块, k 元素) | LCE dw strip |
| `seed_i64x2` | kinds.py:269 | (2,) int64 [seed, offset] | dropout 内部,`fn_facing=False`(:247) |

几何只有一处陈述:`strip_geometry`(kinds.py:38-50),device staging、host view、trace fake 全调它。条带经 aux A 侧 TMA 槽投递——每 k-tile 一个 box 进 per-stage smem,与 A 同 mbarrier(AuxOperandA transform.py:64-81;W4 的字节粒度版 `AuxKTileStrip` :152)。staging 是 pure layout 组合:smem box 以嵌套 mode 广播到 (tile_M, tile_K) 再用 fragment 自己的 tiled_mma partition,每元素读取是恒等索引,fn 数学保持打包(kinds.py:103-180)。刻意不做 k 不变 colvec:逐行 scale 可交换到 epilogue((u⊙A)@B = u⊙(A@B)),fp32 colvec 精确且实测免费(kinds.py:24-26,frontend.py:50-54)。

A 跨边界走 `TransformAOperand`(transform.py:50)bundle:blob 加可选 sf 槽,占一个 mA 参数槽;host 层不学 bundle 解剖,GEMM 签名对普通 GEMM 保持固定元数(:52-58)。host 侧构造:`transform_a_operand`(host.py:260,运行时)、`transform_a_fake_operand`(host.py:277,trace)、`w4_operand_views`/`w4_fake_operands`(host.py:306/:318)。W4 几何一份实现跑在 meta 张量上,同时服务运行时 torch view 与 trace fake,两边布局不可能漂移(host.py:8-11)。

## @a_transform:铸 mod 与 digest

`@a_transform(vec_size=, packed=, consts=, regs=, args=)`(frontend.py:520)把普通 fn 铸成 `ATransformMod`(:323):

- `vec_size` ∈ {2,4,8},上限一个 k16 块——produce(b+1) 与 WGMMA(b) 的交叠属于框架(frontend.py:336-338,模块 docstring :17-21)。向量是 fragment-slot 序(2 邻 k × 2 行 × 2 k 半块),非 k 连续;块对齐使打包 16 位数学可行(transform.py:437-443)。
- `consts=callable` 每 kernel 调一次(提升的 LUT/打包常量),结果作 fn 末参(frontend.py:530-531;设备侧提升 :509-511)。
- `args={param: kind}` 声明运行时操作数;`_normalize_args`(frontend.py:419-431)对 fn 签名与 kind 表校验,按 fn 参数序固化。
- digest:`__dlkernel_semantic_key__`(frontend.py:394-416)把 fn、vec_size、consts、regs、packed 元组、args 交给 identity 的 fail-closed keyer,sha256 后(frontend.py:434-435)写进 `TORCH_OP_TRANSFORM_MODS`(:345)。闭包常量是指纹的一部分。
- `_module_locator`(:356-361)给跨进程 torch.compile 图找 import 锚,与 EpiMod 同契约。

packed 形式:fn 即 `decode_k16` 体,`PackedInput`(frontend.py:114-129)携带几何与 host 侧;mod 铸出 `DecodeFormat`(:363-392),插进一切 class 格式可用的位置。`PackedFormatMod`(:438)包装注册格式名或 DecodeFormat 实例:注册名走按名 ref(`TransformARef("w4_name")`,:465-468,磁盘键稳定、不带货),实例走 mod_local。其 digest 由 decode_k16 与 make_consts 的函数指纹构成(:451-458)——make_consts 生成设备代码必须进键,其余 host callable 的 repack 一致性由 roundtrip fixture 钉住,不进键(:400-403)。dropout handle 的 digest 只含 (threshold, rounds, version)(frontend.py:502):mask 是 (m,k,seed,offset) 的纯函数,任何 kernel 可再生、split-K 不变(transform.py:533-548)。

handle 归一:`as_transform_mod`(host.py:50-68)接受 W4 注册名、DecodeFormat 实例、任意带 digest 的 mod;名称与实例 memoize,warm 路径只付 dict 查询。

## formats/:DecodeFormat 契约

一个格式 = 自包含的解码数学 + blob 几何 + 必须一致的 host 侧(formats/__init__.py:2-9)。加格式 = 一个类加一行注册,零 kernel 改动(:11-13)。

| 契约面 | 行号 | 内容 |
|---|---|---|
| `decode_k16(xw, sfw, b, consts)` | docstring formats/__init__.py:60-65 | 设备侧:一个 m-atom 一个 k16 块,出 4 个打包 bf16x2 寄存器,fragment slot 序 |
| `make_consts` | formats/__init__.py:109 | 每 kernel 提升一次的常量 |
| `quantize_reference` | formats/__init__.py:114 | host:w → (q, sf) |
| `dequant_reference` | formats/__init__.py:117 | host:(q, sf) → fp32 w′ |
| `prepare` | formats/__init__.py:120 | host:离线重打包,字序即 decode_k16 所假设 |

类属性(formats/__init__.py:91-103):`w8`(32 B/thread 原始字)、`tile_k`、`sf_words`(0 = 无 strip)、`promote`(慢累加 W4A8)、`tile_state_words`(每 tile 从 strip 词派生的寄存器状态,`build_tile_state` 供 folded 变体 :306)、`mma_dtype`、`roundtrip`(显式 opt-out;新格式默认被 roundtrip fixture 覆盖,不静默跳过 :93-97)。

`W4_FORMATS`(formats/__init__.py:570)注册 15 个格式:

| 格式(类行号) | w8 | tile_k | sf_words | promote |
|---|---|---|---|---|
| nvfp4(:124) | | 64 | 2 | |
| int4 / int4_g64 / int4_g32(:151) | | 64 | 1;g32 为 2(:163) | |
| int4awq(:190) | | 64 | 2 | |
| int4sm(:241) | | 128 | 1 | 是(:255) |
| int4smf(:284) | | 128 | 1 | |
| int4b8(:332) | | 128 | 1 | 是(:358) |
| mxfp4(:386) | | 64 | 1 | |
| int8(:416) / fp8(:438) | 是 | 64 | 0 | |
| mxfp8(:458) | 是 | 64 | 1 | |
| qtip(:482) / qtip2s(:544) | | 64 | 0 | |
| qtip2(:512) | | 128 | 0 | |

Qtip2 增设 `decode_k16`(:527),`Qtip2S` 覆写为短流 6-SASS 版本(:560),tile_k 退回 64(:554)。int4smf 的 `tile_state_words=4` 在每 tile 块 0 建scaled LUT、decode 读表(:296-313);int4sm/int4b8 的 promote 走 TransformAW4 的 `promote_acc` 路径。

## W4 受限构造与 host 规则

`TransformAW4.__init__`(transform.py:213-294)的约束全是断言:禁 gather(:231)、禁 pingpong(:232)、tile_M % 64(:233)、cluster (1, n, 1)(:234-236)、atom_layout_n == 1(:250);SM120 还要求 (4,1,1)/(8,1,1) 解码布局且 promote 不可用(:243-248)。格式 mma_dtype 必须与 GEMM 构造的 a_dtype 一致(:219-222)。小 N decode 形态延迟绑定的寄存器预算与 occupancy 调整也在构造器(:255-294)。

layout-owning 变换只走 plain dense 路径:编译侧断言无 varlen/gather/batched/swap_ab/concat(gemm_runtime/host.py:142-144);调用侧 `resolve_operands`(frontend.py:248-284)校验 A 为 k-major、tile_K 匹配格式、K 对齐、padded N % tile_M(:269-283)。`w4_padded_n`(host.py:71-74)= blob 行数 × 64,blob-rows 规则的唯一陈述。

config 默认值是测量出的规则,不是枚举:

- `pick_w4_cfg`(host.py:160):H100 不变量——胜者把 grid 放到约 112-128 CTAs 并取能到达的最大 tile,tile_m=128 比 64 快 10-25%,serial split-k 在每 split 保留约 24 k-tiles 时补足覆盖(:169-177)。
- SM120 分支:m<=64 无条件 128 行 tile(RTX 5090 实测,:199-219);prefill(m>256)按 wave 效率 × tile 填充效率评分(:129-157),SM 数惰性查询、无 GPU 时回退常量(:88-108),GPU-blind 交叉编译不触发驱动查询。
- `pick_w4a8_cfg`(host.py:239):慢累加翻转偏好——64 行 tile 赢到 m=128,split-k 总是输,fp32 finalize 往返叠在 promote 上(:247-248)。

autotune 接入:`config_ok`(frontend.py:177-192)做廉价剪枝(swap_ab 拒绝;layout-owning 要求 non-pingpong、cluster_m=1、tile_m % 64、tile_k 匹配),剪漏的 config 由 kernel 断言兜底、bench 计 inf。完整评测协议见 ./autotuning_subsystem.md。

## 相关文档

- ../architecture.md:「A 操作数变换」小节与覆写点清单。
- ./gemm_variant_api.md:transform_a 在变体 API 中的位置。
- ./epilogue.md:EpiOp 操作数种类(ARG_KINDS 是其 (M,K) 镜像,kinds.py:2-4)。
- ./autotuning_subsystem.md:tune 子系统与 W4 tuned 调用的评测协议。
- ../notes/epilogue_transform_reorg.md:宿主管线与 epilogue/transform 分层重组的记录。
