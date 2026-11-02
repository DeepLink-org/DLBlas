# cute_gemm MoE fused grouped-GEMM primitives（SM90）

本文描述当前 MoE 生产边界。边界内是两个相互独立的 SM90 计算 primitive,能力对齐
sonicmoe 的 expert 计算面:七种激活、W13 concat/interleaved 双布局。它们服务于
已完成路由和 dispatch 的调用方,不实现路由、permute/combine、完整 MLP 编排或
autograd。GPU 端要求 CUDA SM90。激活和权重用 BF16/FP16,score 可用激活 dtype 或
FP32。

> 锚定:所有结论均可在源码中核对,`file.py:LINE` 行号对应 2026-10-11 的工作区
> (`GemmBase` 类入口、`identity.py` 中立令牌注册表与 `bind_epilogue` 均已落地)。

## 两个入口

入口定义在 `cute_gemm/fused_grouped_gemm/interface.py`,并从 fused_grouped_gemm 包和
`cute_gemm` 顶层导出:

```python
moe_up_proj_act(
    x, gate_up_weight, expert_offsets, *,
    activation="swiglu", gate_up_layout="concat",
    token_indices=None, gate_up_bias=None,
    store_preact=True, tuned=True,
) -> tuple[Tensor, Tensor | None]

moe_down_proj_dact(
    grad_output, down_weight, gate_up_preact, expert_offsets, *,
    activation="swiglu",
    token_indices=None, router_weights=None,
    compute_router_weight_grad=False, tuned=True,
) -> tuple[Tensor, Tensor, Tensor | None]
```

两个入口都是薄转发:函数体只有一次对 `GemmMoeSm90.up_proj_act`/`down_proj_dact` 的
显式参数转发（`interface.py:47-57`、`:82-92`）。完整的 MoE host 调用契约全部在
`cute_gemm/fused_grouped_gemm/gemm_sm90.py` 的两个 host 类方法里(`up_proj_act` 与
`down_proj_dact`),内容覆盖:

- 白名单常量
- 张量契约校验(含 D=0 拒绝)
- GEMM 参数映射
- 严格解包

调用链依次为:

1. 公开入口
2. `GemmMoeSm90` host 类方法
3. `cls.gemm_with_activation`/`cls.gemm_with_activation_grad`(`GemmBase` 类入口)
4. `gemm_interface.py` 的共用变体实现(eager call helper 或 typed out-op)
5. epilogue/runtime
6. 继承自 `GemmSm90` 的设备实现

构造函数不负责 host 调用。它仍只做 plain 16-bit、无 RS、无 transform 的构造限制
(见"与库内机制的连接"的角色类条目)。

`activation` 白名单与 sonicmoe 的 `ActivationType` 一一对应（字符串同名）:gated
swiglu/geglu/reglu 与非 gated silu/relu/relu_sq/gelu_tanh_approx
（`gemm_sm90.py:51-52` 的类常量）。gated 激活把 `gate_up_weight` 按 gate/up 二分。
存储默认 concat `[all gate; all up]`,可选 interleaved `[g0,u0,g1,u1,...]`。
`gate_up_bias` 随权重同布局。
非 gated 激活消费普通 `[E,I,D]` 投影。`gate_up_layout="interleaved"` 对其显式报错。

第一个入口执行专家 W13 projection 和融合激活,返回 `(postact, preact)`。返回顺序
对齐 sonicmoe `_UpProjection` 的 `(a, h)`。`store_preact=False` 时第二项为 `None`。
第二个入口执行 down projection 的 input-gradient GEMM、dAct 和 postact 重算,返回
`(grad_gate_up, weighted_postact, grad_router_weights)`。返回 tuple 的长度固定。
没有 score 或未请求 score 梯度时,第三项为 `None`。

这两个入口是计算边界。它们不隐式计算 dW13、dW2、dX、bias 梯度,也不承诺
`.backward()` 完成完整训练。调用方负责:

- 保存张量
- W2 forward
- 梯度 GEMM
- 路由聚合
- 生命周期

## 数据与布局合同

设原始 token 数为 `T`,路由项数为 `R`,专家数为 `E`,hidden size 为 `D`,intermediate
size 为 `I`:

| 参数 | 形状与含义 |
|---|---|
| `x`, `grad_output` | `[T, D]`,原始 token 顺序 |
| `gate_up_weight`(gated) | `[E, 2I, D]`,concat 行序 `[all gate; all up]` 或 interleaved `[g0,u0,...]` |
| `gate_up_weight`(非 gated) | `[E, I, D]`,普通投影 |
| `gate_up_bias` | gated `[E, 2I]` / 非 gated `[E, I]`,布局随权重 |
| `down_weight` | `[E, D, I]` |
| `expert_offsets` | CUDA int32 `[E+1]`,expert-major 路由分段,首值为 0、末值为 `R` |
| `token_indices` | CUDA int32 `[R]`,分组行 `r` 指向原始 token 行 `t` |
| `router_weights` | FP32/FP16/BF16/FP8 E4M3FN/E5M2 `[R]` score,与分组行对齐 |
| `gate_up_preact`(gated) | `[R, 2I]`,interleaved `[gate0, up0, ...]`,与输入布局无关 |
| `gate_up_preact`(非 gated) | `[R, I]`,自然序 |
| `postact` | gated `[R, I]`,非 gated `[R, I]` |
| `grad_router_weights` | FP32 `[R]`,顺序与 `router_weights` 相同 |

`token_indices=None` 表示输入已经是 `[R, D]` 的连续路由行。它只是 primitive 的可选
operand。offsets 的首值==0、单调性、末值与 indices 值域属于调用方 producer
contract。生产热路径不通过 `.item()` 做设备回读。

低精度 `router_weights` 必须连续(stride=1),基址按 16 字节对齐。不符合时在
调参前或实际 launch 前抛出 `ValueError`,包括命中缓存后换入未对齐指针。
FP32 保留原加载路径。低精度列向量使用 16 字节对齐窗口的 cp.async,在 shared
中跳过前缀,按当前 expert 的有效行数对尾部补零。读取后在寄存器中转 FP32;
不另发 cast/clone kernel。该契约同样适用于 gather 模式,score 始终按分组行排列。
FP8 score 表示其原生数值,不隐含额外量化 scale。激活/专家权重仍仅支持 FP16/BF16,
`grad_router_weights` 仍为 FP32。

校验自身也不产生回读。校验逻辑位于 `GemmMoeSm90` 的私有方法,在类方法内、任何
backend 调用之前执行:

- 白名单与布局合法性:`_check_activation` 和 `_validate_up_inputs`/
  `_validate_down_inputs` 做 host 全量检查（`gemm_sm90.py:357`、`:213-266`、
  `:268-355`）
- offsets 的首值==0、单调性与末值:仅在 offsets 位于 CPU 时执行(`_check_offsets`,
  `gemm_sm90.py:362-394`)
- `token_indices` 值域检查同样仅 CPU 时执行（`gemm_sm90.py:396-421`）
- CUDA 张量只做 metadata 检查,内容子检查跳过,仍由 producer contract 保证

dS 的逐行归约在 epilogue 内累积。host 侧只对 partials 做一次求和收尾
（`VecReduce.host_finalize` 定义于 `cute_gemm/epilogue/ops.py:1526`,由
`cute_gemm/epilogue/frontend.py:1297-1310` 的 `_finalize_sink` 调用）。路径上没有
`.item()`。

## 上投影的 gather 与融合激活

对 expert 段内的每个路由行 `r`,令 `e` 由 `expert_offsets` 确定,`t=token_indices[r]`:

```text
z[r] = x[t] @ gate_up_weight[e].T + gate_up_bias[e]
gated:    G[r], U[r] = split(z[r]);  H[r] = interleave(G, U);  A[r] = act(G[r], U[r])
ungated:  H[r] = z[r];  A[r] = act(z[r])
```

gated 的 split 按 `gate_up_layout`:concat 取前后半,interleaved 取奇偶列。两种布局
输出的 `H` 恒为 interleaved。实现把 `token_indices` 作为 A operand gather。concat W13
经 `concat_layout=("B","bias")` 地址映射读取。interleaved 与非 gated 走
`concat_layout=None` 的 native 路径（`gemm_sm90.py:124-141`）。`store_preact=False`
只在调用方确定不需要反向输入时使用。

## 下投影 input-gradient 契约

对同一分组行,`G0[r] = grad_output[t] @ down_weight[e]` 是未乘 score 的 accumulator。
当传入 `s=router_weights[r]` 时,primitive 计算:

```text
gated:    A[r] = act(H_gate[r], H_up[r]);  dH = dAct(s[r]*G0[r], H);  A′[r] = s[r]*A[r]
ungated:  A[r] = act(H[r]);  dH = dact(s[r]*G0[r]) * ... ;  A′[r] = s[r]*A[r]
dS[r] = sum(A[r] * G0[r])
```

`dS` 不多乘一次 `s`,不除以 `s`,也不含 down bias 的贡献。因此 score 为零时,`dS`
仍可能非零,而 `A′` 为零。`router_weights=None` 时跳过 score scaling。此时 `A′`
就是未加权的 `A`。请求 `compute_router_weight_grad=True` 时必须提供 `router_weights`。

gated 的 `grad_gate_up` 为 interleaved dH。调用方用 concat-aware GEMM 计算 W13 梯度,
不能把该 primitive 当成 dW13 接口。

## 调用方边界与内存

典型测试调用方的顺序为:

1. 上投影 primitive
2. 独立 W2 forward 和路由聚合
3. 反向调用下投影 primitive
4. 通用 gather GEMM 求 dW2/dW13
5. concat W13 GEMM 求 dX
6. 还原 score 梯度到原始路由顺序

调用方可以短暂物化 `Y`、分组 dX、dH 和 A′。但不应额外构造完整 `X_e`、`dO_e` 或
`G`。

primitive 显式拒绝零收缩:up/down 的校验在收缩维 D=0(K=0)时直接抛 `ValueError`,
错误消息含 "contraction"（`gemm_sm90.py:248-252`、`:308-312`）。通用层共用实现
同样拒绝 K=0 且输出非空的输入,检查位于 `_dispatch_activation_gemm`/
`_dispatch_activation_grad_gemm`。host 侧没有空收缩的退化数学模拟。M=0/N=0(空
输出)仍合法,输出为相应形状的空张量(空输出 fast path)。通用层其它变体
（`gemm`/`gemm_add`/`gemm_norm_act` 等)的退化行为不属于本 primitive 的承诺面。

单个 primitive 不决定跨层保存策略。Sonic 风格的生命周期验证只放在
`tests/test_fused_grouped_gemm_sm90.py` 或 benchmark 中。生产目录不新增 autograd、
router、config 或 runtime 层。

## 与库内机制的连接

两个 primitive 不带专用 GEMM runtime。公开入口只薄转发到 `GemmMoeSm90` 的两个 host
类方法（`interface.py:47-57`、`:82-92`）。类方法调用 `cls.gemm_with_activation`/
`cls.gemm_with_activation_grad`。类入口在函数体内延迟调用 `gemm_interface.py` 的共用
变体实现。role 模块顶层只有 torch、`SplitKMode`、`GemmSm90` 三个依赖,顶层不引用
通用 GEMM 接口。gated 与非 gated 走同一对类入口。数学由 `activation` 选择,concat
调用显式传 `concat_layout`。`cute_gemm/linear.py`、`cute_gemm/mlp.py` 的 Ops 绑定用
类入口,前端选择器是 `linear_act_func`/`linear_gated_func`/`mlp_func`。激活字符串
与 gated 判定在共用实现分发:dispatch 按 `is_gated` 决定 concat_layout 是否生效,
`_run_activation_grad_gemm` 选 `dgated_mod`/`dact_mod`。

- 接线:上投影调 `cls.gemm_with_activation`,下投影调 `cls.gemm_with_activation_grad`。
  共用实现为 `_dispatch_activation_gemm`/`_dispatch_activation_grad_gemm`,执行侧为
  `_run_activation_gemm`/`_run_activation_grad_gemm`。eager 路径在执行 helper 里取
  library 数学 mod,再解析令牌并绑定类(见下一条)。mod 工厂定义于
  `cute_gemm/epilogue/library.py`。`linear_act_mod`、`dact_mod`、`dgated_mod` 只产
  数学 mod。`linear_act_mod` 的 gated 分支生成 `unpack` gate/up 对的 epilogue 体。
  七种激活在通用栈已有覆盖:varlen_m+geglu、MLP 5 激活、reglu 量化、swiglu_oai 截断。
- 角色选择:kernel class 的 SM 分派与 role 校验只有一个入口 ——
  `GemmBase.select_gemm_class(sm_major)`(`cute_gemm/gemm_base.py`)。generic root 按
  major 选基类:8→`GemmSm80`、9→`GemmSm90`、10/11→`GemmSm100`、12→`GemmSm120`。
  具体类按 `supported_sm_majors` 校验(`GemmMoeSm90` 继承 SM90 的 `(9,)`)。未知
  major 报 `ValueError`,类不适用报 `TypeError`,没有静默跨 SM 回退。`_mint` 铸类时
  调用绑定类的 `select_gemm_class`。
  torch.compile 图边界的 `kernel_variant` 字符串是类自持的 `variant_id`
  (`GemmMoeSm90.variant_id == "moe_sm90"`)。令牌在类定义后由
  `register_kernel_variant()` 显式注册进 `gemm_runtime/identity.py` 的中立注册表。
  注册表不 import kernel 模块,`gemm_interface.py` 不反向 import MoE 模块。op
  执行期经 `resolve_kernel_variant` 还原。`None` 常量返回 `GemmBase`。未知令牌
  报错,不回退 generic。无 codec 层,编译图按当前格式整体重建。
  moe_down 对类入口的返回值按 `compute_router_weight_grad` 严格解包。错 arity 报
  RuntimeError,不做 None 补齐。
- 缓存身份:digest 由原始 epilogue 声明与实际 Gemm 类共同决定(`semantic_key` 的
  `("gemm_cls", None|(module, qualname))` 条目)。`GemmMoeSm90.bind_epilogue(math_mod)` 首次绑定时经
  `EpiMod._bind_gemm_class` 从原始构造参数重建 mod。重建走 EpiMod 构造器,保留校验
  与 `TORCH_OP_EPI_MODS` 摘要注册,quantizer 不重复提升。全部运行缓存从空开始。
  绑定 memo `_bound_epilogues` 以实际 Gemm 类为键。warm 调用直接命中 memo。generic
  类绑定与同类重复绑定直接返回原 mod。MoE 绑定 mod 与 generic 数学 mod 的 digest
  不同,各挂各的 plan cache,不新建缓存结构。序列化(`__getstate__`)清空
  `_minted`/`_plan_cache`/`_call_cache`/`_bound_epilogues` 四个运行缓存。worker
  payload 重建出的 digest 与提交的一致。torch.compile 路径把 role 物化为
  `variant_id` 令牌(`kernel_variant` schema 不变)。磁盘缓存兼容不是 key schema 的
  设计约束。摘要结构变化允许 `.o` 与 tune winner 一次性失效重编译(先例见
  `docs/notes/epilogue_transform_reorg.md`)。
- 角色类:`GemmMoeSm90`（`cute_gemm/fused_grouped_gemm/gemm_sm90.py:34`）= host 调用
  契约 + 受限构造 + 独立身份。两个 host 类方法是纯 Python classmethod。它们只使用
  本次调用的 Tensor,不在类上存放 Tensor、plan 或输出(执行状态归通用 runtime)。
  构造函数不参与 host 调用,仍只拒绝非 16-bit dtype、`mma_is_rs`、`transform_a`
  （`gemm_sm90.py:73-77`）。`split_k_mode` 默认 `SplitKMode.SERIAL`（`:69`）。设备侧
  各阶段(含 16-bit WGMMA 主循环 `mma`)全部继承通用 `GemmSm90`,本类不重写任何
  设备方法。未来 MoE 特化的 load/schedule/MMA 改动以方法覆写形式落在这里(今天没有
  独立的 MoE tile scheduler)。minted kernel 类在 SM90 的基类由绑定类的
  `select_gemm_class(9)` 选出,即 `GemmMoeSm90`。令牌只注册静态类,动态铸出的
  子类不进注册表。激活选择只改 epilogue 函数体,不改 mint 结构。
- autotune:两个 host 类方法固定 `dynamic_scheduler=True` 并透传 `tuned`。
  `tuned=True` 走 `GemmEpilogueTunePolicy`
  （定义于 `cute_gemm/gemm_epilogue_tune_policy.py:16`,装配在
  `cute_gemm/gemm_runtime/autotune.py:349-365`）。winner key 由 `gemm_tune_key` 构造。
  平均每专家行数经 `average_rows_bucket` 向上取 2 的幂,下限 32,再作为 `avg_expert_m`
  挂进 key（`cute_gemm/varlen_selector.py:636-645`）。
  `_MIN_AVERAGE_ROWS=32` 在 `:601`,挂 key 在 `:792`。带 `A_idx` 时 routed 行数取
  `A_idx.shape[0]`（`:776-782`）。63 行/专家与 128 行/专家分桶,`A_idx` 的具体取值
  不进 key。

## 测试边界

`tests/test_fused_grouped_gemm_sm90.py` 按验证职责组织(可用 CUDA 且 SM90 时运行),
用例数量随职责增删,不在此固定:

- 端到端对拍:粒度对齐 `.vscode/sonic-moe/tests/moe_test.py`。swiglu(concat/interleaved
  两种 W13 行序)与 relu_sq 等组合,上投影的 preact/postact 直接喂下投影,对拍独立
  PyTorch oracle。覆盖非单调、重复 token-row gather、空 expert 段、零/负 score、
  dS 不乘 score,以及无 score 时固定三返回值与 `dS=None`。
- 零收缩拒绝:D=0 的 up/down 在校验层抛 `ValueError`(消息含 "contraction")。CPU
  用例同样钉死通用层共用实现对 K=0 且输出非空的拒绝。M=0 合法空输出的行为另有
  用例守住。
- offsets host 合同:CPU `expert_offsets` 的首值==0、单调、末值==R 及各违反例,均从
  公开 up 入口驱动。spy 断言非法输入下 backend 调用次数为 0。合法输入下调用参数
  原样透传(offsets 是同一张量、令牌解析回 `GemmMoeSm90`、`dynamic_scheduler=True`)。
- fullgraph 与身份接线:torch.compile(fullgraph=True) 下激活 × gather × reduce 的
  公开面变化对拍 reference。同组还覆盖:
  - role 以 `variant_id` 令牌过编译边界,令牌经 identity 注册表还原,与
    `GemmClassRef` 解析回 role 类本身
  - minted MRO 顺序(role 在 `GemmSm90` 之前)经 wire 上的 decode-spy 观察
  - `select_gemm_class` 的 role/`supported_sm_majors` 规则
  - 受限构造合同(拒绝非 16-bit/RS/transform,`split_k_mode` 默认 SERIAL)
  - moe_down 对类入口返回值的严格解包(错 arity 抛 `RuntimeError`,不 None 补齐)
- 共享主循环矩阵:七种激活(gated 双布局、ungated concat)× fp16/bf16,对拍 fp32
  oracle 的 preact/postact/dH/A′/dS。另含 K 跨多个 k-tile、dS 跨输出 tile 的跨 tile
  形状。
- 正交边界:路由形状(R>T 重复 gather、空头/尾专家、packed 行)、bias/preact 模式
  (无 bias、fp32 宽 bias、`store_preact=False`)。
- tuned 分桶:同桶的第二次调用复用首次 autotune winner(autotuner cache 计数钉死),
  数值对拍 reference。
- config 钉死的 pingpong/K 边界:显式 `GemmConfig` + `GemmMoeSm90` 类入口(令牌
  `"moe_sm90"`)。K 取恰一个 k-tile 与 k-tile 加尾部两种,pingpong 开关各有覆盖。
  每例带无 role 的 generic 对照臂(config 钉死仅 eager 验证,torch.compile 下显式
  config 会报 `NotImplementedError`)。
- 生产形状与开关矩阵:`.vscode/moe_shape.md` 的模型形状 × 多个 token 数对拍独立
  reference,含 16K token 稳态样本。另有一组把各 primitive 开关交叉组合的刁钻矩阵
  (每例含 backward=dact 覆盖)。
- 冷进程与异步池:冷子进程经 `identity.py` 的令牌注册表还原 role,并完成一次数值
  正确的调用。异步编译 worker 内解析 STATIC `GemmClassRef`。

tune key 分桶由 `tests/test_gemm_policy_contract.py` 的用例覆盖。两个 primitive 通过
不等于完整训练框架或分布式 MoE 已验收。调用方需另行验证聚合、参数梯度和通信。
