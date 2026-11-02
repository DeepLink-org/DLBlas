# 变体 API(gemm_interface / gemm_iface)

GEMM 家族的 torch 侧门面。`gemm_interface.py`(3581 行)持有全部变体入口:输入校验、输出分配、blockscaled 容器拆包、typed out-op 注册、eager 快路径 plan cache、reference 实现。`gemm_iface.py`(372 行)把变体之间的重复样板(operand 规范化、semaphore 规则、interface plan 与 warm 重放)收敛为一份 `VariantSpec` 声明加一个通用引擎。两层自身不编译、不启动 kernel:编译与 launch 在低层 dispatch(见 [./gemm_dispatch.md](./gemm_dispatch.md))与通用宿主管线(`gemm_runtime/`)。

约定:`路径:行号` 相对仓库根。对照 2026-10-13 工作区源码核实,行号随代码演进漂移。标【推断】的是分析结论,其余为源码可见事实。

## 职责边界

| 职责 | 位置 | 说明 |
|---|---|---|
| 变体入口(torch 函数) | `gemm_interface.py:892`(`gemm`)、`:1480`(`gemm_add`)、`:3191`(`gemm_rms`)等 20 余个 | 校验、输出分配、空输入快路径 |
| blockscaled 容器拆包 | `gemm_interface.py:191`(`_unpack_operand`)、`:204`(`_prep_blockscaled`)、`:239`(`_fold_per_tensor_scales`) | 只收 `BlockScaledOperand` 容器;`(data, sf)` 元组在拆包点抛 TypeError(`:196-200`) |
| typed out-op 注册面 | `gemm_interface.py:1229` 起,共 14 个 | torch.compile 图边界,见下节 |
| eager 快路径 plan cache | `gemm_interface.py:889`、`:2634` | 命中即换指针重放,见「双层 plan cache」 |
| reference 实现 | `gemm_interface.py:1370`(`gemm_ref`)、`:1666`(`gemm_add_ref`)、`:2213`(`gemm_act_ref`)等 | 纯 torch,测试对照 |
| 变体样板引擎 | `gemm_iface.py:51`(`VariantSpec`)、`:102`(`canonicalize`)、`:232`(`make_iface_plan`)、`:319`(`run_variant`) | 见「VariantSpec」节 |

不做的事:不 import cutlass/cute(`:1-44` 的 import 区无一条),不调 `cute.compile`,不持 jit 缓存,不碰 TVM-FFI。到低层只有一条模块级 import 边:`gemm_interface.py:29` 的 `from cute_gemm.gemm import gemm as gemm_dispatch`,方向不反(见 [../architecture.md](../architecture.md)「import 方向」)。

epilogue 数学分两条路离开本模块:线性族直调 `gemm_dispatch`;激活/梯度/norm/rms 族经 `epilogue/library.py` 的 mod 工厂——`_run_activation_gemm:716`、`_run_activation_grad_gemm:814`、`_gemm_rms_call:3045`、`_gemm_norm_act_call:3296`,mod 的 plan/编译/启动归 `gemm_runtime/host.py`(背景见 [../notes/epilogue_transform_reorg.md](../notes/epilogue_transform_reorg.md))。

## 变体族与成员

四类 GEMM 入口(见 [../architecture.md](../architecture.md)「入口分类」)中,本模块持有 torch 函数与 typed out-op 两类;类入口在 `GemmBase`(`gemm_base.py:333` 的 `gemm_with_activation`),fn 算子收敛于 `dlkernel::gemm_epi`(`gemm_runtime/torch_op.py:93`)。

变体按族组织:

| 族 | 成员(行号) | 语义 |
|---|---|---|
| 基本 | `gemm`:892、`gemm_out`:1237、`gemm_quant_out`:1302 | α·A@B(±bias);`out_dtype` 传 `BlockScaledFormat` 时输出量化容器 |
| add | `gemm_add`:1480、`gemm_add_out`:1612、`gemm_add_inplace`:1852 | α·A@B + β·C;C 与 bias 是独立 epilogue 项,可同 launch 融合(`:1501-1504` docstring) |
| 激活 | 入口在 `GemmBase`;共用实现 `_dispatch_activation_gemm`:2041、`_dispatch_activation_grad_gemm`:2265;4 个 out-op | 前向与梯度;gated 与非 gated 共用入口,数学由 `activation` 字符串选择 |
| norm | `gemm_rms`:3191、`gemm_norm_act`:3437、`gemm_norm_gated_out`:3400 | RMS 统计融合(第二段归约 `rms_final_reduce`);norm×act |
| 特殊 | `gemm_symmetric`:2637 | 对称输出,独立 dispatch(`gemm_symmetric.py`),唯一的 VariantSpec 实例 |

激活族要点:gated 与非 gated 走同一入口,激活表在 `act_to_pytorch_fn_map`:82 与 `gated_to_pytorch_fn_map`:95;gated 输出宽度减半,`concat_layout` 声明 [gate; up] 列拼接(`:2093`、`_act_concat_bias:786`)。`kernel_variant` 令牌从类入口透传到 out-op schema(`gemm_act_out` 的 schema 见 `:2169`),供 MoE role 类复用同一组 op(见 [./fused_grouped_gemm.md](./fused_grouped_gemm.md))。

每族公共骨架(以 `gemm`:892 为例):blockscaled 拆包与 per-tensor scale 折叠(`:965-968`)→ eager 快路径探测(`:1009-1071`,命中即返回)→ 输出分配(`:1072-1098`)→ 空输入快路径(M/N=0 直接返回,K=0 退化算术,`:1118-1127`)→ 尾部分派(`:1128-1226`:量化输出走 `gemm_quant_out`,编译中走 `gemm_out`,eager 走通用路径)。

## typed out-op 注册面

14 个 `torch.library.custom_op`:12 个 mutating out-op,`gemm`/`gemm_add` 为 functional。

| op 名 | 注册行 | 类别 | op body 调 |
|---|---|---|---|
| `dlkernel::gemm` | `gemm_interface.py:1811` | functional | `gemm`(:892) |
| `dlkernel::gemm_add` | `:1762` | functional | `gemm_add`(:1480) |
| `dlkernel::gemm_out` | `:1229` | mutating | `gemm_tuned`(:377) |
| `dlkernel::gemm_quant_out` | `:1297` | mutating | 同上 |
| `dlkernel::gemm_add_out` | `:1604` | mutating | 同上 |
| `dlkernel::gemm_add_inplace` | `:1980` | mutating | 同上 |
| `dlkernel::gemm_act_out` | `:2165` | mutating | `_run_activation_gemm`(:716) |
| `dlkernel::gemm_gated_out` | `:2765` | mutating | 同上 |
| `dlkernel::gemm_dact_out` | `:2395` | mutating | `_run_activation_grad_gemm`(:814) |
| `dlkernel::gemm_dgated_out` | `:2815` | mutating | 同上 |
| `dlkernel::gemm_symmetric_out` | `:2503` | mutating | `_gemm_symmetric_execute`(:2601) |
| `dlkernel::gemm_rms_out` | `:3107` | mutating | `_gemm_rms_call`(:3045) |
| `dlkernel::gemm_norm_act_out` | `:3351` | mutating | `_gemm_norm_act_call`(:3296) |
| `dlkernel::gemm_norm_gated_out` | `:3394` | mutating | 同上 |

注册面的三条机制:

1. **schema 拆分**。torch.library 要求每个参数固定类型,union 参数(`alpha: float | Tensor`)在边界拆成 `alpha: float` + `alpha_tensor: Optional[Tensor]` 两个 schema 参数(`:2962-2967` 注释),eager body 用 `_merge_tensor`:2971 合回。`concat_layout` 元组在边界字符串化,`_parse_concat_layout`:2982 还原。e8m0 SF 以 uint8 视图过边界:`_sf_encode`:333 / `_sf_decode`:350。
2. **functional 门面**。`*_out` op 原地写调用方 buffer,FX pass 无法直接插图;`dlkernel::gemm`:1811 与 `dlkernel::gemm_add`:1762 在 op 内分配输出、带真 fake,给编译器一个干净节点(`:1750-1759` 注释)。只收稠密张量,varlen/blockscaled 调用方走 python 包装。
3. **eager 旁路**。custom op 边界每次调用约 85us(autograd 包装 + `mutates_args` 别名检查,`:173-177` 注释),eager 调用经 `_launch`:173 取 `op._init_fn` 裸实现,编译中才走 op 本体。fake 统一经 `_register_noop_fake`:2997 注册。

## VariantSpec:变体样板收敛为声明

`gemm_iface.py` 的 docstring(`:2-20`)记录了动机:act/gated、dact/dgated、norm_act/norm_gated、rms、symmetric 各变体曾手抄同四块——operand 规范化、tile-count-semaphore 规则、interface-plan NamedTuple + 模块级缓存 + warm 快路径、autotune 包装。`VariantSpec`:51 把这四块变成数据:

| 字段 | 行 | 含义 |
|---|---|---|
| `tensor_roles` | `:56` | `(名字, role)` 序列,role ∈ a/b/mn/row/col |
| `cold` | `:59` | 冷路径启动器,持有变体自己的 dispatch 调用 |
| `warm` | `:64` | warm 钩子,给 `warm_slots` 表达不了的变体(自定义 plan 层、逐调用 buffer) |
| `b_kn_rule` | `:69` | 默认 `sm90_plus and not swap_ab` |
| `dense_2d_ok` | `:72` | dense-2D 直通的变体否决权(act 在 concat_layout 下否决) |
| `semaphore` | `:76` | 默认 dynamic 且 capacity==9 时分配 (1,) int32 |
| `warm_slots`/`warm_epi`/`warm_extras` | `:84`/:87/:90 | 声明式 warm 映射:run_gemm_epi_plan 的四个槽位、epi 值端口、逐调用 kwargs |

### 四步机制

1. **canonicalize**(`gemm_iface.py:102`):按 role 规范化操作数,冷热两路共用(`:108-110` 要求除视图外零分配)。

| role | 规范化 | swap_ab 下 |
|---|---|---|
| `b` | 非 b_kn 则 `.mT`(`:130-132`);非 dense_2d 且 2D 补 batch 维(`:133-139`) | — |
| `a`/`mn` | 非 dense_2d 且 2D 补 batch 维(`:140-142`) | `mn` 再 `.mT`(`:151-152`) |
| `row` | 1D unsqueeze 成 (1, n)(`:143-145`) | 变 colvec(`vec_ports`:362) |
| `col` | 1D unsqueeze,varlen_m 保持 1D(`:146-148`) | 变 rowvec |

   dense_2d 判定在 `:114-123`:sm90_plus、非 varlen_m、`dense_2d_ok` 放行、a/b/mn 全 2D。A/B 互换由 `swap_pair`:356 给出。

2. **make_iface_plan**(`:232`):冷调用解析出 config 与 dispatch plan 后,把一切 metadata 派生决策定版——`_canon_view_codes`:193 把 canonicalize 的分支决策压成每操作数一个位码(`_MT_PRE`/`_UNSQ`/`_MT_POST`,`:188-190`),warm 重放只需 `_apply_code`:219 的三次分支测试;`warm_slots` 非空时铸出直调 `run_gemm_epi_plan` 的特化闭包(`:266-299`,函数内 import 断环见 `:267`),否则包一层 `warm` 钩子重放 Canon(`:301-308`)。semaphore 的有无是 metadata 静态的,record 时探一次(`:262-264`),无 semaphore 的重放跳过规则调用。

3. **冷路径**:`run_variant`:319 调 `spec.cold(canon, semaphore, config, dynamic_scheduler, ctx)`(`:353`),返回 dispatch plan 供 iface plan 捕获。

4. **热路径**:优先 `IfacePlan.replay`(`:334-335` docstring,零重推导);plan-less 的 warm 重放走 `spec.warm`(`:345-352`),声明了 `warm_slots` 的变体没有 `warm` 钩子,该路径直接抛错(`:347-350`)。输出按 `out_recipes` 由 `alloc_outputs`:177 分配。

### 现存实例:gemm_symmetric

全库唯一的 `VariantSpec` 实例是 `_SYMMETRIC_SPEC`(`gemm_interface.py:2588`);act/dact/norm/rms 各族的旧样板已移植到 epilogue-object surface(`:705-713`、`:806-811`、`:3038-3042`、`:3290-3293` 的移植注释)。对称变体的完整装配:

- 声明:roles `(A,a) (B,b) (out,mn) (C,mn)`;`b_kn_rule` 恒 False(`:2594`,symmetric dispatch 收 B 为 (m, k) 形);semaphore 规则 dynamic 且非 warm 才分配(`:2595-2597`)。
- 冷路径 `_symmetric_cold`:2543:按 SM 查 `_symmetric_gemm_config`:2488 定 tile/cluster/pingpong,调 `gemm_symmetric_dispatch`。
- warm 钩子 `_symmetric_warm`:2571:调 `run_gemm_symmetric_plan`。
- eager 入口 `gemm_symmetric`:2637:快路径键 `:2669-2683` 查 `_gemm_symmetric_iface_plan_cache`:2634,命中走 `alloc_outputs` + `plan.replay`(`:2686-2698`);冷路径经 `_gemm_symmetric_execute`:2601 调 `run_variant`;record 用 `make_iface_plan`:2754。

## 新增一个变体要写什么

与 [../architecture.md](../architecture.md)「扩展点清单」第一行对应,`gemm_symmetric` 是范例:

| # | 要写什么 | 对称变体的位置 |
|---|---|---|
| 1 | 公开 eager 函数:签名、输出分配、空输入语义 | `gemm_interface.py:2637`(空输入 `:2711-2715`) |
| 2 | `VariantSpec` 声明 + cold/warm 钩子 | `:2588` / `:2543` / `:2571` |
| 3 | 低层 dispatch 的 plan 构建与 `run_*_plan` | `gemm_symmetric.py`(与 `gemm.py` 同构的另一份) |
| 4 | typed out-op 注册 + fake | `:2503`、fake `:3016` |
| 5 | eager 快路径 iface plan cache | cache `:2634`、键 `:2669`、record `:2754` |
| 6 | torch reference | 多数变体有 `*_ref`(`gemm_ref:1370`、`gemm_act_ref:2213`);对称变体由测试内联构造 |
| 7 | 族级校验:blockscaled 拆包、split-K 支持检查 | `:2654-2664`、`_check_split_k_unsupported:134` |

【推断】第 3 步意味着新增变体还会复制一份镜像 plan cache(全库已四处:`gemm.py:261`、`gemm_interface.py:889`、`:2634`、`gemm_symmetric.py:217`);这是 architecture.md「潜在问题」第 4 条记录的既存代价。

## 双层 plan cache

| 层 | 缓存 | 键 | 命中后 |
|---|---|---|---|
| eager iface plan | `_gemm_iface_plan_cache`:889(`_GemmIfacePlan`:868,6 字段) | `:1017-1037`:张量元数据 + alpha/sr 模式 + tune_mode/dynamic/设备 | `_gemm_execute` 带 `dispatch_plan=` 直接 launch(`:1045-1070`),跳过校验、config 解析(含 autotuner)与 dispatch 层的键 |
| dispatch plan | `gemm.py:261` | `gemm.py:459-501` | `run_gemm_plan`(见 [./gemm_dispatch.md](./gemm_dispatch.md)) |

iface plan 的进入条件(`:1009-1016`):非 torch.compile、无 varlen/gather/permute/concat、非量化输出。record 点在冷路径末尾(`:1217-1225`),config 由 `gemm_tuned`:377 返回的已解析决策组装。

两层并存的原因写在 `gemm_tvm_ffi_utils.py:390-399` 的设计注释:两层缓存不同高度的决策,键空间不同。autotune 期间一个 interface 签名合法地驱动 N 个 dispatch plan(每个候选 config 一个);并成一层会在每次换 config 时重解析 kernel。配套不变式:外层键必须覆盖被捕获内层 plan 键的全部输入(`:393-394`),所以 interface 键带 alpha/sr 模式(`:1002-1007` 注释);warm 调用只付一次键(`:392`)。

【推断】`gemm_iface.py` 的 `IfacePlan`(`:157`)与 `gemm_interface.py` 的 `_GemmIfacePlan`(`:868`)是同一模式的两个世代:前者服务 VariantSpec 变体(out 配方 + 重放闭包),后者服务已移植到 mod 管线的线性族(捕获 dispatch plan 直达 `_gemm_execute`)。改 interface plan 语义时两处都要看。
