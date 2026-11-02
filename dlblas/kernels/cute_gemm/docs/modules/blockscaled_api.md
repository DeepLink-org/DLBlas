# 块缩放张量 API

本文说明块缩放（blockscaled）GEMM 的操作数与输出 API。类型化操作数容器（typed operand container）是 blockscaled（MXFP8 / MXFP4 / NVFP4 / MXFP6）GEMM 的唯一操作数形式。块缩放*输出* dtype 补全 (a, b, d) 格式三元组。旧 `(data, scale_factor)` 元组约定已移除（D10）。本设计与 torchao（`MXTensor`/`NVFP4Tensor`）、PyTorch 子类机制、PTX/CUTLASS fp6 约束对照。

## 1. 动机

操作数必须是类型化容器，理由有三：

1. 检测歧义。未来的压缩操作数类型（带元数据的 2:4 结构稀疏、LUT 索引）同样是 `(data, aux)` 对。`isinstance(X, tuple)` 无法区分它们。
2. 装不下 MXFP6。PyTorch 没有 fp6 dtype。裸张量表达不了 e2m3/e3m2 元素。"scale dtype 隐含向量长度"的约定（e8m0→32、e4m3→16）也表达不了新的 (scale dtype, vec size) 组合。
3. 元组形式丢 NVFP4 的 per-tensor scale。`gemm` 要求调用方手动把 `pts_A * pts_B` 折进 `alpha`。漏折会静默损失精度。容器由 `gemm` 在解包点自动折叠（D8）。

## 2. 设计决策（含理由）

每条决策都写明它阻止的失败模式。

| # | 决策 |
|---|---|
| D1 | **容器是不可微分的普通对象**——frozen dataclass，不是 `torch.Tensor` 子类。它在结构上进不了 autograd。量化操作数上的训练按设计不在范围内：前向量化的操作数对两个反向 GEMM 的缩放轴都是错的（dgrad 收缩 N，wgrad 收缩 M；而 scale 沿 K 分块），方阵权重情形（K==N）*静默*出错。任何训练方案都是 hp 主权重 + 每 GEMM 重量化，架在容器之上。 |
| D2 | **`shape` / `dtype` 报告逻辑 hp 视图**（解包形状、`orig_dtype`、默认 bf16），作为诚实的计算属性。存储真相在 `qdata`/`scale`。容器在每个公共入口顶部解包（`_unpack_operand` 所在处）；**这行以下的一切都看不到容器**——custom op、autotuner、`jit_cache`、TVM-FFI 全部拿到展平的普通张量。 |
| D3 | **转置 = qdata 步长交换视图；scale 原样保留。**`(rm, rk, 32, 4, 4)` 的分块 scale 原子不可视图转置；scale 定义为"任一朝向都沿 K 分块"。这保住了承重的 `weight.T` / `qw.mT` 惯用法。qdata 与 scale 的形状耦合在 GEMM 分发时校验（`validate_blockscaled_sf`），不在构造期——varlen 的 padding scale 缓冲必须始终可构造。 |
| D4 | **格式 = frozen dataclass 描述符**（`BlockScaledFormat`），不是枚举：存储 dtype、元素位数、打包方式、scale dtype、sf_vec_size、K 对齐都是*每格式的数据*。新格式是数据，不是代码。注册表 + 单例；旧短名（`"mxfp8"`）仍可转换。`elems_per_container` 保留原位置槽，以维持构造/pickle 契约；`storage_layout` 为它表达不了的布局（packed fp6）版本化。 |
| D5 | **格式以标量 `bs_format: str` op-schema 参数显式跨 custom-op 边界。**格式从不由张量 dtype 推导（没有 `SF_DTYPE_TO_VEC_SIZE` 表、没有 fp4 `A.dtype` 的逻辑-K 特例、没有 `gemm_blockscaled_ref` 的 sf-vec 三元分支）；`_sf_decode` 由格式驱动——这杀死了"uint8 scale 视图被解码成 e8m0 ⇒ NVFP4 静默跑成 vec-32 MXFP8"这类静默错误。全仓不存在 dtype→格式推断点。 |
| D6 | **MXFP6 存储名无歧义。**`mxfp6_e2m3` / `mxfp6_e3m2` 是每 fp6 元素占一个 `uint8` 的字节容器格式，含其量化器行为与序列化描述符（checkpoint 兼容与 host 侧转换用）。kernel 就绪的 `mxfp6_e2m3_packed` / `mxfp6_e3m2_packed` 用 `storage_layout="packed_lsb_v1"`：小端 6-bit 流，元素 i 位于比特 [6i, 6i+6)，即 CUTLASS SubbyteReference 布局（`pack_uint6`/`unpack_uint6`）。packed gmem 是 TMA unpack 消费的形式（第 6 节），跨 FFI 时是形状 `(..., 3K/4)` 的原始 `torch.uint8`。`to_packed()` 把字节操作数逐位不变地转为 packed。 |
| D7 | **响亮失败、显式表面。**不做 aten 拦截：torch op 对非张量容器抛普通 `TypeError`。支持的表面是显式方法（`mT`/`T`/`transpose` 限最后两维、`to(device)`——dtype 转换抛错并指向 `dequantize()`、`clone`、`dequantize`）。kernel 库里**没有 dequantize 兜底**。非块缩放入口（`gemm_symmetric`、`gemm_rms`、`gemm_norm_act`）有定向 `TypeError` 守卫，以改善报错信息。 |
| D8 | **`per_tensor_scale` 是字段**（仅 NVFP4）。`gemm` 在解包点经既有的 *tensor-alpha* 路径把 `pts_A * pts_B` 折进 alpha（无 `.item()` 同步；带 pts 的 nvfp4 编译 tensor-alpha kernel 变体）。 |
| D9 | **经 pytree 注册支持 torch.compile**，外加一个普通可内联构造 helper 用于图内量化。pytree 上下文是版本化、JSON 安全的编码，涵盖描述符每个字段、`orig_dtype` 与 `quant_dim`；自定义格式与旧格式都能活过 `treespec_dumps`/`treespec_loads`。不得用 `allow_in_graph`：那会让 Dynamo 把 helper 当不透明 op，其非 Tensor 的 `BlockScaledOperand` 返回值在 `fullgraph=True` 下非法。`_sf_encode`（e8m0→uint8 视图跨 op 边界，Inductor `decompose_auto_functionalized` 的变通）位于解包点。 |
| D10 | **容器是唯一的块缩放操作数形式**：`(data, scale_factor)` 元组/列表在解包点被拒，抛 `TypeError` 并指向 `BlockScaledOperand.from_parts`。因此每个块缩放操作数在到达 GEMM 前都经过构造校验。`blockscaled_quantize` / `blockscaled_quantize_dim0` 保留裸 `(q, sf)` 返回——它们是量化器，不是操作数构造器（改返回值会静默破坏 `qa, sfa = ...` 调用点）；经 `from_parts` 包装，或直接用 `BlockScaledOperand.quantize`。TVM-FFI 直连路径（`compile_blockscaled_gemm_tvm_ffi`）永远只收普通张量。 |
| D11 | **A 与 B 端到端携带独立格式**（`bs_format_a` / `bs_format_b` 穿过每一层，含 op schema）。配对*合法性*是硬件事实，由 `operand.py` 的 `mma_kind_for_pair` 拥有：它拒绝字节容器兼容格式，并对每个 kernel 就绪的非 nvfp4、非双 mxfp4 配对返回 `"mxf8f6f4"`（PTX：`mxf4nvf4` 与 `mxf4` 要求元素类型匹配；`mxf8f6f4` 自由混 fp8/fp6/fp4，子字节操作数由 packed gmem 经 TMA unpack 进 8-bit SMEM 容器）。配对*实现*按**架构**强制：每个 `gemm_smXXX` kernel 类对自己支持的 (A, B, SF, D) dtype 组合持有断言（`GemmSm100.is_valid_dtypes_and_scale_factor_vec_size`，在其 blockscaled 设置路径断言；不同 SM 版本支持不同 mx dtype）。SM100 的矩阵**允许混合 fp8 配对（e4m3 × e5m2，两个方向）**：DSL 的 `MmaMXF8F6F4Op` 接受独立 a/b dtype；指令 K 由 kind 推导（mxf4/mxf4nvf4 ⇒ 64 元素，mxf8f6f4 ⇒ 32），不随元素位宽。**含 fp4/fp6 的混合配对走 TMA-unpack 路径**（第 6 节）：packed gmem + `internal_type=Uint8` + 字节域 SMEM；贯穿两条编译路径的 `a/b_mma_dtype` 拆分承载 packed-fp6 的边界 dtype。`validate_blockscaled_sf` 检查每操作数 dtype，并经格式的 `logical_k` 映射交叉校验逻辑 K——两侧打包可以不同。 |
| D12 | **块缩放输出 D 是一等 API**：`gemm` / `gemm_add` / `gemm_with_activation`（含 `postact_dtype`）接受 `out_dtype: torch.dtype \| BlockScaledFormat \| str`；op-schema 管道是 `SFD`（可变更）+ `bs_format_d`，补全 (a, b, d) 格式三元组。见第 7 节。 |

### 被否决的替代方案

- 裸 `(data, scale_factor)` 元组操作数：问题见第 1 节（检测歧义、无 fp6、静默丢 NVFP4 per-tensor scale、还需要一个 dtype→格式推断点）。
- 依赖 torchao `MXTensor`：其 2-D 惰性 swizzle scale 不适配本仓库永久分块的 scale 原子与 varlen padding 缓冲。
- 基于枚举的格式：把按格式 switch 的重复抬升一层而已。描述符 dataclass 让新格式是数据，不是代码（D4）。
- `torch.Tensor` 子类容器：没有 `nn.Parameter`/`state_dict`/autograd 的需求要满足。没有这些时，wrapper 子类构造、`__torch_dispatch__` 表、traceable-subclass 协议与"说谎"的元数据相对 frozen dataclass 上的显式方法毫无收益。容器的字段集与构造器恰好就是 tensor 子类 wrapper 会持有的东西。模块公民身份之后可叠加，不破坏本 API。
- 新 GEMM 用字节容器 gmem 格式（每 fp4/fp6 元素一个 uint8 字节）：TMA unpack 直接消费 packed gmem（第 6 节）。字节容器只会让 gmem 流量翻倍（fp4）或膨胀 4/3（fp6）。字节容器描述符保留注册以兼容 checkpoint 与 host 侧转换。GEMM 拒绝它们并指向 `to_packed()`。

### 范围外

- `nn.Module` / `cute_gemm/linear.py` 集成与 autograd。容器按构造不可微分（D1）。一条 kernel 约束塑造任何未来的模块层：不存在混合 hp × 量化的 GEMM。两个操作数都必须带 scale factor。
- 训练。前向量化的操作数对两个反向 GEMM 的缩放轴都是错的（D1）。训练方案应保留 hp 主权重并按每个收缩轴逐 GEMM 重量化。对块缩放输出（第 7 节）同理：融合量化输出是 gemm + STE 量化。保存的量化输出沿 N 轴分块，对消费者的 wgrad 是错误的轴。训练要么保存 hp 激活，要么接受 dim1 重量化误差（标准 MX 权衡）。
- 用计算出的全局 amax 做 NVFP4 输出（需要延迟或两遍 amax；第 7 节要求显式提供 per-tensor scale）。

## 3. `BlockScaledFormat` 与 `BlockScaledOperand`（`cute_gemm/blockscaled/operand.py`）

```python
@dataclass(frozen=True)
class BlockScaledFormat:
    name: str                    # 含无版本字节名与显式 *_packed fp6
    qdata_dtype: torch.dtype     # e4m3 | e5m2 | float4_e2m1fn_x2 | uint8
    cutlass_dtype_name: str | None  # CuTe-DSL MMA 元素类型（可与存储不同：fp6）；None = 无 DSL 元素类型
    elem_bits: int               # 8 | 4 | 6
    elems_per_container: int     # 位置槽契约；fp4x2 为 2，其余为 1
    scale_dtype: torch.dtype     # float8_e8m0fnu | float8_e4m3fn
    sf_vec_size: int             # 32 | 16（也是逻辑 K 的最小整除）
    has_per_tensor_scale: bool   # 仅 nvfp4
    storage_layout: str | None   # *_packed fp6 为 "packed_lsb_v1"；None = 字节容器语义
```

- 可哈希、可 pickle。可用作 dynamo guard 上下文与 kernel-cache key 材料。只有 `name` 跨 op schema。
- `logical_k(storage_k)` / `storage_k(logical_k)` 在逻辑元素跨度与存储跨度之间映射。容器布局用 `elems_per_container`。版本化 packed-fp6 布局用"每 3 字节 4 元素"的比特比率，整数字段表达不了该比率。`is_packed` 与 `is_byte_container` 区分两条路径。`to_cutlass_dtype()`（惰性导入 cutlass：`Float8E4M3FN`、`Float8E5M2`、`Float4E2M1FN`、`Float6E2M3FN`、`Float6E3M2FN`）、`from_name`、`from_cutlass_dtypes`。dtype 三元组查找过滤字节容器，因此把 fp6 解析到显式 packed 描述符。
- kernel 支持不是注册表属性：描述符的注册与构造（checkpoint、往返）独立于任何 kernel。(A, B, D) dtype 组合能否执行由各 `gemm_smXXX` 类按架构断言（SM100 在 kind::mxf8f6f4 下接受任意 fp8/fp6/fp4 混合，见第 6 节）。
- E5M2：`quantize` 经 `to_mx(..., elem_dtype=torch.float8_e5m2)` 编码（fp8_max 57344，max_pow2 15）；`from_parts` 也接受预量化数据。
- `mxfp4_byte` 与无版本字节容器 MXFP6 checkpoint 加载后保留逻辑形状与反量化。它们只是 host 侧兼容格式。`BlockScaledOperand.to_packed()` 为 GEMM 逐位保留其码字/scale。
- pickle 反序列化按结构区分两种 schema：含 `elems_per_container` 的状态保持字节语义；缺两个打包字段的状态把分数 FP6 迁移到显式 `*_packed` 名。

```python
@dataclass(frozen=True, eq=False)        # 普通容器，不是 Tensor
class BlockScaledOperand:
    qdata: Tensor                        # 存储 dtype、packed 形状、K-/MN-连续
    scale: Tensor                        # (rm, rk, 32, 4, 4) 或 (L, ...)，原子步长 (16, 4, 1)
    format: BlockScaledFormat
    per_tensor_scale: Optional[Tensor]   # 标量 fp32，仅 NVFP4
    orig_dtype: torch.dtype              # hp 出处；由 .dtype 报告
    quant_dim: int                       # 量化轴标志（见下）
```

- 逻辑元数据是计算属性：`shape`（解包形状；packed 维 = qdata 的单位步长维——子字节 packed 格式按 kernel 要求 K-main，所以单位步长 == packed == K）、`dtype = orig_dtype`（hp 出处；存储真相是 `qdata.dtype`）、`device`、`ndim`。
- 量化轴 `quant_dim in {-1, -2}`：scale 向量沿哪个逻辑维走——CUTLASS `UMMA::Major` SF-atom 参数的直接对应物。单靠存储表达不了它：方阵 fp8 规范操作数与其 `.mT` 视图字节相同。子字节打包把它钉在 packed 维上（冲突请求抛错）。字节格式默认 -1（quantize 的约定）。转置视图翻转它。`quantize(dim=-2)` 与 `from_parts(quant_dim=-2)` 直接构造另一方向。随 pytree/pickle 携带。`dequantize()` 使用它。**接口按操作数槽强制"量化轴 = 收缩轴"**（A: -1，B: -2）：方阵形状的错误轴情形变成响亮的错误，而不是静默垃圾。
- 与 CUTLASS 交叉核对（sm100_blockscaled_layout.hpp）：CUTLASS 在 `Sm1xxBlockScaledBasicChunk` 里把同样两个量化方向建模为 `SfKMajorAtom` / `SfMNMajorAtom`——MN-major 原子就是 K-major 原子的模交换（转置），同一个 512 B 物理块。本仓库的 `quant_dim` 标志 + `.mT` 视图与该模板参数同构："MN 量化的 (M, K) 张量"与"K 量化的 (K, M) 张量的转置视图"字节相同。MMA *操作数*只消费 K 方向（`Sm1xxBlockScaledConfig`——每个 SM100 blockscaled mainloop 都用它——钉死 K-major chunk；tcgen05 按指令定义沿归约维施加 SF）。MN 方向服务于*输出*侧与非 GEMM 消费者——见第 7 节。
- 构造校验（仅模式无关项，D3，在 `__post_init__`）：qdata dtype 与格式一致。qdata 有单位步长维。scale ndim ∈ {5, 6}，内层原子 `(32, 4, 4)`、步长 `(16, 4, 1)`。scale dtype = `fmt.scale_dtype` 或其 uint8 视图（重视图为格式 dtype——规范化）。仅 `fmt.has_per_tensor_scale` 时可有 pts。不校验 qdata 与 scale 的形状耦合（varlen 缓冲）。
- 构造器：`from_parts(qdata, scale, format, *, per_tensor_scale=None, orig_dtype=bf16, quant_dim=-1)` 与 `quantize(x_hp, format, *, dim=-1, per_tensor_scale=None)`，都经一个普通 helper——Dynamo 在 `torch.compile` 下可内联穿过。显式 `dequantize()`；从不隐式。它要求规范的稠密 scale 形状：padding 的 varlen 缓冲需要容器不携带的 `cu_seqlens` 偏移，因此反量化拒绝它们，而不是把 padding 当真实 scale 静默应用。
- torch.compile / pytree：`register_pytree_node`，children 为 `(qdata, scale, pts)`，上下文为 `(format.name, orig_dtype, quant_dim)`。
- 序列化：原生 dataclass pickle；导入时 `add_safe_globals([BlockScaledOperand, BlockScaledFormat])`，支持 `weights_only=True` 加载。
- `repr` 只含元数据（不触碰存储）。

### 扩展接缝（前向指针）

描述符捆着两件正交的事：*元素编码*（qdata dtype、位数、打包、DSL 类型）与 *scale 配方*（scale dtype、`sf_vec_size` = `(1, sf_vec_size)` 块网格的 K 跨度）。其余轴——2D scale 网格、物理 scale 布局、无 DSL 类型的格式、按 kind 的单双侧性——在第 9 节设计、第 10 节加压测试。它们证成的加固（`cutlass_dtype_name` Optional；`mma_kind_for_pair` 里的显式元素类 + 配方门）已落地。

## 4. GEMM 接口集成（`cute_gemm/gemm_interface.py`）

- `_unpack_operand(X)` → `_Operand(data, sf, fmt, pts, quant_dim)` NamedTuple；接受 `BlockScaledOperand | Tensor`。tuple/list 抛 `TypeError` 并指向 `BlockScaledOperand.from_parts`。
- 支持块缩放的入口（`gemm`、`gemm_add`、`gemm_add_inplace`、`gemm_with_activation` 类入口——gated 与非 gated 都在内）：配对合法性经 `mma_kind_for_pair` 检查（D11）。scale 原子已在构造期校验（每个块缩放操作数都是容器）。解包点只做 `_sf_encode`。pts 折进 alpha（D8）。
- 两个操作数必须同为块缩放或同为普通：不存在混合 hp × 量化的 GEMM kernel，单侧 scale factor 被拒。
- custom op `dlkernel::gemm_out`、`gemm_add_out`、`gemm_add_inplace`、`gemm_act_out`、`gemm_gated_out`、`gemm_dact_out`、`gemm_dgated_out` 携带 `bs_format_a` / `bs_format_b`（`Optional[str]`；字符串参数无需 fake 侧改动；fake 保持 no-op——D11：A 与 B 格式独立）。op body 解析两个描述符。`_sf_decode(SF, fmt)` 按操作数、按格式驱动。
- `gemm_add` 的 add-to-output 路径直接传展平的已解包部件，不重建操作数容器。
- `gemm_blockscaled_ref`：接受容器。`sf_vec = fmt.sf_vec_size`。应用 `pts_A * pts_B`。
- 拒绝守卫（`TypeError`）：`gemm_symmetric`、`gemm_rms`、`gemm_norm_act`。

## 5. kernel 路径管道（`gemm.py`、`gemm_runtime/host.py`、`gemm_tvm_ffi_utils.py`、`blockscaled/utils.py`）

- `cute_gemm.gemm.gemm`（`gemm.py:378`）与 `gemm_with_activation` 类入口（实现共享于 `cute_gemm.gemm_interface` / `gemm_runtime`）接收 `bs_format_a`/`bs_format_b`。传入 SFA 时二者必填。kernel 级 API 与 custom op 一样显式携带格式。
- `validate_blockscaled_sf(A, B, SFA, SFB, ..., fmt_a=, fmt_b=)`（`gemm_tvm_ffi_utils.py:70`）：按描述符检查每操作数的 vec size / scale dtype / qdata dtype。经各格式 `logical_k(storage_k)` 映射做 A↔B 逻辑 K 交叉核对（没有 fp4 dtype 特例；fp6 天生正确）。只校验布局与一致性，dtype 组合的支持由 kernel 类按架构强制。activation 与 gated GEMM 走 `gemm_epilogue` 的共享 `EpiMod` 路径。`epilogue/frontend.py:1121-1127` 在经 `gemm_runtime/host.py:314` 的 `build_gemm_epi_plan` 建 plan 前执行同一校验。
- `_compile_gemm`（`gemm.py:49`）/ `_compile_gemm_epi`（`gemm_runtime/host.py:84`）的 cache key：`(sf_dtype, sf_vec_size, a_mma_dtype, b_mma_dtype)`，都在调用方从 `fmt` 推导。两条编译路径都按完整参数元组（含 MMA dtype）进入 `@jit_cache` key。因此两种 packed-fp6 编码即使共享 uint8 存储也各自成键。AB 的 cute 边界 dtype 来自 `torch2cute_dtype_map[A.dtype]`。packed fp6 时是 `Uint8`，而 `a/b_mma_dtype` 驱动 MMA 构造器与 dtype 合法性检查。fake 张量规格给 fp6 用字节跨度 K 符号：`div_for_dtype` 在位宽 6 下非整（96 字节 = 128 个逻辑元素）。
- `_blockscaled_format_of`（`blockscaled/utils.py:335`）是 `BlockScaledFormat.from_cutlass_dtypes` 的薄封装，返回该模块测试/基准生成器分支用的短名。该路径拒绝 packed FP6：它没有独立的 uint8 存储 dtype / FP6 MMA dtype 管道。统一 API 与 varlen API 有。
- `compile_blockscaled_gemm_tvm_ffi`（`blockscaled/utils.py:684`）：断言只收普通张量；docstring 写明该策略。

## 6. fp4/fp6/fp8 混合的子字节操作数

{mxfp8_e4m3, mxfp8_e5m2, mxfp4, mxfp6_e2m3_packed, mxfp6_e3m2_packed} 的任意 A/B 配对在 SM100 上跑 tcgen05 `kind::mxf8f6f4`——包括两侧都要 unpack 的 fp6 × fp4。双 fp4 跑单个更稠密的 `kind::mxf4nvf4` 原子，按格式的 scale 配置参数化（vec 32 e8m0 = mxfp4——PTX 把该实例拼作 `kind::mxf4`；vec 16 = nvfp4，对应 CUTLASS C++ 的 `SM100_MMA_MXF4_SS<..., VS>`）。nvfp4 只与自己配对（scale 配置是指令级的）。具体地，cute_gemm 的 SM100 kernel 子类化 DSL 的 `MmaMXF4NVF4Op` 以解除其写死的 vec 16，并对双 fp4 恒定构造它（`cute_gemm/sm100_utils.py:42` `make_blockscaled_trivial_tiled_mma`）。这是安全的：两个 DSL fp4 op 类构建相同的 MLIR 原子类型（无 kind 属性；后端从 vec size 推导 PTX 拼法）。

TMA-unpack 配方（对应 CUTLASS C++ 的 `SmemAllocType=uint8_t`）：每个 unpack 操作数——非"双 packed fp4"配对里的 packed 子字节 gmem 操作数：

- gmem 保留真实的子字节元素类型（`Float4E2M1FN` / `Float6E*FN`，packed）。向 `make_tiled_tma_atom_A/B` 传 `internal_type=cutlass.Uint8` 会置 `use_unpack`（cutlass DSL 的 `cutlass/cute/nvgpu/helpers.py:153-175`）并选择 `TmaDataFormat.U4_UNPACK_U8` / `U6_UNPACK_U8` tensormap（= `CU_TENSOR_MAP_DATA_TYPE_16U4/16U6_ALIGN16B`）。把 gmem recast 成 Uint8 会杀掉 `use_unpack`（那是 sm103 示例的 *packed*-smem mxf4 路径，另一套约定）。
- SMEM 侧一切是字节域（Uint8）：`make_smem_layout_a/b` 的 dtype、`MemRange` 存储、stage 预算。TMA 把每 16 个 packed 元素展开成 16 字节 smem 足迹（fp4 8B / fp6 12B packed 块 + 间隙），即每元素一字节足迹。没有低于 16B 的软件索引。
- Uint8 类型的 smem 张量直接喂 `make_fragment_A/B`。DSL 从不检查 fragment dtype 与 MMA dtype 的一致性。它按"位宽 × 步长"的乘积构建 smem 描述符，所以字节布局给出正确的 LBO/SBO（16 字节单位）。这里用 f4 类型的 `make_smem_layout_a(..., Float4E2M1FN)` 布局是**错的**——那是 SW64 packed-mxf4 约定，K 步长减半。子字节身份只经 tiled_mma 的 `a/b_mma_dtype`（指令描述符的格式位）到达指令。
- mbarrier tx 计数按 packed 字节（元素数 × 4 或 6 bit——间隙不写），对应 CUTLASS 的 `sizeof_bits<ElementA> x cosize(SmemLayoutA)`。布局进入字节域且 `a_dtype` 保持子字节后，cute_gemm 既有的 `size_in_bytes(a_dtype, a_smem_layout)` 一行自动做到。
- 约束（CUDA driver，cuTensorMapEncodeTiled）：unpack 操作数 K-major 且逻辑 K % 128（globalDim[0] 规则；也是 fp6 96B / fp4 64B 内跨度规则）、基址 32B 对齐、字节步长 % 32、boxDim[0] 恰 128 元素（SW128 字节域原子满足）。已强制：arg-spec k 整除 128（fp4 混合）、`validate_blockscaled_sf` 里的 fp6 K%128 断言、`GemmSm100.can_implement` 的 K%128 + 每操作数 K-major、32B 的 fake 参数/每次发射对齐检查（后者同时保护不含数据指针的缓存 plan）。

kernel 结构（`gemm_sm100.py`）：每操作数的 `a/b_unpack` 标志 + `a/b_smem_dtype`（unpack 时 Uint8）在 dtype 解析旁推导（`:748-752`）。其余要点：

- 字节域 smem 布局、存储与 stage 预算。
- A/B TMA 原子上 `internal_type=Uint8`。
- dtype 门接受任意混合的 fp8/fp4/fp6 MMA dtype（fp6 拷贝 dtype 可为 Uint8，见下面的 FFI 边界）。
- 16B 对齐 helper 对位宽 6 用 128 元素粒度（16*8//6 是分数）。

fp6 FFI 边界：torch 没有 fp6 dtype（也没有 `float6_x4` 式 packed dtype 让 fp6 像 fp4 骑 `float4_e2m1fn_x2` 那样搭车）。所以 packed-fp6 qdata 以形状 `(..., 3K/4)` 的原始 `torch.uint8` 跨 tvm-ffi——小端 6-bit 流，元素 i 位于比特 [6i, 6i+6)，CUTLASS SubbyteReference 布局（与 DSL 自带 fp6 填充 kernel 逐位相同）。arg spec 给字节跨度一个独立 sym（整除 96 = 128 元素；共享 sym 之间的 4/3 关系表达不了——逻辑 K 一致性在 host 侧检查）。`GemmSm100.__call__` 经 `_reinterpret_packed_fp6` 重解释（recast_ptr 到 `Float6E*FN` + 逻辑布局：K 跨度 ×4/3、K 步长保持 1、外层字节步长 ×4/3——行是完整的 96 字节组，故精确）。贯穿两条编译路径的 `a/b_mma_dtype` 就是在边界上标记 packed fp6 的东西。

## 7. 块缩放输出的 SF 生成 epilogue

其余 kernel 的 `sf_*` 管道都在输入侧。块缩放 D 补上输出侧。

状态：`gemm` 已交付（`cute_gemm.epilogue.quantize_out.BlockScaleFactorStore` + `dlkernel::gemm_quant_out`；tests/test_gemm_quant_out.py）。`out_dtype` 接受格式名。`out_quant_dim=-1|-2` 是方向旋钮（-2 仅 fp8：fp4 沿 N 打包，收缩 M 的消费者用不了）。调用返回 `BlockScaledOperand`。gemm_add / act-mod 的 postact 量化仍经 `_reserve_blockscaled_out` 拒绝。两个方向都跑在 SM100/SM110 与 SM120（行方向：三种格式、普通与块缩放输入、k- 与 n-major B；列方向：fp8 数值）。SM120 上 vec-32 行 SFD 把 tiled-MMA 的 warp 连续 N 列跑宽到 32（`GemmSm120.mma_n_warp_run`，`:504`；epilogue 重的形状上比交换高约 1.5%），使 SF 向量保持 warp 局部。当 tile_n 不是 `32 * atom_n` 的倍数、以及列方向恒定如此（其 32 行向量跨两个 16 行 warp M 条带）时，epilogue 改用每子 tile 的 smem amax 交换 + 一个 epilogue barrier（见 BlockScaleFactorStore）。gated postact（swiglu FC1）量化同样跑在 SM120：减半的 postact 存储保持 16 列 warp 跑（其 dummy-MMA STSM retile 契约编码了这一点；窄 dtype 在转换前按 fp32 粒度施加同一寄存器 permute——`layout_utils.permute_gated_Cregs_f32`），量化骑在交换上。

实测（RTX 5090，2026-08-05，benchmarks/benchmark_gemm.py --quant_out，tile 128,128 pingpong dynamic-persistent，boost 时钟）：SFD epilogue 免费——mxfp8-in 8192³ 在 mxfp8 输出下 728 TF，bf16 输出 726 TF（fp8 D 把存储流量减半，抵消 SF 工作）。对 cuBLASLt 的 D-out-scale 路径（即 CUTLASS C++ cutlass3x_sm120_bstensorop kernel + Sm120BlockScaleFactorRowStore，JIT 分发）：cute_gemm 在 mxfp8/nvfp4、4096³/8192³ 上快 1.02–1.07 倍（如 nvfp4 8192³：1495 对 1435 TF），且数值与 SF 字节与 cuBLAS 逐位相同。cublasLt 没有 mxfp4-out 基线（fp4-D + e8m0 out-scale 的 AlgoGetHeuristic NOT_SUPPORTED）；cute_gemm 按 nvfp4 速率跑。

- API（D12）：`out_dtype` 在 `gemm`/`gemm_add`/`gemm_with_activation` 上接受 `BlockScaledFormat`（或名）。普通块缩放输出的默认保持 bf16。格式请求使 `gemm` 分配 qdata + 分块布局的 SFD 缓冲并返回 `BlockScaledOperand`。`out=` 接受容器（在操作数接缝解包）。op 携带 `SFD: Optional[Tensor]`（在 `mutates_args` 里）+ `bs_format_d`，补全 (a, b, d) 格式三元组。`_sf_encode` 应用于 SFD。
- 语义：沿最后一维（N）量化，即下一个 GEMM 的收缩维。N 连续的输出自动成为消费者的 K-major 操作数（fp4 要求的）。仅 MX 格式（单遍每 tile amax）。NVFP4 输出要求显式提供 per-tensor scale（全局 amax 单遍算不出）。
- SFD 方向是参数，不是常量（与 CUTLASS 交叉核对）：`Sm1xxBlockScaledOutputConfig<SFVecSize, major>` 同时支持 vec 沿 N（Major::K，D 作为下一个 GEMM 的 A 消费）与 vec 沿 M（Major::MN，D 作为下一个 B / dgrad 式消费者）。SM100 epilogue 的 SFD 存储实例化两者，由请求的 SFD 布局标签选择（另见 examples/91_fp4_gemv 的 `kIsKMajorSFD`）。因此 API 在 `bs_format_d` 旁携带方向旋钮（默认：vec 沿 N），产出带对应 `quant_dim` 的 `BlockScaledOperand`。
- kernel：CUTLASS `LinCombBlockScaleFactor` 先例——对累加器子 tile 做每 32 块 amax，e8m0 scale 用与 `to_mx` 相同的取整规则（对 `to_mx(gemm_ref(...))` 逐位可测）、除法 + 转换、写 qdata + 512 B SF 原子。`tile_n % 64` 已强制。要求 `split_k == 1`（量化只在完整 K 归约后有效）。hp 输入 GEMM 也可用（bf16 × bf16 → mxfp8 是网络的第一级量化）。`gemm_with_activation` 组合（量化 postact，gated 与否）是 MLP 前向融合的收益点。

## 8. 测试

一切数值上都对照反量化参考验证，绝不只查形状或冒烟。

- 容器（`tests/test_blockscaled_operand.py`）：构造拒绝（dtype/格式不匹配、坏原子步长、非 nvfp4 带 pts）、uint8 scale 规范化、varlen padding scale 构造；`clone`/`.to(device)`（dtype-`to` 抛错并指向 dequantize）；`.mT`/`.T`/`transpose` 往返复用同一个 scale 对象；torch op 的响亮 TypeError；repr；deepcopy；save/`load(weights_only=True)`；pytree 往返（含朝向标志），含自定义与旧格式的 JSON TreeSpec 序列化；非块缩放入口的拒绝守卫。
- 接口（`tests/test_gemm_blockscaled_interface.py`）：dtype 矩阵跑在容器上（唯一操作数形式）；一条负测试钉死 tuple/list 的 `TypeError`；回归测试编码本设计要阻止的失败模式：pts 折叠（非单位 pts、默认 alpha ⇒ 匹配反量化参考）、方阵权重朝向（K==N 的 `.mT`）、uint8 scale 视图解码（nvfp4 保持 vec-16）、混合格式拒绝；一条 e5m2 from_parts 运行。
- 编译：容器作为 fullgraph 输入（属性读取 + e8m0/Inductor 回归）；图内 `quantize` + gemm fullgraph；一个用裸 (q, sf) 图输入的图内 `from_parts` 编译测试。
- 混合配对：代表性的 fp8/fp6/fp4 A/B 组合对照"反量化操作数"参考验证，容差与 fp8 配对基线相同，含两侧都 unpack 的 fp6 × fp4。packed 6-bit 流与 DSL fp6 填充 kernel 逐位核对。
- 数值参考：`gemm_blockscaled_ref` 的反量化 matmul，外加经 `scale_blocked_for_cublas` 的逐位 `torch._scaled_mm` 对照。

## 9. 向 scale 配方、布局与软件 kind 泛化

把操作数 API 扩展到 MX/NVFP4 微缩放之外的设计。消费者在途：SM90 块升级 GEMM（sm90kscale：fp32 scale、`kscale_block ∈ {128, 256}`、单侧 bf16）、SM100 线性 SF 的 cp.async 加载器（sfcpasync：每操作数 `sfa_linear`/`sfb_linear`）、SM90 weight-only 家族（sm90w4：W4A8 / W4A16 / W8A16）、DeepSeek 式 fp8 配方（激活 1x128、权重 128x128）。泛化随其首个消费者 kernel 落地。本节只有接缝。

### 9.1 模型

块缩放操作数表示

```
X[i, j] ~= decode(qdata)[i, j] * S[i // bm, j // bk] * (pts if present)
```

- Level-0 scale 网格：逻辑 `(MN, K)` 索引空间上的块形 `(bm, bk)` → 网格 `(ceil(MN/bm), ceil(K/bk))`，任意 scale dtype。
- Level-1：可选的每张量标量。缩放是层级。恰好支持两层，level 1 固定为每张量 fp32（有意封顶，同 D1 的口吻）。
- 元素编码正交，且包含全精度类型：bf16 元素 + 每 K 块 fp32 scale（kscale）是合法操作数。容器关乎*块缩放*。量化只是窄元素类型的特例。

配方轴是生态自己的分类学，拆成字段而非枚举。cuBLASLt 的 `cublasLtMatmulMatrixScale_t`（`SCALAR_32F`、`VEC16_UE4M3`、`VEC32_UE8M0`、`OUTER_VEC_32F`、`VEC128_32F`、`BLK128x128_32F`、`PER_BATCH_SCALAR_32F`——经 `A_SCALE_MODE`/`B_SCALE_MODE` 每操作数独立设置，即 D11 也是他们的设计）与 torch 的 `_ScalingType`（`TensorWise, RowWise, BlockWise1x16/1x32/1x128/128x128`）都是同一批 `(bm, bk, scale dtype)` 点的平铺枚举。

| 配方 | bm | bk | scale dtype | 元素 |
|---|---|---|---|---|
| mxfp8 / mxfp4 / mxfp6_*_packed | 1 | 32 | e8m0 | fp8/4/6 |
| nvfp4 | 1 | 16 | e4m3 (+pts) | fp4 |
| DeepSeek 1D（激活） | 1 | 128 | fp32 | e4m3 |
| DeepSeek 2D（权重） | 128 | 128 | fp32 | e4m3 |
| kscale bf16 | 1 | 128/256 | fp32 | bf16 |
| int4-g128（AWQ/GPTQ，W4A16） | 1 | 128 | bf16 | uint4b8 |
| rowwise（每 token / 每通道） | 1 | K（全长） | fp32 | fp8/int8 |

### 9.2 三个轴与三个家

| 关注点 | 落在 | 跨 op 边界为 |
|---|---|---|
| 元素编码 + scale 配方 | `BlockScaledFormat` | `bs_format_{a,b}` 名 |
| 物理 scale 布局 | `BlockScaledOperand` | `bs_layout_{a,b}` 名 |
| 朝向 | `BlockScaledOperand.quant_dim` | （既有） |

格式是交换契约（数值），布局是存储细节（同 stride）。同一格式合法地存在于多个布局（tcgen05 的 mxfp8 分块、融合量化生产者的 mxfp8 线性）。所以布局按操作数（kernel 独立接收 `sfa_linear` / `sfb_linear`）且永不属于格式身份。布局从不推断：分块（5/6-D）对线性（2/3-D）今天可以嗅 rank，但那是重蹈 D5 要堵的路径，第一个同 rank 布局就破。

### 9.3 全部带默认的描述符改动（既有格式零改动）

- `sf_block_mn: int = 1`——块网格的 MN 跨度（2D 为 128）。
- `sf_vec_size: Optional[int]`——`None` = 完整收缩跨度（rowwise；cuBLASLt `OUTER_VEC`）。修订于初稿：由 w4a8 的每 token 激活 scale 与 w8a16 的每通道权重 scale 逼出——格式描述符与形状无关，写不了字面 K。
- `canonical_scale_layout: str = "mx_blocked"`——`quantize()` 产出什么、未指定布局的操作数意指什么。已定：MX 格式保持分块。"linear" 是被接受的操作数选项。fp32 scale 配方规范为线性（它们没有分块形态）。
- 新注册表行随其消费者 kernel 落地（从不投机）。见 `BLOCKSCALED_FORMAT_REGISTRY` 旁的注释示例与 `test_future_recipe_examples`。
- `quantize()` 策略：fp8 配方获得通用 `(bm, bk)` amax 量化器（e8m0 保持取整规则）。bf16 与 int-group 格式抛错——浮点相对精度对 scale 不变，group scale 是上游结构性的（GPTQ/AWQ 组、折起的 norm）。所以 `from_parts` 是构造路径。

### 9.4 scale 布局（每操作数）

`BlockScaledOperand.scale_layout: Optional[str]`（`None` → 格式的规范布局；由 `_view`、pytree 上下文、pickle 携带）。

| 标签 | 形状 | 约束 |
|---|---|---|
| `"mx_blocked"` | `(L?, rm, rk, 32, 4, 4)`，原子步长 `(16,4,1)` | 要求 `bm == 1` 且 e8m0/e4m3 scale（它就是 tcgen05 原子） |
| `"linear"` | `(L?, ceil(MN/bm), ceil(K/bk))`，K 块维连续 | 生态名（FlashInfer/TRT-LLM `layout_linear`；torchao "plain"；scaled_mm NO_SWIZZLE） |

构造校验变成按标签的分发，未来标签只增不改。`repack_scales(layout)` 负责转换（linear→blocked 就是既有的 `pack_scale_2d_to_blocked_contig`；量化器内部本就先产出线性网格，所以线性规范的 `quantize` 更便宜，不是更贵）。`quant_dim` 与 `.mT` 携带 scale 的手法不动——网格轴经 `quant_dim` 解释，D3 免费泛化到 2D 网格。

### 9.5 kind 与单双侧性函数 `gemm_kind_for_pair`（设计）

取代 `mma_kind_for_pair`（后者保留并被委托，负责硬件子集——它已同时门元素类与配方，所以软件配方会在 kind 选择处带着原因失败）。两侧都收 `Optional` 格式。单双侧性按 kind，不是接口不变量。

```
(bs, bs)  hw recipe on both (bm==1, bk in {16,32}, e8m0/e4m3, DSL elem)
    -> mxf4nvf4 | mxf4 | mxf8f6f4          # 既有规则，不变
(bs, bs)  fp8 elements, fp32 scales         -> blockwise_promote  # 1d1d / 1d2d
(bs, None) bf16 elements, fp32 K-block scales -> blockwise_promote  # kscale
(None, bs) plain acts x group-scaled weights -> upconvert (W4A16/W8A16 family)
(None, None)                                 -> the ordinary GEMM path
```

配对规则（修订）：阻塞 K 的两侧必须在 `bk` 上一致（升级区间必须对齐）。K 不变量 scale 自由组合——rowwise 与每张量 scale 完全逸出 K 循环（`out[m,n] = sfa[m]*sfb[n]*sum_k ...`），在 epilogue 或升级末尾施加。所以 w4a8 合法地把 rowwise fp8 激活与 group-128 int4 权配对。单侧配对的未缩放侧用普通 Tensor（已定——不做恒等 scale 包装：容器的契约是"携带块 scale"）。scale 在寄存器解码（W4A16 nvfp4：2 位尾数 × e4m3 的 4 位恰好精确放进 bf16）、k-tile 升级（w4a8、DeepGEMM）还是 epilogue（rowwise）施加，按架构实现，对 API 不可见。整数元素（e0m3/nvint4 = int4 + 配方）经 tcgen05 `kind::i8` 获得整数累加升级变体——精确 int32 内积，scale 在升级处。

按架构的门拥有支持矩阵，同 D11：SM100 tcgen05 kind + 线性加载器约束（`K/vec % 4 == 0`；`sfa_linear` 要求 cluster N == 1）；SM90 升级/升转换矩阵（`bk ∈ {128, 256}`、`split_k == 1`、无 `gather_A`——kernel 既有断言，由描述符前置）。

### 9.6 host 管道

`gemm()` 签名不变——一切骑在操作数上。op schema 增加 `bs_layout_{a,b}` 字符串（同 `bs_format_{a,b}` 模式）。tune 与编译 key 增加布局标签。kernel 标志由操作数推导（`sfa_linear = (opA.scale_layout == "linear")`、`kscale_sfb_1d = (fmt_b.sf_block_mn == 1)`……）。D8 先例延伸：*解释操作数字节*所需的 scale 随操作数走。W8A16 GEMM 之后忘乘一个 `(N,)` 每通道向量，就是 pts 折叠那类静默精度问题的重演。容器正是杀掉那类 bug 的东西。

### 9.7 预处理操作数不是格式（一道栅栏）

weight-only kernel 离线把权重重排进 kernel 私有 blob（WGMMA fragment 的 nibble 洗牌、`(N/64, K/64, 32, 4, 2)` 式 SF 条带）。它们绝不能进格式/布局词汇：它们不是交换形态（没有第二个消费者能读），且随 kernel 版本化。它们是第三层——容器之上的 prepare-and-cache，按 `(qdata 存储身份, kernel key)` 键控。权重静态，一次性成本可摊销。可序列化的是规范容器。prepared blob 加载时重建——这正是 kernel 布局改动不破坏兼容的原因。先例：torch.ao prepacked 参数、TRT-LLM `preprocess_weights_for_mixed_gemm`。

### 9.8 决策与顺序

`sf_vec_size` 保留其名（`sf_block_mn` 是除全长哨兵外唯一的新配方字段）。MX 规范布局保持分块。单侧 = 普通 Tensor。bk=256 行、ue8m0-scale 的 DeepSeek 变体、全部注册表行都在首次需求时随消费者 kernel 落地（sm90kscale / sfcpasync / sm90w4 集成——谁先合入谁携带描述符泛化）。

## 10. 压力测试（演算实例）

这张表是该抽象的回归套件。每个新格式提案都对照它检查。提案要么落成一行数据（好），要么逼出一次最小的、有名字的修订（记录在案）。七个对抗用例产出两条单行修订。可执行版本：`test_future_recipe_examples` / `test_format_without_dsl_element_type`（`tests/test_blockscaled_operand.py:736` / `:709`）。

| 用例 | 考察轴 | 逼出的改动 |
|---|---|---|
| e3m4（无 DSL 类型；捆绑 MLIR 有 `f8E3M4`，但 DSL 的反向分发与 dtype 表没有） | 无 DSL 类型元素 | `cutlass_dtype_name` Optional + 元素类门（已落地） |
| e0m3 / nvint4（= int4 元素 + nvfp4 配方；MLIR `i4`、DSL `Int4`、torch.int4 都在） | 整数元素 | 无——经 `kind::i8` 的 int 升级是 kind，不是描述符改动 |
| DeepSeek 1x128 / 128x128 fp32 scale | 配方：scale dtype + 2D 网格 | `sf_block_mn`（后续字段） |
| kscale bf16 + 每行 fp32 K 块 scale | 全精度元素；单侧 | 词汇（"量化" → "块缩放"）；单双侧性移入 kind 规则 |
| sfcpasync 线性 SF `(MN, K/vec)` | 物理布局 | 每操作数 `scale_layout` 标签（后续字段） |
| W4A16 nvfp4 权重 × bf16 激活 | 同一格式、新配对 | 无——SM100 tcgen05 操作数与 SM90 升转换权重是**同一个**格式 |
| W4A8 / W8A16（rowwise 激活 / 每通道权重） | 全长 bk；混合配方配对 | `sf_vec_size` 哨兵；配对规则放宽为"K 不变量自由组合" |

已知未建模（有意，有客户再加）：每组零点（第三个块网格张量；uint4b8 的 FIXED 偏移是元素解码，零成本）与每 L 批 scale（cuBLASLt `PER_BATCH_SCALAR_32F`——第三个网格跨度）。
