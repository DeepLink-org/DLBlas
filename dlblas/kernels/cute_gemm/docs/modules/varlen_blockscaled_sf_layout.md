# 变长 blockscaled SF 布局（按 batch 对齐 tile 的填充）

本文说明 cute_gemm SM100 blockscaled GEMM 如何存放 scale factor（SFA / SFB）。
适用条件：变长维（`varlen_m` 的 M、`varlen_k` 的 K）上各专家的长度不必是 128 的
倍数。不填充时，SF tile（覆盖 128 个源行/列）会跨专家边界。填充后，每个专家的
SF 区从一个 128 对齐的 tile 边界开始。kernel 按 batch 施加偏移，从单一统一
buffer 读取 scale。

本文称该方案为“按 batch 对齐 tile 的填充”（tile-aligned per-batch padding）。

适用范围：

- `varlen_m`：支持 MXFP8 / MXFP4 / NVFP4。fp4 的 K-major operand 要求与
  `varlen_m` 的一致（A 必须 K-major）。按 batch 的偏移按整 M 行切分。该切分保持
  字节对齐（每行 `k/2` 字节）和 TMA 对齐（`k % 32`）。
- `varlen_k`：仅支持 MXFP8。它需要 M/N-major operand，与 fp4 的 K-major 要求
  冲突。按专家的 K 偏移还会落入子字节打包。每专家 `k_i` 任意，连 `sf_vec_size`
  （32）对齐都不要求。未对齐的 `k_i` 使该专家最后一个 scale block 覆盖不完整的
  chunk。ragged 值 TMA 在 `cu_seqlens_k[i+1]` 之外零填充，尾部贡献恰为 0。

## SF pad 与 ragged K

K 是收缩维。kernel 会加载每个专家最后一个 SF atom 列内的 pad：SF TMA 加载
整个 512 B atom，不受 ragged 边界限制。A/B 的加载受该限制。这些 pad 会与零填充
的值尾部配对。若 pad 是 `0xFF`（e8m0 的 NaN），`0 × NaN` 会毒化 accumulator。

kernel 跳过 pad block 的 MMA 指令，规避该问题。对 mxfp8，一条 tcgen05
blockscaled MMA 指令恰好消费一个 SF k-block：

```
inst_k = 256 bit / 8 = 32 = sf_vec_size
```

因此 ragged batch 的最后一个 k-tile 上，mma 循环只发射
`ceil((len_k % 128) / 32)` 条有效指令（见 `GemmSm100.mma` 的 `sf_valid_insts`）。
被跳过指令的 A/B 值由 TMA 零填充，贡献为 0。这不改变任何数值。tmem 里的 pad
scale 从不被引用。指令发射是 leader-only 的决定。2-CTA MMA（tile_M 256）因此
天然覆盖，不需要额外同步。实测代价约 0：指令跳过在 ragged 尾部严格减少 MMA 工作。
结论：gmem SF pad 可以是任意字节，`torch.empty` 的 buffer 即可。

补充四点：

1. 备选方案：MMA warp 在 UTCCP 前清零 smem 的 pad 字节（每个 scale word 一次
   32-bit RMW，加 `fence.proxy.async.shared`）。该方案同样近乎零开销，但盖不住
   2-CTA：UTCCP 从两个 CTA 的 smem 取 SF，只有 leader 等待 stage。因此不采用。
2. ragged SF TMA 不是替代方案。ragged-extent 手法按整数 atom（128 源 K）粒度
   截断。危险字节在最后一个 atom 内部。
3. 指令跳过要求 `inst_k == sf_vec_size`。mxfp4 / nvfp4（inst_k 跨 2 或 4 个 SF
   block）维持零初始化 pad 的契约。`varlen_k` 仅支持 MXFP8，不受此点影响。
4. `varlen_m` 没有这个问题。`varlen_m` 的 M-pad 垃圾落在 epilogue 从不存储的
   输出行里。

## 记号

| 记号 | 含义 |
|---|---|
| `L` | 专家 / batch 数 |
| `m_b` | 专家 `b` 沿变长维（M 或 K）的长度 |
| `c_b` | `cu_seqlens[b] = Σ_{j<b} m_j`，排他前缀和；`c_0 = 0`，`c_L = total_m`（或 `total_k`） |
| `rm_b` | `⌈m_b / 128⌉`，专家 `b` 的 SF 沿变长维占的 tile 数 |

128 = MXFP8 的 `sf_vec_size * 4`。一个 512 字节 SF tile 覆盖 128 个源行 × 4 个
scale 列。

## 格式

SF 是单一统一 buffer。变长维经过填充，每个专家从 tile 边界开始。分配按
`off[L]` 个 tile 计大小，即“假想的下一个 batch”的起始位置。kernel 用同一个公式
索引它。

### 每专家 tile 偏移

```
off[b] = c_b // 128 + b          # in tile units
```

等价形式是 `((c_b + b·128) // 128 * 128) / 128`，即“把移位后的前缀和向下取整到
tile 边界”。但该式省去了 `// * *` 的往返。

专家 `b` 占用 tile `[off[b], off[b] + rm_b)`。

### 分配大小（tile 单位）

```
total_padded_rm = ⌈total_m / 128⌉ + (L − 1)
```

充分性见下方证明 2（“更紧的替代”）。`total_m` 是 128 的整数倍时，该形式零浪费。
其他情况等于 `total_m//128 + L`。字节大小为 `total_padded_rm × rk × 512`。

### Torch 存储形状

```
SFA: (1, total_padded_rm, rk_const, 32, 4, 4)
SFB: (1, rn,              total_padded_rk, 32, 4, 4)   # for varlen_k
```

`rk_const` / `rn` 是非变长维的 tile 数。内层 `(32, 4, 4)` atom（512 字节，
stride `(16, 4, 1)`）是硬件固定的 swizzled tile（见 `BlockScaledBasicChunk`）。

## 正确性：两个证明

### 证明 1：相邻专家不重叠

需要证明 `off[b] + rm_b ≤ off[b+1]`。设 `c_b = 128q + r`（`0 ≤ r < 128`）、
`m_b = 128p + s`（`0 ≤ s < 128`）：

```
(q + b) + ⌈m_b/128⌉  ≤  ((c_b + m_b) // 128) + (b+1)
⇔  ⌈m_b/128⌉  ≤  (r + m_b) // 128 + 1
```

| case | `s` | `r + s` | `⌈m_b/128⌉` | `(r+m_b)//128 + 1` | verdict |
|---|---|---|---|---|---|
| A1 | `= 0` | — (`r==0`) | `p`   | `p + 1`   | slack 1 |
| A2 | `= 0` | — (`r>0`)  | `p`   | `p + 1`   | slack 1 |
| B1 | `> 0` | `< 128`    | `p+1` | `p + 1`   | **tight** |
| B2 | `> 0` | `≥ 128`    | `p+1` | `p + 2`   | slack 1 |

每种情况都有 LHS ≤ RHS。无重叠。∎

### 证明 2：分配足够（`⌈total_m/128⌉ + (L−1)`）

需要证明 `off[L−1] + rm_{L−1} ≤ ⌈total_m/128⌉ + (L−1)`。两边消去 `L−1`，设
`c = c_{L−1} = 128q + r`、`m = m_{L−1} = 128p + s`、`total_m = c + m`。问题化为
证明 `q + ⌈m/128⌉ ≤ ⌈total_m/128⌉`：

| case | `s` | `r + s` | `⌈m/128⌉` | `⌈total_m/128⌉` | verdict |
|---|---|---|---|---|---|
| A1 | `= 0` | `r==0` | `p`   | `q+p`   | tight |
| A2 | `= 0` | `r>0`  | `p`   | `q+p+1` | slack 1 |
| B1 | `> 0` | `< 128`| `p+1` | `q+p+1` | tight |
| B2 | `> 0` | `==128`| `p+1` | `q+p+1` | tight |
| B3 | `> 0` | `>128` | `p+1` | `q+p+2` | slack 1 |

每种情况都有 LHS ≤ RHS。分配足够。∎

### 更简单的替代（未采用）

`total_m // 128 + L`（等于 `off[L]`，假想的下一个 batch 起点）也是合法上界。对
证明 1 迭代可得 `off[L−1] + rm_{L−1} ≤ off[L]`。`total_m % 128 > 0` 时该上界等于
`⌈total_m/128⌉ + (L−1)`。`total_m` 是 128 的倍数时该上界大 1 个 tile。LLM 负载常见
128 对齐的 `total_m`（prefill、批量训练），因此选更紧的形式。

## 最小性：还能少填充吗

三个层级，从当前方案到信息论下界。

### 层级 1：给定 `off[b] = c_b // 128 + b` 时分配恰好取紧

`⌈total_m/128⌉ + (L−1)` 可以取到，例如 `m = [1, 1, …, 1, rest]`（每个边界都是
证明 1 的紧 case B1）。`cu_seqlens` 在设备上，host 侧的分配和校验只能用
`(total_m, L)`。对所有切分取最大，就是只看形状的上界能达到的最好结果。据此，
`validate_blockscaled_sf` 要求 `SFA.shape[1] ≥ ⌈total_m/128⌉ + (L−1)`。它拒绝按
具体切分更小的分配，因为校验该分配需要 host 上有 seqlen 值（引入同步）。

### 层级 2：只用 `(c_b, b)` 可算的偏移距最优不超过 ⌈(L−1)/128⌉ 个 tile

这类方案的最优是 `off'[b] = (c_b − b) // 128 + b`。

- 合法（无重叠）：设 `c_b − b = 128q + r`、`m_b = 128p + s`。无重叠条件化简为
  `q + p + [s>0] ≤ q + p + ⌊(r + s − 1)/128⌋ + 1`，恒成立。`s > 0` 时
  `r + s − 1 ≥ 0`。`s = 0` 且 `r = 0` 时取等（`⌊−1/128⌋ = −1`）。
- 最优（下界）：任何合法的 `f` 对每个模式满足
  `f(c_{j+1}, j+1) ≥ f(c_j, j) + rm_j`。因此对 `c` 的任意 `b` 份切分都有
  `f(c, b) ≥ Σ_{j<b} rm_j`。对抗性切分取每份 `≡ 1 (mod 128)`，得
  `f(c, b) ≥ ⌊(c − b)/128⌋ + b`，恰是 `off'`。
- 与当前方案的差距：`c//128 − ⌊(c−b)/128⌋ ≤ ⌈b/128⌉`，即每 128 个专家至多
  1 个 tile。分配 `⌊(total_m − L)/128⌋ + L` 在 `L ≤ 128` 时至多省 1 个 tile，
  且仅当 `0 < total_m % 128 < L`。
- 不切换的理由：除节省可忽略外，`off'` 要对 `c_b − b` 做向下取整除法。前置
  专家为空时（`m_b = 0`，`c_b < b`，MoE 中常见）该值变负。kernel 的 Int32 `//`
  向零截断，在负数上舍入方向错误。当前公式只对非负值做除法。它同时是用户可见的
  存储契约，切换要求所有 padded SFA 的生产方同步改动。

### 层级 3：打包最小值需要额外的 kernel 输入

真正的下界是 `Σ_b ⌈m_b/128⌉` 个 tile。每个专家需要整 tile 独占：区域必须
tile 对齐，一个 512 B atom 不能跨专家。达到该下界所需的偏移是 `Σ_{j<b} ⌈m_j/128⌉`。
该偏移不是 `(c_b, b)` 的函数：两个前缀和相同（同 `c`、同 `b`）的切分可能需要不同的
tile 数。实现该下界需要第二个前缀数组（`cu_padded_rm`），在 host 计算（引入同步）或
设备 prologue 里算。收益是至多 `L−1` 个 tile，即 `(L−1) · rk · 512` 字节，约
`4k(L−1)` 字节（k=7168 时约 28 KB/专家）。该收益相对最坏情况（所有
`m_b = 128`）约 2 倍 SF 内存。SFA 只占 A 字节数的 `1/sf_vec_size` = 1/32。
即便如此，收益也只有 A operand 的约 3%。不值得为此加输入。

三条结论都经穷举验证：`L ≤ 5`、`m_b ≤ 3·tile`，按缩放的 tile 尺寸遍历全部模式
（代数与 tile 尺寸无关）。

## kernel 索引

`tile_atom_to_shape_SF_strided` 构造 SF 布局。外层 tile 维（`varlen_m` 的
`rm`、`varlen_k` 的 `rk`）是复合模式 `((32, 4), rm_or_rk)` 的第二个元素。
偏移只作用于该外层元素，经 `cute.domain_offset` 传复合坐标：

```python
# varlen_m (M padded)
offset_tile = cu_seqlens_m[batch_idx] // 128 + batch_idx
mSFA_batch = cute.domain_offset(((0, offset_tile), None), mSFA_mkl)

# varlen_k (K padded)
offset_tile = cu_seqlens_k[batch_idx] // 128 + batch_idx
mSFA_batch = cute.domain_offset((None, (0, offset_tile)), mSFA_mkl)
mSFB_batch = cute.domain_offset((None, (0, offset_tile)), mSFB_nkl)
```

热路径上没有 `* 128`。tile 对齐是语法层面的：编译器看到外层 rm/rk 的 stride
（字节的 `s_rm` / `s_rk`）作用在一个 tile 单位的偏移整数上。

## 实现指引

- `cute_gemm/varlen_utils.py`
  - `VarlenManager.offset_batch_SFA`：经复合坐标施加 padded M 或 K 偏移。
  - `VarlenManager.offset_batch_SFB`：`varlen_k` 的 padded K 偏移。
- `cute_gemm/gemm_sm100.py`：布局 setup 区分两种模式。`varlen_m` 从
  `mSFA.shape[1] * 128` 取 padded M。`varlen_k` 从 `mSFA.shape[2] * 128` 取
  padded K。
- `cute_gemm/blockscaled/utils.py`
  - `create_blockscaled_varlen_m_operands(seqlens_m=...)`
  - `create_blockscaled_varlen_k_operands(seqlens_k=...)`
  - 两者先在源单位的 torch buffer 里按偏移 `(c_b // 128 + b) * 128` 填充，再经
    `pack_scale_2d_to_blocked_contig` 打包成 `(1, rmn, rk, 32, 4, 4)` 布局。
- 公开 API（`varlen_m` 与 `varlen_k` 通用）：
  - `cute_gemm/gemm.py::gemm(..., cu_seqlens_m=..., SFA=..., SFB=...)`。SFA 是
    padded buffer 的 `(1, total_padded_rm, rk, 32, 4, 4)` 视图。SFB 按 batch
    存放，形状 `(L, rn, rk, 32, 4, 4)`。`_compile_gemm` 里 fake SFA 有独立的
    batch 符号（batch 维是 1，不是 `l`）。
  - `cute_gemm/gemm.py::gemm(..., cu_seqlens_k=..., SFA=..., SFB=...)`。A 是
    `(m, total_k)` 的 m-major，B 是 `(n, total_k)` 的 n-major。SFA 与 SFB 都是
    K-padded 的 `(1, rm/rn, total_padded_rk, 32, 4, 4)` buffer（两个 fake SF
    tensor 各有独立的 batch 符号）。
  - `cute_gemm/gemm_tvm_ffi_utils.py::validate_blockscaled_sf(num_batches=...,
    varlen_k=...)`：校验 `SFA.shape[1] >= ceil(total_m/128) + (L-1)`
    （varlen_m），或 `SF*.shape[2] >= ceil(total_k/128) + (L-1)`（varlen_k，
    两个 buffer 都校验）。
  - `cute_gemm/gemm_interface.py::gemm((A, SFA), (B, SFB), cu_seqlens_m=...)` 与
    `gemm((A, SFA), (B, SFB), cu_seqlens_k=...)`（B 按 `(total_k, n)` 传入）。
- `cute_gemm/layout_utils.py`
  - `tile_atom_to_shape_SF_strided(shape, sf_vec_size, sf_strides)`：用 mSFA
    自身的 stride 和 shape 构造 CuTe 布局。padded 总量（而非未 padded 的
    `mA.shape`）决定外层 rm/rk 计数。

## 测试

`tests/test_gemm_sm100_blockscaled.py`：

- `test_blockscaled_varlen_m_public_api`：同一套 setup 走 `cute_gemm.gemm.gemm`
  （3 种 seqlen 模式 × 2 种 B-major × 3 种格式）。接口层覆盖在
  `tests/test_gemm_blockscaled_interface.py::test_blockscaled_gemm_varlen_m`。
- `test_blockscaled_varlen_m_nonaligned`：4 种 seqlen 模式 × 2 种 B-major ×
  3 种格式。模式含 `[128, 128, 128]`、`[100, 200, 150]`、`[30, 300, 64, 200]`、
  `[1, 128, 127, 129]`。
- `test_blockscaled_mxfp8_varlen_k`：6 种模式，含非 128 对齐的
  `[96, 160, 128]`、`[32, 256, 64, 128]` 和非 32 对齐的 `[100, 220, 65]`、
  `[1, 33, 158, 192]`（每专家最后一个 scale block 不完整）。
- `test_blockscaled_varlen_k_public_api`：同一套 setup 走 `cute_gemm.gemm.gemm`
  （3 种模式，含非 32 对齐）。接口层覆盖在
  `tests/test_gemm_blockscaled_interface.py::test_blockscaled_gemm_varlen_k`。

每个专家都用参考 `a_ref_list[i] @ b_ref_list[i].T`（或沿非变长维 cat 的
等价形式）对比 kernel 单趟输出。这验证每个专家的区域填充正确，无重叠、无
欠填充。
