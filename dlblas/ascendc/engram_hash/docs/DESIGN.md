# engram_hash AscendC 算子架构设计文档

> **版本**: v1.0
> **日期**: 2026-07-01
> **算子名称**: engram_hash
> **算子类型**: 整数索引计算类（Integer Index / Hash，Elementwise-per-token）
> **目标芯片**: Ascend 910B2 (DAV_2201)
> **CANN 版本**: 9.0.0

---

## 1. 需求分析

### 1.1 数学定义

engram_hash 为 N-gram embedding 索引哈希算子。给定 N-gram token id、逐层乘子、逐表词表大小与偏移，计算每层、每 token、每个 (ngram 位置 × embedding 表) 的嵌入索引。

记号：
- `NT`  = num_tokens
- `NG`  = max_ngram_size（下文简写 N）
- `L`   = num_ngram_layers
- `T`   = num_embed_table_per_ngram
- `P`   = (N-1)                     （产生索引的 ngram 位置数）
- `W`   = P * T = (N-1) * T          （每层每 token 的输出宽度）

核心算法（逐 (layer, token) 独立）：

```
给定: ngram_token_ids[NT, N] (int32)
      multipliers[L, N]       (int64)
      vocab_sizes[L, N-1, T]  (int32)
      offsets[L, W]           (int32)

对每个 layer l ∈ [0, L), 每个 token tk ∈ [0, NT):

  # 1) 逐元素乘 (int32 提升 int64) —— prod[i] = ngram[tk,i] * mult[l,i]
  # 2) XOR 链哈希 —— hash 从 prod[0] 起, 依次异或 prod[1..N-1]
  h = int64(ngram[tk,0]) * mult[l,0]
  for i in 1 .. N-1:
      h ^= int64(ngram[tk,i]) * mult[l,i]      # bitwise_xor 累积
      # 3) 取模 + 4) 加偏移 —— 每个位置 i 产生 T 个索引
      for t in 0 .. T-1:
          col = (i-1) * T + t
          out[l, tk, col] = int32( h % int64(vocab_sizes[l, i-1, t]) ) + offsets[l, col]

输出: output[L, NT, W] (int32)
```

**关键语义点**：`hashes` 在位置 i 更新后，会被后续所有位置 j>i 继续异或。即位置 i 的输出使用的是"截止到位置 i 的前缀 XOR 哈希"，而非最终哈希。这是逐 token 状态机式的前缀累积，必须严格按 i 递增顺序执行。

### 1.2 输入输出规格

| 参数 | 形状 | 数据类型 | 说明 |
|------|------|---------|------|
| ngram_token_ids | (NT, N) | int32 | N-gram token id，值域 [0, 100000) |
| multipliers | (L, N) | int64 | 逐层哈希乘子，值域 [0, 100000) |
| vocab_sizes | (L, N-1, T) | int32 | 逐层逐 ngram 词表大小，值域 [100000, 1000000) |
| offsets | (L, W) | int32 | 逐层 embedding 表偏移（vocab_sizes 的 exclusive prefix-sum） |
| **输出** | (L, NT, W) | int32 | 嵌入索引 |

**默认基线 shape**（`get_inputs`）：NT=4096, N=3, L=2, T=8 → P=2, W=16，输出 (2, 4096, 16)。

### 1.3 数值精度与正确性约束（决定性约束）

按 `ops-precision-standard` 判定：输入输出**均为整型且含算术运算** → **整数计算类**，通过标准为 **二进制一致 / 绝对误差为 0（bit-exact）**。这是硬性约束，不是容差比对。因此设计必须保证：

1. **int64 全程精确**：乘法、XOR、取模均在 64-bit 整数域完成，不得引入任何浮点近似（浮点 mantissa 仅 52-bit，无法表示 34-bit 乘积的精确 XOR/取模结果）。
2. **溢出/回绕语义对齐**：需与 PyTorch int64 语义一致。

**数值范围核算（已用 PyTorch 验证）**：
- token < 100000，multiplier < 100000 → `prod < 1e10`，占 **≤ 34 bit**，远在 int64 (63 有效位) 内，乘法无溢出。
- prod 恒为非负，XOR 结果 **bit63 恒为 0**（永远非负）。因此 `hashes` 恒 ≥ 0。
- vocab_sizes ∈ [1e5, 1e6) 恒为正。两操作数均非负 ⇒ **C++ `%` 与 PyTorch `%`（torch.remainder）结果完全一致**（分歧只在负操作数时出现，此处不发生）。
- `h % vocab < 1e6`，加 offsets（prefix-sum，随 W 增长最大约 P*T*1e6 ≈ 1.6e7）后 < 2^31，**int32 输出无溢出**。

> 结论：kernel 内可安全使用标量 `int64_t` 原生乘法/`^`/`%`，结果与 PyTorch 逐位一致。已用 tiny case（L=2,NT=3,N=3,T=3）对 `Model.forward` 做标量重算，`torch.equal == True`，公式锁定。

### 1.4 计算特征归纳

| 特征 | 判断 |
|------|------|
| 计算类型 | 纯整数标量运算（mul / xor / mod / add），**无任何浮点、无归约、无矩阵乘** |
| 数据复用 | multipliers[l,:]、vocab_sizes[l,:,:]、offsets[l,:] 在同层内被所有 token 复用；ngram[tk,:] 被所有 layer 复用 |
| 访存密度 | 输出量 = L*NT*W*4 bytes（基线 2*4096*16*4 = 512 KB）；输入量极小（ngram 4096*3*4=48KB，其余 <1KB） |
| 并行度 | (layer × token) 二维完全独立，天然可并行 NT*L 份 |
| per-元素工作量 | 每个 (l,tk) 做 N-1 次(乘+异或) + W 次(取模+加)，极小 |

---

## 2. 架构环境与技术路线决策

### 2.1 硬件环境

| 参数 | 值 | 来源 |
|------|:---:|------|
| 芯片型号 | Ascend 910B2 | npu-smi info |
| **NpuArch** | **DAV_2201** | `/npu-arch` |
| **`__NPU_ARCH__`** | **2201** | `/npu-arch` |
| `--npu-arch` 编译参数 | `dav-2201` | `/npu-arch` |
| SocVersion | ASCEND910B | `/npu-arch` |
| CANN 版本 | 9.0.0 | 环境探测 (`$ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0`) |

**DAV_2201 / Ascend910B2 关键参数**（`npu-hardware-params.md`，运行时以 `PlatformAscendC` 为准）：

| 资源 | 值 |
|------|:---:|
| CubeCore 数 | 24 |
| **VectorCore 数** | **48** |
| CubeCore : VectorCore | 1 : 2 |
| UB (Unified Buffer) | 192 KB (196608 B) |
| L1 | 512 KB |
| L2（共享） | 192 MB |
| Cube MAC 阵列 | 16×16×16 |

### 2.2 技术路线决策：Vector / Matmul / 混合？

```
决策树:
  NpuArch == DAV_3510 ?  → 否 (DAV_2201)，排除 RegBase / Blaze / TensorAPI 路线
  算子含矩阵乘 (GEMM/BMM/卷积) ?  → 否，纯整数 mul/xor/mod/add，无 K 维累加
    ⇒ 排除 Cube/Matmul 指令（Cube 只做 float/int8 MAC，无法表达 XOR/mod）
  算子核心是 int64 逐元素 + 位运算 + 取模 ?  → 是
    ⇒ Vector 逐元素 API 能否胜任？
       - DAV_2201 Vector 单元不支持 int64 元素级算术（Mul/Xor/Mod 无 int64 重载/指令）
       - XOR 链是 per-token 顺序前缀状态，且 mod 的除数逐 (table) 变化（非标量广播）
       - 输出布局跨 (layer, ngram-pos, table) 交织，非规则连续 Vector pattern
    ⇒ Vector 向量化收益低且 int64 不被支持
```

**最终决策：AIV-only（Vector 核）+ 标量整数计算路线**。

**决策依据**：
1. **架构**：DAV_2201，非 DAV_3510，排除 RegBase / Blaze / tensor_api。
2. **无矩阵乘**：算子不含任何 M×K×N 累加结构，Cube 阵列无法表达 `^`/`%`，排除 Matmul 指令与混合路线。
3. **int64 位运算/取模不被 Vector 支持**：DAV_2201 Vector 逐元素 API（Add/Mul/And/Or 等）主要面向 fp16/fp32/int32/int16；**无 int64 元素级 Mul、无逐元素 Xor over int64、无 Mod 指令**。强行用 Vector 需拆分 int64 为高低 32 位手工模拟乘法/取模，代码复杂且极易破坏 bit-exact。
4. **算法结构不利于向量化**：XOR 链是 per-token 顺序状态机（位置 i 依赖 i-1 的前缀），mod 除数 `vocab_sizes[l,i-1,t]` 逐 table 变化（不是标量广播），输出跨 (layer×pos×table) 交织写出。
5. **标量单元原生支持 int64**：AI Core 标量单元（scalar pipe）是通用 ALU，C++ `int64_t` 的 `*`/`^`/`%` 原生编译为标量指令，**天然 bit-exact**，且本仓已有 MTPBlock 用相同标量 + 原始 `__gm__` 指针范式在 DAV_2201 上跑通并验证。

> 该算子按 AscendWiki `patterns/scalar-bound` 属于典型 **scalar-bound**（控制/地址/整数计算主导，无 float 数学）。这里 scalar 是**语义必需**而非退化——不应试图"消除 scalar 比例"，而应通过合适的多核切分与减少 launch 次数把标量吞吐打满。

### 2.3 计算单元与数据搬运选择

| 维度 | 选择 | 理由 |
|------|------|------|
| 核类型 | **AIV_ONLY**（`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`） | 纯 Vector 核标量计算，不涉及 Cube |
| 计算方式 | **标量 int64**（`int64_t` 原生 `* ^ %`） | bit-exact 强约束 + Vector 不支持 int64 |
| 输入访存 | **UB 缓存 + 标量读**（ngram tile 经 DataCopy 入 UB；multipliers/vocab/offsets 因极小可整表入 UB） | 复用度高的小张量常驻 UB，避免重复 GM 访问 |
| 输出访存 | **UB 暂存 + DataCopy 批量写出**（按对齐要求，非对齐用 DataCopyPad 或整 tile 对齐规划） | 输出 512KB 级，批量 DMA 写出优于逐元素 |

> **关于 `GetValue/SetValue` 黑名单**：`ascendc-api-best-practices` 将 **GlobalTensor**::SetValue/GetValue 列入黑名单（GM 逐元素访问极慢）。本设计对 **GM** 一律用 DataCopy/DataCopyPad 批量搬运；仅对**已在 UB 内**的 LocalTensor 或已读入寄存器/栈的标量做逐元素运算（LocalTensor::GetValue/SetValue 在 UB 上是允许的，MTPBlock 已验证）。这与黑名单不冲突。

---

## 3. 数据流分析

### 3.1 数据布局与索引映射

所有输入按 PyTorch 行主序（contiguous）：

```
ngram_token_ids[NT, N]        : elem(tk,i)      offset = tk*N + i
multipliers[L, N]             : elem(l,i)       offset = l*N + i
vocab_sizes[L, N-1, T]        : elem(l,i-1,t)   offset = l*(P*T) + (i-1)*T + t = l*W + (i-1)*T + t
offsets[L, W]                 : elem(l,col)     offset = l*W + col
output[L, NT, W]              : elem(l,tk,col)  offset = l*(NT*W) + tk*W + col
```

其中 `W = P*T = (N-1)*T`，`col = (i-1)*T + t`。注意 vocab_sizes flatten 后宽度恰为 W，与 offsets/output 的 W 对齐，便于统一按 col 索引。

### 3.2 数据流图

```
                    ┌──────────────────────── engram_hash_kernel (AIV_ONLY) ────────────────────────┐
                    │                                                                               │
 GM (multipliers) ──┤ ① DataCopy 整表 → mulUb   [L*N int64]      （常驻，全核相同，极小 <1KB）        │
 GM (vocab_sizes) ──┤ ① DataCopy 整表 → vocUb   [L*W int32]                                          │
 GM (offsets)     ──┤ ① DataCopy 整表 → offUb   [L*W int32]                                          │
                    │                                                                               │
 GM (ngram_ids)   ──┤ ② 按 token-tile DataCopy → ngUb  [tileTokens*N int32]  （本核负责的 token 段）  │
                    │        │                                                                       │
                    │        ▼   ③ 逐 (layer, token, i) 标量计算                                     │
                    │   for l in 0..L:                                                               │
                    │     for tk in tile:                                                            │
                    │       h = (int64)ngUb[tk,0] * mulUb[l,0]                                       │
                    │       for i in 1..N-1:                                                         │
                    │         h ^= (int64)ngUb[tk,i] * mulUb[l,i]                                    │
                    │         for t in 0..T-1:                                                       │
                    │           col=(i-1)*T+t                                                        │
                    │           outUb[..] = (int32)(h % (int64)vocUb[l,col]) + offUb[l,col]          │
                    │        │                                                                       │
                    │        ▼   ④ DataCopy outUb → GM                                               │
 GM (output)      ◄─┤   （按 layer 分段写 output[l, tokenOff:.., :]，每段 tileTokens*W int32）        │
                    └───────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 复用分析

- **multipliers / vocab_sizes / offsets**：总量 `L*N*8 + L*W*4 + L*W*4`（基线 = 2*3*8 + 2*16*4*2 = 48 + 256 = 304 B），远小于 UB。**一次性整表载入 UB，被本核处理的所有 token 复用**，消除重复 GM 访问。
- **ngram_token_ids[tk,:]**：每个 token 的 N 个 id 被 L 个 layer 复用。按 token-tile 载入 UB 后，内层 layer 循环直接复用。
- **hashes 前缀状态**：仅存在于寄存器/栈标量 `h`，per (l,tk) 生命周期，无需落 UB/GM。

---

## 4. 并行策略

### 4.1 并行维度选择

输出 (L, NT, W)，其中 **(layer, token) 两维完全独立**，是理想的并行切分维度。W 内部（跨 pos×table）由于共享同一 token 的前缀哈希链，**不跨核切分**（切了会重复算 XOR 链，得不偿失），在核内串行完成。

**切分轴：token 维（NT）**。每个核负责一段连续 token，核内对该段 token 遍历全部 L 层。

理由（对照 `AscendWiki/multi-core-scheduling` 的 Goldilocks 原则）：
- NT（基线 4096）≫ 核数，切 token 可让 48 个 Vector 核全部吃满，且每核工作量大、launch 开销被摊薄（对抗 scalar-bound 的首要杠杆：更大 tile / 更少 launch）。
- 不选 layer 维切分：L 通常很小（基线 2），切 layer 无法喂满 48 核。
- layer 放核内循环还能让 ngram tile 在 UB 内被 L 次复用。

### 4.2 多核切分公式

```
usedCores    = min(VECTOR_CORE_NUM, NT)          // 不超过 token 数；VECTOR_CORE_NUM 由平台接口获取(48)
tokensPerCore = ceil(NT / usedCores)
blockNum     = ceil(NT / tokensPerCore)          // 实际启核数
tailTokens   = NT - tokensPerCore * (blockNum-1) // 尾核 token 数
```

- **负载均衡**：除尾核外各核处理相同 token 数，差异 ≤ 1 个 tile。
- **coreDim 约束**：blockNum ≤ 65535（远不触及）。
- **核数不硬编码**：通过 `GetBlockNum()` / 平台接口获取，保证 910B3（20 Cube/40 Vector）可移植。

### 4.3 核内 tile（UB 切分）

每核 tokensPerCore 个 token，若一次性放不下则分 tile：

```
每 tile 处理 tileTokens 个 token（对齐考量见 §5）
tileLoop = ceil(tokensPerCore / tileTokens)
```

**tileTokens 预算**（Host 侧计算）：

```
固定常驻 UB  = L*N*sizeof(int64)      // multipliers
             + L*W*sizeof(int32)      // vocab_sizes
             + L*W*sizeof(int32)      // offsets
per-token UB = N*sizeof(int32)        // ngram tile 输入
             + L*W*sizeof(int32)      // 输出（L 层各 W）
UB_AVAIL     ≈ 184 KB（预留 8KB 给栈/对齐，参考 MTPBlock 的 MTP_UB_AVAIL）
tileTokens   = (UB_AVAIL - 固定常驻 - margin) / per-token UB
tileTokens   = 向下对齐到合适倍数（见 §5 对齐）；下限 clamp 到 ≥ 8
```

基线（N=3, L=2, W=16）：固定常驻 ≈ 304 B；per-token = 3*4 + 2*16*4 = 12 + 128 = 140 B。UB 可容纳上千 token/tile，故基线场景 tokensPerCore（4096/48≈86）通常单 tile 完成，无需多轮。设计仍保留多 tile 循环以覆盖极大 NT。

### 4.4 并行度评估

| 场景 | NT | usedCores | tokensPerCore | 说明 |
|------|:--:|:--:|:--:|------|
| 基线 | 4096 | 48 | 86（尾核 82） | 满核，负载均衡好 |
| 小 batch | 32 | 32 | 1 | token 少于核数，用 NT 个核 |
| 大 batch | 65536 | 48 | 1366 | 每核多 token，核内可能多 tile |

---

## 5. 内存管理策略

### 5.1 UB Buffer 规划

```
UB 布局 (192 KB total, 可用 ~184 KB):
┌─────────────────────────────────────────────────────────────┐
│ mulUb  (VECIN/常驻) : L*N * sizeof(int64)   ← multipliers 整表 │
├─────────────────────────────────────────────────────────────┤
│ vocUb  (VECIN/常驻) : L*W * sizeof(int32)   ← vocab_sizes 整表 │
├─────────────────────────────────────────────────────────────┤
│ offUb  (VECIN/常驻) : L*W * sizeof(int32)   ← offsets 整表     │
├─────────────────────────────────────────────────────────────┤
│ ngUb   (VECIN)      : tileTokens*N * sizeof(int32)  ← 输入 tile │
├─────────────────────────────────────────────────────────────┤
│ outUb  (VECOUT)     : tileTokens*L*W * sizeof(int32) ← 输出 tile│
└─────────────────────────────────────────────────────────────┘
```

| Buffer | Position | 大小公式 | 用途 |
|--------|----------|---------|------|
| mulUb | VECIN | `L*N*8` | multipliers（int64），常驻，全 tile 复用 |
| vocUb | VECIN | `L*W*4` | vocab_sizes（int32），常驻 |
| offUb | VECIN | `L*W*4` | offsets（int32），常驻 |
| ngUb | VECIN | `tileTokens*N*4` | 本 tile token 的 ngram id |
| outUb | VECOUT | `tileTokens*L*W*4` | 本 tile 输出（L 层交织或分段） |

**常驻小张量策略**：mulUb/vocUb/offsets 在 tile 循环外一次性 DataCopy 载入（每核都载，因数据小；避免核间同步）。

### 5.2 int64 在 UB 的处理

- multipliers 为 int64。DataCopy 按字节搬运，UB 上以 `LocalTensor<int64_t>` 视图承载（或以 `int64_t*` 视 UB 基址）。逐元素通过 `mulUb.GetValue(idx)`（UB 上标量读，允许）取出参与标量乘。
- 若 `LocalTensor<int64_t>` 的 GetValue 在目标 CANN 版本行为不确定，**兜底方案**：multipliers 直接用原始 `__gm__ int64_t*` 标量读（其总量 L*N 极小，GM 标量读次数 = usedCores*L*N，可忽略），无需入 UB。这一兜底与 MTPBlock 的 `((__gm__ float*)post)[idx]` 小表直读范式一致。

### 5.3 对齐策略（DataCopy 32B 约束）

DataCopy(GM↔UB) 要求严格 32 字节对齐，否则须用 DataCopyPad。相关对齐点：

| 数据 | 元素大小 | 32B 对齐要求 | 处理 |
|------|:--:|------|------|
| ngram tile 输入 | int32(4B) | `tileTokens*N*4 % 32 == 0` ⇒ `tileTokens*N % 8 == 0` | tileTokens 选择使 `tileTokens*N` 为 8 倍数；否则 DataCopyPad |
| 输出写出 | int32(4B) | 每段 `count*4 % 32 == 0` ⇒ `count % 8 == 0` | 按 layer 分段写 `output[l, off:off+tileTokens, :]`，段长 `tileTokens*W`，选 `tileTokens*W % 8 == 0`；否则 DataCopyPad |
| 常驻小表 | 混合 | 量小 | 首选整表 DataCopy；非对齐用 DataCopyPad，或走 §5.2 GM 直读兜底 |

> 基线 N=3、W=16：`tileTokens*W % 8 == 0` 对任意 tileTokens 成立（W=16 是 8 的倍数）；`tileTokens*N=tileTokens*3` 需 tileTokens 为 8 的倍数才对齐——故 **tileTokens 默认对齐到 8 的倍数**，同时满足两处；边界尾 tile 用 DataCopyPad 收尾。

### 5.4 输出组织：layer 交织 vs 分段

输出 `output[L, NT, W]` 中，同一 tile 的 token 在**不同 layer 下地址不连续**（layer 是最外维，stride = NT*W）。两种组织：

- **方案 A（分段写，推荐）**：outUb 布局为 `[L][tileTokens][W]`，计算完后**按 layer 循环**做 L 次 DataCopy，第 l 次写 `output + l*NT*W + tokenOff*W`，段长 `tileTokens*W`（连续）。简单、对齐可控。
- **方案 B（单次写）**：不可行——L 层输出在 GM 非连续，无法一次 DataCopy 覆盖。

采用**方案 A**。outUb 内部按 `[l*tileTokens*W + local_tk*W + col]` 排布。

### 5.5 内存生命周期（Host 侧）

| 对象 | 分配 | 大小 | 生命周期 |
|------|------|------|------|
| 4 个输入 | 外部传入（contiguous 化） | 见 §1.2 | 调用期间 |
| output | `at::empty({L,NT,W}, int32)` | `L*NT*W*4` | 返回后由调用者管理 |
| TilingData(Device) | `aclrtMalloc` | `sizeof(EngramHashTilingData)` | kernel 完成即 `aclrtFree` |

### 5.6 Tiling 数据结构

```cpp
struct EngramHashTilingData {
    // ── 问题规模 ──
    uint32_t numTokens;      // NT
    uint32_t ngramSize;      // N (= max_ngram_size)
    uint32_t numLayers;      // L
    uint32_t numTables;      // T
    uint32_t ngramPos;       // P = N-1
    uint32_t outWidth;       // W = (N-1)*T
    // ── 多核切分 ──
    uint32_t tokensPerCore;  // 每核 token 数
    uint32_t tailTokens;     // 尾核 token 数
    uint32_t blockNum;       // 实际启核数
    // ── 核内 tile ──
    uint32_t tileTokens;     // 每 tile token 数（8 对齐）
    uint32_t lastTileTokens; // 尾 tile token 数
    // ── 对齐标志 ──
    uint32_t inAligned;      // ngram tile 是否 32B 对齐
    uint32_t outAligned;     // 输出段是否 32B 对齐
};
```

---

## 6. Kernel 详细设计

### 6.1 核函数签名（直接调用 ABI）

```cpp
extern "C" __global__ __aicore__ void engram_hash_kernel(
    GM_ADDR ngram_token_ids,   // int32  [NT, N]
    GM_ADDR multipliers,       // int64  [L, N]
    GM_ADDR vocab_sizes,       // int32  [L, N-1, T]
    GM_ADDR offsets,           // int32  [L, W]
    GM_ADDR output,            // int32  [L, NT, W]
    GM_ADDR tiling);
```

Host 侧直接调用别名（供可执行文件 / torch 扩展调用）：

```cpp
extern "C" void engram_hash_kernel(
    uint32_t blockDim, void* l2ctrl, aclrtStream stream,
    void* ngram, void* mult, void* vocab, void* offsets,
    void* output, void* tiling);
```

### 6.2 Kernel 执行流程

```
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
td  = (__gm__ EngramHashTilingData*)tiling;
bid = GetBlockIdx();
if (bid >= td->blockNum) return;

myTokens = (bid < blockNum-1 || tailTokens==0) ? tokensPerCore : tailTokens;
if (myTokens == 0) return;
tokenBase = bid * tokensPerCore;

// ── 常驻小表载入 UB（tile 循环外，仅一次）──
DataCopy(mulUb, multipliers[0], L*N);        // int64；或走 GM 直读兜底
DataCopy(vocUb, vocab_sizes[0], L*W);        // int32
DataCopy(offUb, offsets[0],     L*W);        // int32
(同步 EnQue/DeQue)

// ── token tile 循环 ──
loops = ceil(myTokens / tileTokens);
for (t = 0; t < loops; t++) {
    cur = (t==loops-1) ? (myTokens - t*tileTokens) : tileTokens;
    tOff = tokenBase + t*tileTokens;                 // 全局 token 起点

    // ① DMA-IN ngram tile
    DataCopy(ngUb, ngram[tOff*N], cur*N);            // 非对齐→DataCopyPad
    (EnQue/DeQue 同步)

    // ② 标量计算（核心）
    for (l = 0; l < L; l++) {
      for (j = 0; j < cur; j++) {                    // j: tile 内局部 token
        int64_t h = (int64_t)ngUb.GetValue(j*N + 0) * mulUb.GetValue(l*N + 0);
        for (i = 1; i < N; i++) {
          h ^= (int64_t)ngUb.GetValue(j*N + i) * mulUb.GetValue(l*N + i);
          for (tb = 0; tb < T; tb++) {
            uint32_t col = (i-1)*T + tb;
            int64_t v = (int64_t)vocUb.GetValue(l*W + col);   // int32→int64
            int32_t idx = (int32_t)(h % v) + offUb.GetValue(l*W + col);
            outUb.SetValue(l*tileTokens*W + j*W + col, idx);   // UB 写，允许
          }
        }
      }
    }
    (outUb EnQue/DeQue 同步)

    // ③ DMA-OUT（按 layer 分段）
    for (l = 0; l < L; l++) {
      DataCopy(output[l*NT*W + tOff*W],
               outUb[l*tileTokens*W], cur*W);        // 非对齐→DataCopyPad
    }
}
```

### 6.3 API 映射表

| 步骤 | API / 操作 | 类型 | 说明 |
|------|-----------|------|------|
| 常驻/输入载入 | `DataCopy` / `DataCopyPad`（GM→UB） | MTE2 | 32B 对齐用 DataCopy，否则 DataCopyPad |
| int64 乘法 | `int64_t a * int64_t b` | Scalar | 原生标量指令，bit-exact |
| XOR 链 | `h ^= prod` | Scalar | 原生标量按位异或 |
| 取模 | `h % v`（均非负） | Scalar | 与 PyTorch % 一致 |
| int32 加偏移 | `(int32_t)(...) + off` | Scalar | 结果 < 2^31 |
| UB 元素读 | `LocalTensor::GetValue(idx)` | Scalar/UB | 允许（非 GM） |
| UB 元素写 | `LocalTensor::SetValue(idx, v)` | Scalar/UB | 允许（非 GM） |
| 输出写出 | `DataCopy` / `DataCopyPad`（UB→GM） | MTE3 | 按 layer 分段 |

> **禁止项**：不得对 **GlobalTensor** 使用 GetValue/SetValue（黑名单）。GM 侧一律 DataCopy 系列。

### 6.4 同步机制

```
TPipe
 ├─ VECIN  Que: mulUb, vocUb, offUb（常驻，载入时 EnQue/DeQue 一次）+ ngUb（每 tile）
 └─ VECOUT Que: outUb（每 tile EnQue/DeQue）
```

- **常驻表**：载入后 EnQue/DeQue 一次，之后只读，无需重复同步。
- **ngUb**：每 tile `AllocTensor → DataCopy → EnQue → DeQue`，保证 MTE2 完成后再标量读。
- **outUb**：标量写完 `EnQue → DeQue → DataCopy` 写出，保证计算完成后再 MTE3。
- **核间**：各核处理独立 token 段，**无核间同步**。
- **可选 Double Buffer**（优化项）：ngUb/outUb depth=2，tile 间 CopyIn/Compute/CopyOut overlap；因本算子访存量小、计算标量密集，收益需实测（见 §8）。

### 6.5 边界处理

| 场景 | 处理 |
|------|------|
| `bid >= blockNum` | 直接 return |
| `myTokens == 0` | 直接 return |
| 尾 tile `cur < tileTokens` | `cur = myTokens - t*tileTokens` 精确计算 |
| ngram/输出段非 32B 对齐 | DataCopyPad（尾 tile 尤其需要） |
| N==1（无 XOR 位置） | P=0, W=0，输出为空张量；kernel 应提前判 W==0 return（防御性） |
| NT < 核数 | usedCores=NT，多余核 return |

### 6.6 通用化（多 shape 支持）

kernel 完全参数化（NT/N/L/T/W 全部来自 TilingData），**不硬编码任何 shape 常量**，支持任意 (num_tokens, max_ngram_size, num_ngram_layers, num_embed_table_per_ngram)。栈上临时数组（若用于缓存 mult 行等）需按 N 的合理上界（如 N≤16）静态分配，或直接从 UB/GM 逐元素读避免栈数组。

---

## 7. Host 侧与 PyTorch 接入

### 7.1 Tiling 计算流程（ComputeTiling）

```
输入: NT, N, L, T, coreNum(平台获取), UB_AVAIL
1. P = N-1;  W = P*T
2. usedCores    = min(coreNum, NT)
   tokensPerCore = ceil(NT / usedCores)
   blockNum      = ceil(NT / tokensPerCore)
   tailTokens    = NT - tokensPerCore*(blockNum-1)
3. residentUb   = L*N*8 + L*W*4 + L*W*4
   perTokenUb   = N*4 + L*W*4
   tileTokens   = (UB_AVAIL - residentUb - MARGIN) / perTokenUb
   tileTokens   = alignDown(tileTokens, 8); clamp ≥ 8; clamp ≤ tokensPerCore
4. loops          = ceil(tokensPerCore / tileTokens)
   lastTileTokens = tokensPerCore - (loops-1)*tileTokens
5. inAligned  = ((tileTokens*N) % 8 == 0)
   outAligned = ((tileTokens*W) % 8 == 0)
6. 填充 TilingData
```

### 7.2 直接调用主程序（op_host/*.asc）

命令行驱动：`argv = NT N L T [devId] [coreNum]`。生成随机输入（对齐 `generate_test_data` 值域）→ ACL 分配 device 内存 → memcpy H2D → ComputeTiling → 拷贝 tiling → 启动 kernel → D2H → 落盘供 verify。

### 7.3 PyTorch 扩展（op_extension）

```cpp
// TORCH_LIBRARY 注册
TORCH_LIBRARY_FRAGMENT(npu, m) {
  m.def("engram_hash(Tensor ngram_token_ids, Tensor multipliers, "
        "Tensor vocab_sizes, Tensor offsets) -> Tensor");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) { m.impl("engram_hash", TORCH_FN(ascend_kernel::engram_hash_torch)); }
TORCH_LIBRARY_IMPL(npu, Meta,        m) { m.impl("engram_hash", &engram_hash_meta); }
```

前端 `engram_hash_torch`：校验 device/dtype/contiguous → 从 shape 推 NT/N/L/T → 分配 `output[L,NT,W]` int32 → ComputeTiling → `c10_npu::getCurrentNPUStream()` → 启动 kernel → 返回 output。Meta 实现推导输出形状 `(L, NT, W)`、dtype int32，供 torch.compile 追踪。

### 7.4 dtype 契约

- ngram_token_ids: `at::kInt`(int32)
- multipliers: `at::kLong`(int64)
- vocab_sizes / offsets / output: `at::kInt`(int32)

kernel 内 `GM_ADDR` reinterpret 为对应指针类型；multipliers 按 int64 处理。

---

## 8. 性能分析与优化方向

### 8.1 性能特征

- **瓶颈定位**：scalar-bound（`AscendWiki/scalar-bound`）。每 (l,tk) 做 (N-1) 次乘+异或、W 次取模+加，取模是标量整数除法（相对慢）。基线总标量运算量 ≈ L*NT*(N-1)*T 次取模 = 2*4096*2*8 ≈ 131K 次 mod。
- **访存**：输出 512KB / 输入 ~48KB，L2 可缓冲，非主瓶颈。
- **结论**：性能由标量吞吐（尤其 `%`）与核利用率决定。

### 8.2 优化路线图（按优先级）

| 优先级 | 方向 | 预期收益 | 依据 |
|:--:|------|:--:|------|
| P0 | **满核 token 切分**（48 核，摊薄 launch） | 接近 ×核数 | `multi-core-scheduling` Goldilocks；对抗 scalar-bound 首要杠杆 |
| P0 | **常驻表 UB 复用**（mult/vocab/offset 不重复读 GM） | 减少 MTE2 | §3.3 复用分析 |
| P1 | **减少取模开销**：同一 (l,tk,i) 的哈希 h 固定，T 个 mod 除数不同无法合并；但可将 `h` 缓存寄存器避免重复读 | 小 | 循环不变量提升 |
| P1 | **Double Buffer**（ngUb/outUb depth=2）tile 间 overlap | 需实测（访存占比低，可能收益有限） | §6.4 |
| P2 | **循环展开**（N、T 小且编译期可知时对内层展开） | 减少循环控制标量开销 | scalar-bound 控制开销 |
| P2 | **tileTokens 增大**（单 tile 覆盖更多 token，减少 tile 循环） | 减少 DMA/同步次数 | §5.1 UB 充裕 |

### 8.3 不采用的方向及原因

| 方向 | 不采用原因 |
|------|-----------|
| Vector 向量化 mul/xor/mod | DAV_2201 Vector 无 int64 元素级 mul/xor/mod；手工拆高低 32 位极易破坏 bit-exact |
| Cube/Matmul | 无矩阵乘结构，Cube 无法表达 XOR/mod |
| 浮点近似取模 | 违反 bit-exact 硬约束（float 52-bit mantissa 无法精确表示 34-bit 乘积链） |
| 跨核切分 W（pos×table） | 破坏 XOR 前缀链的 token 内串行依赖，需重算，得不偿失 |

---

## 9. 验证与精度

### 9.1 精度标准

按 `ops-precision-standard` → **整数计算类**：通过标准 = **二进制一致 或 绝对误差为 0（bit-exact）**。无 rtol/atol 容差。

### 9.2 标杆

CPU/PyTorch golden：直接调用参考 `Model.forward`（源文件 `origin/engram_hash.py`）。已验证标量重算与 `Model.forward` `torch.equal == True`。

### 9.3 验证矩阵

| 维度 | 取值 | 目的 |
|------|------|------|
| num_tokens | 32, 256, 1024, 4096, 65536 | 小/中/大 batch、满核/欠核 |
| max_ngram_size (N) | 2, 3, 4, 5 | P、XOR 链长度变化 |
| num_ngram_layers (L) | 1, 2, 4 | 核内 layer 循环 |
| num_tables (T) | 1, 4, 8, 16 | W 宽度、取模次数 |

多 shape 每例独立比对 bit-exact，全通过方判定 pass。

---

## 10. 文件结构

```
operators/engram_hash/
├── CMakeLists.txt                     # 双目标 (可执行文件 + libengram_hash_ops.so)，--npu-arch=dav-2201
├── op_kernel/
│   ├── engram_hash_kernel.asc         # AIV-only 标量 kernel
│   └── engram_hash_tiling.h           # EngramHashTilingData + 常量
├── op_host/
│   ├── engram_hash_host.asc           # 直接调用主程序（命令行驱动测试）
│   └── data_utils.h                   # 文件读写 + 日志宏
├── op_extension/
│   ├── engram_hash_torch.cpp          # PyTorch 集成（host tiling + launch）
│   ├── register.cpp                   # TORCH_LIBRARY 注册（PrivateUse1 + Meta）
│   └── ops.h                          # 函数声明
├── scripts/
│   ├── gen_data.py                    # 测试数据生成 + golden（复用 origin 参考）
│   ├── verify_result.py               # bit-exact 整数比对
│   └── benchmark.py                   # 性能采集
└── docs/
    ├── DESIGN.md                      # 本文档
    └── PLAN.md                        # 开发计划
```

---

## 11. 参考

| 文档 | 用途 |
|------|------|
| `origin/engram_hash.py` | PyTorch 参考实现（golden 来源） |
| `test/MTPBlock/*` | 本仓已验证的 AIV-only 标量 + `__gm__`/UB 范式模板（kernel/host/torch/CMake） |
| `AscendWiki: patterns/scalar-bound` | scalar-bound 诊断与优化杠杆 |
| `AscendWiki: techniques/multi-core-scheduling` | Goldilocks 网格、GetBlockNum 不硬编码 |
| `AscendWiki: hardware/davinci-910b2` | 24 Cube / 48 Vector / 192KB UB 硬件真值 |
| `npu-arch: npu-hardware-params.md` | DAV_2201 参数 |
| `ascendc-api-best-practices: api-restrictions.md` | GetValue/SetValue 黑名单、std:: 禁用、静态分配 |
| `ascendc-api-best-practices: api-datacopy.md` | DataCopy 32B 对齐 / DataCopyPad |
| `ops-precision-standard: integer_compute.md` | 整数计算类 bit-exact 标准 |
