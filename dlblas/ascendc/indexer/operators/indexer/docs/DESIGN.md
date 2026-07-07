# Indexer 算子 Triton-Ascend Kernel 架构设计

> 版本: v1.1
> 日期: 2026-07-03
> 目标设备: Ascend910B2 (DAV_2201), CANN 9.0.0
> 编程框架: Triton-Ascend DSL (Python, @triton.jit)
> 场景: world_size=1 单卡推理

**注意**: 本文档描述的是 Triton-Ascend DSL 实现方案（Python + @triton.jit），而非 C++ AscendC API。
Triton-Ascend 提供与 AscendC 等价的硬件能力访问（Cube/Vector 单元、GM/L1 内存层级），
但通过 Python JIT 编译代替手写 C++ 算子。两个版本的核心计算逻辑和 tiling 策略保持一致。

---

## 1. 算子概述

Indexer 是 DeepSeek 风格 MoE 模型中的一个子模块，位于 Attention 层之前，负责从压缩后的 KV Cache 中**选取 top-k 个最相关的位置索引**，用于后续的稀疏注意力计算。其核心思想是用一个轻量级的"打分网络"快速评估每个 KV 位置的重要性，从而避免在全长 KV Cache 上做稠密注意力。

### 1.1 输入输出

| 名称 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `x` | (B, S, dim) | bf16 | 当前层的 hidden states |
| `qr` | (B, S, q_lora_rank) | bf16 | 低秩压缩后的 query 表示（来自上一层 Q 的 LoRA 投影） |
| `start_pos` | scalar | int | 当前 chunk 的起始位置（prefill 阶段为 0） |
| `offset` | scalar | int | 索引偏移量 |
| **输出** | (B, S, index_topk) | int64 | 每个 query 位置选出的 top-k KV 位置索引 |

### 1.2 固定权重/缓存

| 名称 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `wq_b.weight` | (n_heads * head_dim, q_lora_rank) | bf16 | Q 投影权重 |
| `weights_proj.weight` | (n_heads, dim) | bf16 | 分数权重投影 |
| `kv_cache` | (B, max_seq_len // compress_ratio, head_dim) | bf16 | 压缩后的 KV Cache |
| `freqs_cis` | (max_seq_len, rope_head_dim // 2) | complex64 | RoPE 频率表 |

---

## 2. 计算流程图

```
输入: x(B,S,dim), qr(B,S,q_lora_rank), start_pos, offset

  Stage A: 线性投影 (两个独立的 MatMul)
  ┌─────────────────────────────────────────────────────────────┐
  │  q_flat = qr @ wq_b.weight^T                               │
  │         (B,S,q_lora_rank) @ (q_lora_rank, n_heads*head_dim) │
  │         → (B, S, n_heads * head_dim)                        │
  │                                                             │
  │  weights = x @ weights_proj.weight^T                        │
  │         (B,S,dim) @ (dim, n_heads)                          │
  │         → (B, S, n_heads)                                   │
  └─────────────────────────────────────────────────────────────┘
                           │
                           ▼
  Stage B: Q 重塑 + RoPE + 打分 (核心计算)
  ┌─────────────────────────────────────────────────────────────┐
  │  q = q_flat.unflatten(-1, (n_heads, head_dim))             │
  │    → (B, S, n_heads, head_dim)                              │
  │                                                             │
  │  q[..., -rope_head_dim:] = RoPE(q[..., -rope_head_dim:])   │
  │    freqs_cis[start_pos : start_pos+S] 用于旋转              │
  │                                                             │
  │  scores = einsum("bshd,btd->bsht", q, kv_cache)            │
  │    q:       (B, S, n_heads, head_dim)                       │
  │    kv_cache:(B, kv_len, head_dim) 其中 kv_len = end_pos//ratio│
  │    scores:  (B, S, n_heads, kv_len)                         │
  └─────────────────────────────────────────────────────────────┘
                           │
                           ▼
  Stage C: 后处理 + Top-K 选择
  ┌─────────────────────────────────────────────────────────────┐
  │  scores_act = ReLU(scores)          # 正值保留，负值清零     │
  │                                                             │
  │  scores_weighted = scores_act * weights.unsqueeze(-1)       │
  │    weights:  (B, S, n_heads, 1)                             │
  │    scores_act: (B, S, n_heads, kv_len)                      │
  │                                                             │
  │  index_score = scores_weighted.sum(dim=2)  # 沿 head 维求和 │
  │    → (B, S, kv_len)                                        │
  │                                                             │
  │  if start_pos == 0:                                         │
  │    causal_mask: 对每个 query 位置 i, 屏蔽 kv 位置 j         │
  │      满足 j >= floor((i+1) / ratio) 的位置 → -inf           │
  │                                                             │
  │  topk_idxs = TopK(index_score, k=min(topk, kv_len), dim=-1) │
  │                                                             │
  │  if start_pos == 0:                                         │
  │    无效索引 → -1                                             │
  │  topk_idxs += offset                                        │
  └─────────────────────────────────────────────────────────────┘

输出: topk_idxs (B, S, k)  int64
```

---

## 3. Shape 详细分析

以 `get_inputs()` 提供的默认参数为例：

| 参数 | 值 |
|------|-----|
| B (batch_size) | 2 |
| S (seq_len) | 64 |
| dim | 1024 |
| q_lora_rank | 256 |
| n_heads (index_n_heads) | 16 |
| head_dim (index_head_dim) | 64 |
| rope_head_dim | 32 |
| index_topk | 128 |
| compress_ratio | 4 |
| max_seq_len | 1024 |

### 3.1 Prefill 阶段 (start_pos=0)

```
end_pos = S = 64
kv_len  = end_pos // ratio = 16

q_flat:     (2, 64, 1024)       # 1024 = 16*64
weights:    (2, 64, 16)
q:          (2, 64, 16, 64)
scores:     (2, 64, 16, 16)     # [B,S,H,kv_len]
weighted:   (2, 64, 16, 16)
index_score:(2, 64, 16)
topk(k=16): (2, 64, 16)         # k = min(128, 16) = 16
```

### 3.2 Decode 阶段 (start_pos>0, seqlen=1)

```
end_pos = start_pos + 1
kv_len  = end_pos // ratio ≈ start_pos // ratio (如果 start_pos 不是 ratio 对齐)

q_flat:     (2, 1, 1024)
weights:    (2, 1, 16)
q:          (2, 1, 16, 64)
scores:     (2, 1, 16, kv_len)
index_score:(2, 1, kv_len)
topk:       (2, 1, min(128, kv_len))
```

**关键观察**：
- Prefill 阶段 compute-intensive（S 可达 4096），matmul 计算量主导
- Decode 阶段 memory-bound（S=1），kernel launch overhead 更敏感
- kv_len 随 decode 步数线性增长，但最终受 max_seq_len//ratio 限制

---

## 4. 架构决策：多 Kernel 组合 vs 单一大 Kernel

### 4.1 决策：多 Kernel 组合方案

选择**拆分为 4 个独立 Triton-Ascend Kernel**，理由如下：

| 维度 | 单一大 Kernel | 多 Kernel 组合（选择） |
|------|--------------|----------------------|
| **Cube 利用率** | 不同运算争抢 Cube 单元，难以统一 tiling | 每个 MatMul 独立调优，Cube 利用率高 |
| **Vector 与 Cube 流水** | 手工编排 Vector/Cube 双流水极其复杂 | 各 Kernel 内部流水清晰 |
| **可维护性** | 代码量巨大，一处修改影响全局 | 模块化，便于调试和单独优化 |
| **Decode 场景** | 大 Kernel 对 S=1 场景浪费资源 | 可选择跳过不必要的 Kernel（如 causal mask） |
| **开发风险** | 高，调试周期长 | 低，逐步验证 |
| **Launch Overhead** | 无 | 4 个 Kernel Launch，在 ms 级可接受 |

### 4.2 Kernel 拆分总览

```
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────────┐
│ Kernel 1 │    │ Kernel 2 │    │   Kernel 3   │    │   Kernel 4   │
│ Q投影    │    │ W投影    │    │ RoPE + 打分  │    │ 后处理+TopK  │
│ (MatMul) │    │ (MatMul) │    │ (Vector+Cube)│    │ (Vector)     │
└────┬─────┘    └────┬─────┘    └──────┬───────┘    └──────┬───────┘
     │               │               │                    │
     ▼               ▼               ▼                    ▼
  q_flat         weights          scores            topk_idxs
(B,S,H*D)       (B,S,H)       (B,S,H,kv_len)       (B,S,K)
```

**数据依赖关系**：
- Kernel 1 和 Kernel 2 **无依赖**，可并行/并发执行
- Kernel 3 依赖 Kernel 1 的输出（q_flat）
- Kernel 4 依赖 Kernel 2 的输出（weights）和 Kernel 3 的输出（scores）

---

## 5. 各 Kernel 详细设计

### 5.1 Kernel 1: q_projection (MatMul)

**功能**：`qr @ wq_b.weight^T`

**Triton-Ascend DSL 实现**：使用 `@triton.jit` + `tl.dot`（Cube 单元）

```
输入:
  qr:          (M, K) = (B*S, q_lora_rank)    bf16, row-major
  wq_b_weight: (K, N) = (q_lora_rank, H*D)    预转置为 (K, N) row-major
输出:
  q_flat:      (M, N) = (B*S, H*D)            bf16, row-major

M = B * S,   K = q_lora_rank,   N = n_heads * head_dim
```

**Tiling 策略**：
- 使用 2D grid `(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))` + K 维循环
- `BLOCK_M=64, BLOCK_N=64, BLOCK_K=32`
- 每个 block 使用 `tl.dot` 在 Cube 单元上完成 (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N)
- 逐 K tile 加载 A/B 到寄存器，fp32 累加，最终转为 bf16 写回 GM

**关键实现细节**：
- wq_b.weight 在传入 kernel 前预先转置为 (K, N) row-major，避免 kernel 内 transpose
- qr 的 (B, S, q_lora_rank) 在传入前 reshape 为 (B*S, q_lora_rank)
- Triton 自动管理 GM → L1 → 寄存器的数据搬运，无需手工指定 TPosition

**性能预估**：
- 计算量: 2 × M × K × N = 2 × 8192 × 256 × 1024 ≈ 4.3 GFLOPS (prefill)
- Cube 利用率: 理论峰值约 80-90%（矩阵大小适中，对齐良好）

---

### 5.2 Kernel 2: weights_projection (MatMul)

**功能**：`x @ weights_proj.weight^T`

**Triton-Ascend DSL 实现**：复用 Kernel 1 的 `matmul_kernel`，仅 tiling 参数不同

```
输入:
  x:                  (M, K) = (B*S, dim)       bf16, row-major
  weights_proj_weight:(K, N) = (dim, n_heads)   预转置为 (K, N) row-major
输出:
  weights:            (M, N) = (B*S, n_heads)   bf16, row-major

M = B * S,   K = dim,   N = n_heads
```

**Tiling 策略**：
- `BLOCK_M=64, BLOCK_N=16, BLOCK_K=64`（N 较小，缩窄 N tile 避免浪费）
- 其他与 Kernel 1 相同

**关键实现细节**：
- 权重预转置为 (K, N) row-major
- 与 Kernel 1 无数据依赖，可在 Host 端顺序调用（或通过 NPU Stream 并发）
- 当 `n_heads` 较小时（16-64），此 MatMul 计算量不大，主要开销在数据搬运

**性能预估**：
- 计算量: 2 × M × K × N = 2 × 8192 × 1024 × 16 ≈ 0.27 GFLOPS
- Memory-bound: K×N=16384 权重较小，但 x 的 M×K=8M 元素搬运是瓶颈

---

### 5.3 Kernel 3: rope_score_compute (Fused Vector + Cube)

**功能**：Q 重塑 → RoPE → Batched MatMul（打分计算）

这是**计算量最大、设计最复杂**的 Kernel。

**子步骤**：

#### 5.3.1 Q 重塑 + RoPE

```
输入 q_flat: (B*S, H*D) → reshape → q: (B, S, H, D)

RoPE 仅作用于最后一维的后 rope_head_dim 个元素:
  q[..., -rope_head_dim:] ← apply_rotary_emb(q[..., -rope_head_dim:], freqs_cis[start_pos:start_pos+S])
```

RoPE 实现（参考 `apply_rotary_emb`）：
```
x_real = q[..., -rd:]           # (B, S, H, rd), rd=rope_head_dim
x_complex = view_as_complex(x_real)  # 将相邻两元素组成一个复数
freqs_cis = polar(1.0, freqs)        # cos + i*sin
x_rotated = x_complex * freqs_cis    # 复数乘法 = 旋转
x_rotated_real = view_as_real(x_rotated)
```

在 AscendC 中，频率表 `freqs_cis` 预先按 `(cos, sin)` 对存储在 GM 中。RoPE 退化为逐元素乘加：
```
对于每个 (b, s, h, i) 其中 i < rd//2:
  a = q[b,s,h, D-rd+2i]       # 实部
  b = q[b,s,h, D-rd+2i+1]     # 虚部
  c = freqs_cis[s, 2i]         # cos
  d = freqs_cis[s, 2i+1]       # sin
  q'[b,s,h, D-rd+2i]   = a*c - b*d
  q'[b,s,h, D-rd+2i+1] = a*d + b*c
```

**Tiling 策略（RoPE 部分）**：
- 按 (B, S, H) 维度并行化，每个 block 处理若干 (b, s, h) 组合
- 对于每个 (b, s, h)，连续读取 head_dim 个元素，仅修改后 rd 个
- Vector 计算单元直接完成乘加

#### 5.3.2 Batched MatMul: einsum("bshd,btd->bsht")

这是 Indexer 的**计算核心**。

```
对每个 (b, h) 对:
  q_slice[b,h]  : (S, D)         # 当前 batch+head 的 Q 切片
  kv_slice[b]   : (kv_len, D)    # 当前 batch 的 KV（所有 head 共享）
  score[b,h]    : (S, kv_len)    # q_slice @ kv_slice^T
```

**Triton-Ascend DSL 实现方案：Flatten 为 BatchMatMul**

```
q_perm:  (B, S, H, D) → permute → (B, H, S, D) → reshape → (B*H, S, D)
kv_bc:   (B, kv_len, D) → expand(B, H, kv_len, D) → reshape → (B*H, kv_len, D) → contiguous()

BatchMatMul via Triton kernel:
  grid = (B*H, cdiv(S, BLOCK_S), cdiv(kv_len, BLOCK_KV))
  每个 block 用 tl.dot 计算 (BLOCK_S, BLOCK_D) @ (BLOCK_D, BLOCK_KV)
```

**方案选择**：使用 3D grid Triton kernel（BLOCK_S=32, BLOCK_KV=32, BLOCK_D=64），理由：
- B*H 作为 grid 第 0 维，每个 block 独立处理一对 (bh, s_tile, kv_tile)
- tl.dot 直接在 Cube 单元执行，充分利用硬件算力
- Triton 自动管理内存层级，无需手工指定 L1/L0 搬运
- KV Cache 通过 `expand()` + `contiguous()` broadcast 到 B*H

**性能预估**：
- RoPE: O(B×S×H×rd) ≈ 2×4096×16×32 ≈ 4.2M 元素操作，纯 Vector，带宽瓶颈
- MatMul: 32 × 2 × S × D × kv_len ≈ 32 × 2 × 4096 × 64 × 256 ≈ 4.3 GFLOPS
- 总体：MatMul 是主要瓶颈，RoPE 开销可忽略

---

### 5.4 Kernel 4: postprocess_topk (Vector + TopK)

**功能**：ReLU → 加权求和 → Causal Mask → TopK

#### 5.4.1 ReLU + 加权求和 + Causal Mask (Triton kernel)

```
输入: scores (B, H, S, kv_len), weights (B*S, H)
输出: index_score (B, S, kv_len)

index_score[b,s,:] = sum_h( ReLU(scores[b,s,h,:]) * weights[b,s,h] )
if start_pos == 0:
  index_score[b,s,kv_idx] = -inf  for kv_idx >= floor((s+1)/ratio)
```

**Triton-Ascend DSL 实现**：
- 使用 `@triton.jit` fusion kernel: 3D grid `(B, S, cdiv(kv_len, BLOCK_KV))`
- 每个 block 处理一个 (b, s) 对的 BLOCK_KV 个 kv 位置
- 内层循环遍历 H 维度，逐 head 做 ReLU + 乘加（fp32 累加）
- Causal mask 在 kernel 内完成：对满足 `kv_idx >= (s+1)//compress_ratio` 的位置写入 -1e30
- Triton 自动管理 Vector 单元调度，无需手工指定 TPosition

#### 5.4.2 TopK 选择 (PyTorch NPU op)

```
topk_idxs = TopK(index_score, k=min(topk, kv_len), dim=-1)
```

**实现选择**：使用 PyTorch `torch.topk()`（底层调用 Ascend NPU 优化的 TopK 算子）

Triton-Ascend 当前不提供高效的 topk / partial-sort 原语，因此此步骤使用 PyTorch NPU 算子。
未来可考虑 k-pass argmax 实现以进一步减少 Python 调度开销。

#### 5.4.3 后处理

```
if start_pos == 0:
  mask = topk_idxs >= floor((s+1) / ratio)
  topk_idxs[mask] = -1
else:
  topk_idxs += offset
```

**实现**：PyTorch `torch.where` / `torch.arange` 逐元素操作。对于 decode 阶段（无 causal mask），仅需一次 `+ offset`，开销可忽略。

---

## 6. 数据流总览

```
GM Memory Layout
═══════════════════════════════════════════════════════════════════

  [输入 Tensor — 已在 NPU GM]
  x:          (B, S, dim)            @ GM_addr_x
  qr:         (B, S, q_lora_rank)    @ GM_addr_qr
  kv_cache:   (B, max_kv_len, D)     @ GM_addr_kv

  [权重 — 已在 NPU GM]
  wq_b_w:     (H*D, q_lora_rank)     @ GM_addr_wq      # column-major
  w_proj_w:   (n_heads, dim)         @ GM_addr_ww      # column-major
  freqs_cis:  (max_seq_len, rd)      @ GM_addr_freqs

  ───────────────────── Kernel 1 (Cube/MatMul) ─────────────────────
  q_flat:     (B*S, H*D)             @ GM_addr_q_flat   [临时]

  ───────────────────── Kernel 2 (Cube/MatMul) ─────────────────────
  weights:    (B*S, n_heads)         @ GM_addr_weights  [临时]

  ───────────────────── Kernel 3 (Vector + Cube/BatchMatMul) ───────
  scores:     (B*S, H, kv_len)       @ GM_addr_scores   [临时]

  ───────────────────── Kernel 4 (Vector) ──────────────────────────
  topk_idxs:  (B, S, K)              @ GM_addr_output   [最终输出]
```

**中间临时 tensor 内存估算（prefill B=2, S=4096）**：

| Tensor | 大小 | 说明 |
|--------|------|------|
| q_flat | B×S×H×D×2 = 2×4096×16×64×2 ≈ 16.8 MB | bf16 |
| weights | B×S×n_heads×2 = 2×4096×16×2 ≈ 0.26 MB | bf16 |
| scores | B×S×H×kv_len×2 = 2×4096×16×256×2 ≈ 67 MB | bf16 |
| **合计** | **≈ 84 MB** | 在设备 HBM 容量内 |

---

## 7. Ascend910B2 硬件适配要点

### 7.1 关键硬件参数

| 参数 | 值 |
|------|-----|
| AI Core 数量 | 32 (910B2) |
| Cube 算力 (bf16) | ~256 TFLOPS |
| Vector 算力 (bf16) | ~32 TFLOPS |
| HBM 带宽 | ~1.2 TB/s |
| L1 Buffer / Core | ~192 KB |
| L0C Buffer / Core | ~64 KB |

### 7.2 数据搬运策略

- **Kernel 1/2 (MatMul)**：使用 Triton `tl.load` / `tl.store` + `tl.dot`，Triton 编译器自动管理 GM ↔ L1 ↔ 寄存器的数据搬运和流水线编排
- **Kernel 3 (Batched MatMul)**：KV Cache 通过 `expand()` + `contiguous()` 在 Host 端 broadcast 后传入 kernel，避免 kernel 内重复从 GM 读取
- **Kernel 4 (Vector Fusion)**：按 (B, S) 维度 2D block 分派，每个 block 独立处理一行，数据局部性好，Triton 自动 cache L1 复用

### 7.3 bf16 精度考量

| 运算 | 精度风险 | 缓解措施 |
|------|---------|---------|
| MatMul (Kernel 1/2) | 低（Cube 内积支持 bf16） | 使用 AscendC 标准 MatMul |
| RoPE 复数乘法 | 中等（乘积累加） | 使用 bf16 乘法 + fp32 累加 |
| scores 加权求和 | 中等（H 维求和） | 使用 fp32 累加器，最终转 bf16 |
| TopK 比较 | 无（仅比较和索引操作） | bf16 比较无精度损失 |
| -inf 写入 | 无 | 使用 bf16 的 -inf 表示 (0xFF80) |

### 7.4 Kernel 3 的 BatchMatMul 实现细节

在 Triton-Ascend 中，使用 `@triton.jit` + `tl.dot` 实现 batched matmul：

```python
@triton.jit
def score_matmul_kernel(
    q_ptr, kv_ptr, scores_ptr,
    S, D, kv_len,
    stride_q_bh, stride_q_s, stride_q_d,
    stride_kv_bh, stride_kv_j, stride_kv_d,
    stride_sc_bh, stride_sc_s, stride_sc_j,
    BLOCK_S: tl.constexpr, BLOCK_KV: tl.constexpr, BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    s_block = tl.program_id(1)
    kv_block = tl.program_id(2)
    # Load q[bh, s_block, :] → (BLOCK_S, BLOCK_D)
    # Load kv[bh, kv_block, :] → (BLOCK_KV, BLOCK_D)
    # tl.dot(q, trans(kv)) → (BLOCK_S, BLOCK_KV)
    # Store → scores[bh, s_block, kv_block]
```

Launch grid: `(B*H, cdiv(S, BLOCK_S), cdiv(kv_len, BLOCK_KV))`

**kv_cache broadcast 策略**：
- kv_cache 原始形状: (B, kv_len, D)，按 GM 线性存储
- Host 端通过 `kv_cache.unsqueeze(1).expand(B, H, kv_len, D).reshape(B*H, kv_len, D).contiguous()` 完成 broadcast
- 广播后的 kv_bc 每个 head 拥有独立内存副本，Triton kernel 内线性寻址，无 stride 跳转开销
- 内存开销: B*H*kv_len*D*2 bytes（prefill S=4096 时约 32*256*64*2 ≈ 1MB，可接受）

---

## 8. 潜在优化方向

### 8.1 Kernel 融合

- **Kernel 1+2 融合**：将两个独立 MatMul 合并到一个 Kernel，共享 L1 Buffer 减少 GM 访问
  - 收益有限，因为 qr 和 x 是不同的输入张量
- **Kernel 3+4 融合**：将 score 计算和后处理合并，避免 score 矩阵的 GM 写回再读入
  - **值得探索**，score 矩阵 (B,S,H,kv_len) 约 67MB，避免此 GM 往返可节省约 134MB 带宽

### 8.2 Decode 阶段优化

- S=1 时，两个 MatMul 的 M=2（B=2），计算量极小，Kernel Launch Overhead 成为瓶颈
- 可考虑为 decode 专用的小 Kernel 或融合路径

### 8.3 异步执行

- Kernel 1 和 Kernel 2 无依赖，可通过 PyTorch NPU Stream 机制并发执行
- 中间 tensor 的内存管理可使用对象池，避免重复分配/释放

---

## 9. 开发文件结构

```
operators/indexer/
├── docs/
│   ├── DESIGN.md          ← 本文件
│   └── PLAN.md            ← 实施计划
├── ascendc/
│   ├── __init__.py            ← 包导出
│   ├── kernel_q_proj.py       ← Kernel 1: Q 投影 MatMul (tl.dot)
│   ├── kernel_w_proj.py       ← Kernel 2: 权重投影 MatMul (复用 K1 kernel)
│   ├── kernel_rope_score.py   ← Kernel 3: RoPE (PyTorch) + Score MatMul (tl.dot)
│   ├── kernel_post_topk.py    ← Kernel 4: ReLU+Sum+Mask (Triton fusion) + TopK (PyTorch)
│   └── indexer_launcher.py    ← Host 侧调度 + 内存管理
├── test/
│   ├── __init__.py
│   ├── torch_ref/
│   │   ├── __init__.py
│   │   └── indexer_torch.py   ← PyTorch 参考实现（精度基准）
│   ├── test_prefill.py        ← Prefill 正确性测试
│   └── test_decode.py         ← Decode 正确性测试
├── benchmark/
│   └── bench_indexer.py       ← 性能测试
└── configs/
    └── default.json           ← 默认配置
```

---

## 10. 总结

Indexer 算子的 Triton-Ascend 实现采用**4 Kernel 组合方案**：

| Kernel | 功能 | 计算单元 | 实现方式 | 复杂度 | 风险 |
|--------|------|---------|---------|--------|------|
| 1. q_projection | Q 线性投影 | Cube | `@triton.jit` + `tl.dot` | 低（标准 GEMM） | 低 |
| 2. weights_projection | 权重线性投影 | Cube | `@triton.jit` + `tl.dot`（复用 K1） | 低（标准 GEMM） | 低 |
| 3. rope_score | RoPE + Batched MatMul | Vector + Cube | RoPE: PyTorch NPU ops; MatMul: `@triton.jit` + `tl.dot` | **高** | **中** |
| 4. postprocess_topk | ReLU+Sum+Mask+TopK | Vector | Fusion: `@triton.jit`; TopK: PyTorch NPU op | 中 | 中 |

**核心挑战**在 Kernel 3 的 Batched MatMul 实现：需要正确处理 Q 的 permute/reshape、KV Cache 的 broadcast，以及 3D grid 的 tiling 策略。

**Cuurent limitations**: Kernel 3 的 RoPE 部分和 Kernel 4 的 TopK 部分仍使用 PyTorch NPU 算子，
原因是 Triton-Ascend 当前不提供复数乘法（RoPE）和 partial-sort（TopK）的高效原语。
后续可通过自定义 Triton kernel 逐步替换这些 PyTorch fallback。

**后续优化方向**：Kernel 3+4 融合减少中间数据搬运、Decode 阶段专用快速路径、Triton-ize RoPE 复数乘法。
