# DeepSeek-V4 模型结构

> 源文件：`skills/skills/deepseek_v4/model.py`

## 整体结构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Transformer (DeepSeek-V4)                    │
│                                                                     │
│  input_ids [B, S]                                                   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────┐                                            │
│  │  ParallelEmbedding  │  F.embedding + dist.all_reduce (TP)        │
│  └─────────────────────┘                                            │
│       │  [B, S, D]                                                  │
│       ▼                                                             │
│  unsqueeze + repeat  →  [B, S, hc_mult, D]   (Hyper-Connection扩展) │
│       │                                                             │
│       ▼                                                             │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Block × N_layers  (Hyper-Connection Transformer Block)       │  │
│  │                                                               │  │
│  │  x [B,S,hc,D]                                                │  │
│  │    │                                                          │  │
│  │    ├──── hc_pre (Attn) ──────────────────────────────────┐   │  │
│  │    │      F.linear(hc_fn) · rsqrt                        │   │  │
│  │    │      hc_split_sinkhorn  → pre, post, comb           │   │  │
│  │    │      weighted_sum → y [B,S,D]                       │   │  │
│  │    │                                                      │   │  │
│  │    ▼                                                      │   │  │
│  │  ┌──────────────────────────────────────────────────┐    │   │  │
│  │  │  Attention (MLA - Multi-head Latent Attention)   │    │   │  │
│  │  │                                                  │    │   │  │
│  │  │  attn_norm: RMSNorm                              │    │   │  │
│  │  │       │                                         │    │   │  │
│  │  │  Q路径:                                          │    │   │  │
│  │  │   wq_a: Linear(D → q_lora_rank)                 │    │   │  │
│  │  │   q_norm: RMSNorm                               │    │   │  │
│  │  │   wq_b: ColumnParallelLinear(q_lora→n_h·hd)    │    │   │  │
│  │  │   rsqrt(q²均值+eps)  [动态归一化]               │    │   │  │
│  │  │   apply_rotary_emb (RoPE/YaRN)                  │    │   │  │
│  │  │                                                  │    │   │  │
│  │  │  KV路径:                                         │    │   │  │
│  │  │   wkv: Linear(D → head_dim)                     │    │   │  │
│  │  │   kv_norm: RMSNorm                              │    │   │  │
│  │  │   apply_rotary_emb (RoPE/YaRN)                  │    │   │  │
│  │  │   act_quant [FP8量化, nope部分]                  │    │   │  │
│  │  │                                                  │    │   │  │
│  │  │  KV压缩 (可选, compress_ratio∈{4,128}):          │    │   │  │
│  │  │  ┌─────────────────────────────────────┐        │    │   │  │
│  │  │  │  Compressor                         │        │    │   │  │
│  │  │  │   wkv/wgate: Linear (fp32)          │        │    │   │  │
│  │  │  │   + ape (绝对位置编码)               │        │    │   │  │
│  │  │  │   softmax pooling (门控加权池化)     │        │    │   │  │
│  │  │  │   norm: RMSNorm                     │        │    │   │  │
│  │  │  │   apply_rotary_emb (RoPE)           │        │    │   │  │
│  │  │  │   act_quant / fp4_act_quant         │        │    │   │  │
│  │  │  │   → 写入 kv_cache                   │        │    │   │  │
│  │  │  └─────────────────────────────────────┘        │    │   │  │
│  │  │                                                  │    │   │  │
│  │  │  稀疏索引 (compress_ratio==4):                   │    │   │  │
│  │  │  ┌─────────────────────────────────────┐        │    │   │  │
│  │  │  │  Indexer                            │        │    │   │  │
│  │  │  │   wq_b: ColumnParallelLinear        │        │    │   │  │
│  │  │  │   apply_rotary_emb                  │        │    │   │  │
│  │  │  │   hadamard_transform (旋转激活)      │        │    │   │  │
│  │  │  │   fp4_act_quant                     │        │    │   │  │
│  │  │  │   weights_proj: ColumnParallelLinear│        │    │   │  │
│  │  │  │   einsum(Q·compKV) + relu           │        │    │   │  │
│  │  │  │   dist.all_reduce (TP)              │        │    │   │  │
│  │  │  │   topk → topk_idxs                 │        │    │   │  │
│  │  │  └─────────────────────────────────────┘        │    │   │  │
│  │  │                                                  │    │   │  │
│  │  │  sparse_attn(Q, KV, attn_sink,                  │    │   │  │
│  │  │             topk_idxs, scale)  [自定义kernel]   │    │   │  │
│  │  │                                                  │    │   │  │
│  │  │  apply_rotary_emb(逆, RoPE去旋转输出)            │    │   │  │
│  │  │                                                  │    │   │  │
│  │  │  O路径:                                          │    │   │  │
│  │  │   reshape → [B,S,n_groups,hd_per_group]         │    │   │  │
│  │  │   einsum(wo_a): grouped low-rank投影 (BF16)     │    │   │  │
│  │  │   wo_b: RowParallelLinear + dist.all_reduce     │    │   │  │
│  │  └──────────────────────────────────────────────────┘    │   │  │
│  │                                                           │   │  │
│  │    hc_post (Attn): post·x + sum(comb·residual)  ◄────────┘   │  │
│  │       │  [B,S,hc,D]                                           │  │
│  │       ├──── hc_pre (FFN) ────────────────────────────────┐    │  │
│  │       │                                                   │    │  │
│  │       ▼                                                   │    │  │
│  │  ┌──────────────────────────────────────────────────┐    │    │  │
│  │  │  MoE (Mixture of Experts)                        │    │    │  │
│  │  │                                                  │    │    │  │
│  │  │  ffn_norm: RMSNorm                               │    │    │  │
│  │  │       │                                         │    │    │  │
│  │  │  ┌─────────────────────────────────────────┐   │    │    │  │
│  │  │  │  Gate                                   │   │    │    │  │
│  │  │  │   linear(x.float(), weight.float())     │   │    │    │  │
│  │  │  │   score_func:                           │   │    │    │  │
│  │  │  │     softmax / sigmoid /                 │   │    │    │  │
│  │  │  │     softplus().sqrt() [sqrtsoftplus]    │   │    │    │  │
│  │  │  │   + bias (负载均衡偏置)                  │   │    │    │  │
│  │  │  │   topk(n_activated) → indices,weights  │   │    │    │  │
│  │  │  └─────────────────────────────────────────┘   │    │    │  │
│  │  │       │                                         │    │    │  │
│  │  │       ▼                                         │    │    │  │
│  │  │  n_routed_experts个路由专家 (TP分片):            │    │    │  │
│  │  │  ┌─────────────────────────────────────────┐   │    │    │  │
│  │  │  │  Expert (SwiGLU FFN)                    │   │    │    │  │
│  │  │  │   w1: Linear(D→inter) gate投影          │   │    │    │  │
│  │  │  │   w3: Linear(D→inter) up投影            │   │    │    │  │
│  │  │  │   clamp (swiglu_limit, 可选)            │   │    │    │  │
│  │  │  │   F.silu(gate) * up                    │   │    │    │  │
│  │  │  │   * routing_weight                     │   │    │    │  │
│  │  │  │   w2: Linear(inter→D) down投影          │   │    │    │  │
│  │  │  └─────────────────────────────────────────┘   │    │    │  │
│  │  │  dist.all_reduce (专家TP合并)                   │    │    │  │
│  │  │  + shared_expert (SwiGLU, 无swiglu_limit)      │    │    │  │
│  │  └──────────────────────────────────────────────────┘    │    │  │
│  │                                                            │    │  │
│  │    hc_post (FFN) ◄─────────────────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│       │  [B, S, hc, D]                                              │
│       ▼                                                             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  ParallelHead                                                │   │
│  │   hc_head: F.linear(hc_fn) · rsqrt · sigmoid → 加权求和     │   │
│  │   norm: RMSNorm                                              │   │
│  │   F.linear(weight, fp32 lm_head)  → logits                  │   │
│  │   dist.all_gather (vocab TP合并)                             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│       │  logits [B, vocab_size]                                     │
│                                                                     │
│  MTPBlock (Multi-Token Prediction, 可选):                           │
│   embed(next_token) → enorm(RMSNorm)                               │
│   hnorm(RMSNorm) → e_proj + h_proj → Block → ParallelHead          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 模块层级说明

### Transformer（顶层）

| 子模块 | 类型 | 说明 |
|--------|------|------|
| `embed` | `ParallelEmbedding` | 词表按 TP 分片，all_reduce 合并 |
| `layers` | `Block × N_layers` | 主干 Transformer 层 |
| `norm` | `RMSNorm` | 输出归一化 |
| `head` | `ParallelHead` | 分片 LM Head，all_gather 合并 logits |
| `mtp` | `MTPBlock × N_mtp` | 多 token 预测辅助头（共享 embed/head） |
| `hc_head_fn/base/scale` | `Parameter (fp32)` | 顶层 Hyper-Connection 头权重 |

---

### Block（Hyper-Connection Transformer 块）

```
x [B,S,hc,D]
  │
  ├─ hc_pre(attn)  →  weighted_sum → [B,S,D],  post, comb
  │    └─ F.linear + rsqrt + hc_split_sinkhorn
  │
  ├─ attn_norm (RMSNorm)
  ├─ Attention (MLA)
  ├─ hc_post(attn): post·x + Σ(comb·residual)  → [B,S,hc,D]
  │
  ├─ hc_pre(ffn)
  ├─ ffn_norm (RMSNorm)
  ├─ MoE
  └─ hc_post(ffn)
```

---

### Attention（MLA）

```
输入 x [B,S,D]
  │
  ├─ Q路径
  │   wq_a → Linear(D → q_lora_rank)
  │   q_norm → RMSNorm
  │   wq_b → ColumnParallelLinear → [B,S,n_heads,head_dim]
  │   动态归一化: q *= rsqrt(mean(q²) + eps)
  │   apply_rotary_emb(rope部分) [RoPE/YaRN]
  │
  ├─ KV路径
  │   wkv → Linear(D → head_dim)
  │   kv_norm → RMSNorm
  │   apply_rotary_emb(rope部分)
  │   act_quant(nope部分, FP8, block=64)  ← 仅量化非RoPE维度
  │
  ├─ KV Cache写入（滑窗循环缓冲）
  │
  ├─ Compressor（compress_ratio > 0）
  │   门控加权池化 → 压缩KV → act_quant/fp4_act_quant → kv_cache压缩区
  │
  ├─ Indexer（compress_ratio == 4）
  │   wq_b + RoPE + hadamard_transform + fp4_act_quant
  │   weights_proj · einsum(Q·compKV) · relu → topk → topk_idxs
  │
  ├─ sparse_attn(Q, KV, attn_sink, topk_idxs, scale)  [自定义稀疏注意力kernel]
  │
  ├─ apply_rotary_emb(逆)  ← 对输出去旋转
  │
  └─ O路径
      reshape → [B,S,n_groups, head_dim//n_groups]
      einsum("bsgd,grd->bsgr", wo_a)  [分组低秩, BF16]
      wo_b → RowParallelLinear + dist.all_reduce
```

---

### MoE（混合专家）

```
输入 x [B,S,D]
  │
  ├─ Gate
  │   linear(x.fp32, weight.fp32)
  │   打分: softmax | sigmoid | softplus().sqrt()
  │   + bias (负载均衡偏置, 仅影响选择不影响权重)
  │   topk(n_activated_experts) → indices, weights
  │
  ├─ n_routed_experts 路由专家（TP分片，每卡持有 n/world_size 个）
  │   Expert: w1(gate) + w3(up) → clamp → silu(gate)*up → *weight → w2(down)
  │   dist.all_reduce (汇聚各卡专家输出)
  │
  └─ shared_expert (1个常驻共享专家, SwiGLU, 无 swiglu_limit)
```

---

### Compressor（KV 压缩模块）

```
输入 x [B,S,D], start_pos
  │
  ├─ wkv(fp32):   Linear(D → coff·head_dim)
  ├─ wgate(fp32): Linear(D → coff·head_dim)
  ├─ + ape (可学习绝对位置编码, fp32)
  ├─ softmax(score, dim=time) × kv → 加权求和 (门控时序池化)
  │   overlap=True(ratio=4): 交叠窗口双路压缩
  ├─ norm: RMSNorm
  ├─ apply_rotary_emb(RoPE, 压缩位置)
  └─ act_quant(nope, FP8) 或 fp4_act_quant(全部, FP4) + 写入 kv_cache
```

---

### Indexer（稀疏注意力索引）

```
输入 x [B,S,D], qr [B,S,q_lora_rank], start_pos
  │
  ├─ wq_b: ColumnParallelLinear(q_lora → n_heads·head_dim)
  ├─ apply_rotary_emb(RoPE)
  ├─ hadamard_transform(scale=dim^-0.5)  ← 随机 Hadamard 旋转
  ├─ fp4_act_quant(Q, block=32)
  ├─ 内部 Compressor（含 Hadamard 旋转）构建压缩 KV
  ├─ weights_proj: ColumnParallelLinear(D → n_heads)
  ├─ einsum("bshd,btd→bsht", Q, compKV)  · relu · weights → index_score
  ├─ dist.all_reduce (TP 汇聚)
  └─ topk(index_topk) → topk_idxs
```

---

### MTPBlock（多 Token 预测块）

```
输入 x [B,S,hc,D]（前序层隐状态）, input_ids（下一 token）
  │
  ├─ embed(input_ids) → enorm(RMSNorm) → e
  ├─ hnorm(RMSNorm) → h
  ├─ e_proj(e).unsqueeze(2) + h_proj(h)   → 融合嵌入与隐状态
  ├─ super().forward()  → Block（完整 HC+MLA+MoE）
  └─ head.forward()     → ParallelHead → logits [B, vocab_size]
```

---

## 算子汇总表

| 类别 | 算子 | 所在位置 |
|------|------|----------|
| **嵌入** | `F.embedding` | `ParallelEmbedding` |
| **归一化** | `RMSNorm`（rsqrt + 均值 + weight缩放） | Q/KV/FFN/输出等多处 |
| **线性变换** | `F.linear` / `fp8_gemm` / `fp4_gemm` | `Linear`、`ColumnParallelLinear`、`RowParallelLinear` |
| **量化** | `act_quant`（FP8）、`fp4_act_quant`（FP4） | KV写缓存、Indexer、Compressor |
| **位置编码** | `apply_rotary_emb`（RoPE / YaRN 复数旋转） | Q、KV、输出去旋转、Compressor、Indexer |
| **注意力** | `sparse_attn`（自定义稀疏 kernel） | `Attention` |
| **KV压缩** | softmax 门控加权时序池化 | `Compressor` |
| **稀疏索引** | `einsum` + `relu` + `topk` | `Indexer` |
| **Hadamard旋转** | `hadamard_transform`（scale=dim^-0.5） | `Indexer`、`Compressor`（rotate=True） |
| **Hyper-Connection** | `hc_split_sinkhorn`、sigmoid 加权求和 | `Block.hc_pre/hc_post`、`ParallelHead.hc_head` |
| **MoE路由** | `softmax` / `sigmoid` / `softplus().sqrt()`、`topk`、`bincount` | `Gate` |
| **激活函数** | `F.silu`（SwiGLU gate）、`relu`、`sigmoid` | `Expert`、`Gate`、HC头 |
| **专家FFN** | w1/w3 上投影 + silu 门控 + w2 下投影（SwiGLU） | `Expert` |
| **输出投影** | `einsum("bsgd,grd->bsgr")`（分组低秩） | `Attention.wo_a` |
| **分布式通信** | `dist.all_reduce`、`dist.all_gather` | TP 并行各处（Embedding、专家、Indexer、Head） |

---

## 量化精度说明

| 组件 | 权重精度 | 激活精度 |
|------|----------|----------|
| 主干 Linear | FP8 (e4m3) 或 BF16 | FP8（act_quant 动态量化） |
| 专家 Expert | FP4 (e2m1) 或 BF16 | FP8（act_quant） |
| Compressor wkv/wgate | FP32 | FP32 |
| Indexer Q/KV | — | FP4（fp4_act_quant） |
| RMSNorm weight | FP32 | — |
| LM Head weight | FP32 | FP32 |
| wo_a（输出低秩） | FP8（checkpoint） | BF16（实现中简化） |
