"""
Pure PyTorch implementation of MTPBlock from
DeepSeek-V4-Pro/inference/model.py.

Simplifications vs the original:
  - No tensor parallelism (world_size = 1)
  - No FP8 / FP4 quantisation (Linear -> nn.Linear in BF16)
  - No Hadamard rotation / act_quant
  - hc_split_sinkhorn: inline PyTorch (equiv to big_fuse_torch.py pattern)
  - Attention: supports prefill (start_pos=0) only; no KV-cache / Compressor
"""

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModelArgs:
    vocab_size:          int   = 129280
    dim:                 int   = 512          # reduced for demo
    moe_inter_dim:       int   = 512
    n_layers:            int   = 1
    n_heads:             int   = 8
    n_routed_experts:    int   = 8
    n_shared_experts:    int   = 1
    n_activated_experts: int   = 2
    q_lora_rank:         int   = 256
    head_dim:            int   = 64
    rope_head_dim:       int   = 32
    norm_eps:            float = 1e-6
    o_groups:            int   = 2
    o_lora_rank:         int   = 128
    window_size:         int   = 8
    hc_mult:             int   = 4
    hc_sinkhorn_iters:   int   = 20
    hc_eps:              float = 1e-6
    rope_theta:          float = 10000.0
    max_seq_len:         int   = 64
    # MTP specific
    score_func:          str   = "sqrtsoftplus"
    route_scale:         float = 1.0
    swiglu_limit:        float = 0.0
    n_hash_layers:       int   = 0


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


@lru_cache(2)
def precompute_freqs_cis(
    dim: int, seqlen: int, base: float = 10000.0
) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    y = x
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if xc.ndim == 3:
        freqs_cis = freqs_cis.view(1, xc.size(1), xc.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
    xc = torch.view_as_real(xc * freqs_cis).flatten(-2)
    y.copy_(xc)
    return y


# ---------------------------------------------------------------------------
# hc_split_sinkhorn — pure PyTorch
# Equivalent to hc_split_sinkhorn_kernel in kernel.py
# ---------------------------------------------------------------------------

def hc_split_sinkhorn_ref(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Splits mixes into (pre, post, comb) and doubly-stochastic normalises comb.

    mixes:    [b, s, (2+hc)*hc]  float32
    hc_scale: [3]                float32
    hc_base:  [(2+hc)*hc]        float32
    """
    hc = hc_mult
    # Expand scale to match each segment
    scale_exp = torch.cat([
        hc_scale[0].expand(hc),
        hc_scale[1].expand(hc),
        hc_scale[2].expand(hc * hc),
    ])  # [(2+hc)*hc]
    x = mixes * scale_exp + hc_base   # [b, s, (2+hc)*hc]

    pre  = torch.sigmoid(x[..., :hc]) + eps               # [b, s, hc]
    post = 2.0 * torch.sigmoid(x[..., hc:2 * hc])         # [b, s, hc]
    comb = x[..., 2 * hc:].view(*x.shape[:-1], hc, hc)    # [b, s, hc, hc]

    # Sinkhorn: softmax → col-norm → (row-norm → col-norm) × (iters-1)
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb


def sparse_attn_ref(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """
    Pure PyTorch equivalent of sparse_attn_kernel (kernel.py).

    TileLang uses block-wise online softmax (FlashAttention style).
    This implementation is mathematically equivalent using a single-pass
    masked softmax with an attention sink contributing only to the denominator.

    Args:
        q:            [b, m, h, d]  bfloat16
        kv:           [b, n, d]     bfloat16  (shared key-value per position)
        attn_sink:    [h]           float32   (learnable sink; only in denominator)
        topk_idxs:    [b, m, topk]  int32     (-1 = invalid / padding)
        softmax_scale: float        (typically head_dim ** -0.5)

    Returns:
        o: [b, m, h, d] bfloat16
    """
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]

    valid_mask = topk_idxs >= 0                                # [b, m, topk]
    safe_idxs  = topk_idxs.clamp(min=0).long()                # replace -1 with 0 for safe gather

    # Gather KV: [b, m, topk, d]
    b_idx = torch.arange(b, device=q.device)[:, None, None].expand(b, m, topk)
    gathered_kv = kv[b_idx, safe_idxs]                        # [b, m, topk, d]
    # Zero out positions that came from invalid (-1) indices
    gathered_kv = gathered_kv.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

    # Attention scores: [b, m, h, topk]
    scores = torch.einsum("bmhd,bmtd->bmht",
                          q.float(), gathered_kv.float()) * softmax_scale
    # Mask invalid positions to -inf so they don't affect softmax
    scores = scores.masked_fill(~valid_mask.unsqueeze(2), float("-inf"))

    # Numerically stable softmax with attn_sink only in the denominator.
    # Equivalent to the TileLang kernel line:
    #   sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
    sink = attn_sink.float().view(1, 1, h, 1)                 # broadcast over b, m, topk

    max_scores = torch.amax(scores, dim=-1, keepdim=True)     # [b, m, h, 1]
    # When all topk positions are invalid, max_scores = -inf; clamp with sink to stay finite
    max_scores = torch.maximum(max_scores, sink)

    exp_scores = torch.exp(scores - max_scores)
    # Re-zero invalid positions (exp(-inf - finite) = 0, but be explicit)
    exp_scores = exp_scores.masked_fill(~valid_mask.unsqueeze(2), 0.0)

    exp_sink  = torch.exp(sink - max_scores)                  # [b, m, h, 1]
    sum_exp   = exp_scores.sum(dim=-1, keepdim=True) + exp_sink

    attn_weights = exp_scores / sum_exp                        # [b, m, h, topk]

    # Weighted sum of gathered KV
    output = torch.einsum("bmht,bmtd->bmhd",
                          attn_weights, gathered_kv.float())   # [b, m, h, d]
    return output.to(q.dtype)

# ---------------------------------------------------------------------------
# Attention (simplified: prefill only, no KV-cache, no Compressor/Indexer)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads        = args.n_heads
        self.q_lora_rank    = args.q_lora_rank
        self.head_dim       = args.head_dim
        self.rope_head_dim  = args.rope_head_dim
        self.nope_head_dim  = args.head_dim - args.rope_head_dim
        self.n_groups       = args.o_groups
        self.o_lora_rank    = args.o_lora_rank
        self.window_size    = args.window_size
        self.softmax_scale  = args.head_dim ** -0.5
        self.norm_eps       = args.norm_eps

        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        self.wq_a  = nn.Linear(args.dim, args.q_lora_rank, bias=False, dtype=torch.bfloat16)
        self.q_norm = RMSNorm(args.q_lora_rank, args.norm_eps)
        self.wq_b  = nn.Linear(args.q_lora_rank, args.n_heads * args.head_dim, bias=False, dtype=torch.bfloat16)
        self.wkv   = nn.Linear(args.dim, args.head_dim, bias=False, dtype=torch.bfloat16)
        self.kv_norm = RMSNorm(args.head_dim, args.norm_eps)
        self.wo_a  = nn.Linear(
            args.n_heads * args.head_dim // args.o_groups,
            args.o_groups * args.o_lora_rank, bias=False, dtype=torch.bfloat16,
        )
        self.wo_b  = nn.Linear(args.o_groups * args.o_lora_rank, args.dim, bias=False, dtype=torch.bfloat16)

        freqs = precompute_freqs_cis(args.rope_head_dim, args.max_seq_len, args.rope_theta)
        self.register_buffer("freqs_cis", freqs, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]
        rd = self.rope_head_dim

        # Q
        q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.norm_eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        # KV
        kv = self.kv_norm(self.wkv(x))            # [b, s, head_dim]
        apply_rotary_emb(kv[..., -rd:], freqs_cis)

        # Causal sliding-window topk indices (dense, for prefill)
        win = self.window_size
        s_idx = torch.arange(seqlen, device=x.device)
        base  = s_idx.unsqueeze(1)                 # [s, 1]
        cols  = (base - win + 1).clamp(0) + torch.arange(min(seqlen, win), device=x.device)
        topk_idxs = torch.where(cols > base, torch.full_like(cols, -1), cols)
        topk_idxs = topk_idxs.unsqueeze(0).expand(bsz, -1, -1).int()  # [b, s, win]

        o = sparse_attn_ref(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)

        # De-rotate rope dims of output
        apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)

        # Output projection (grouped low-rank)
        o = o.view(bsz, seqlen, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2))


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------

class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False, dtype=torch.bfloat16)
        self.w2 = nn.Linear(inter_dim, dim, bias=False, dtype=torch.bfloat16)
        self.w3 = nn.Linear(dim, inter_dim, bias=False, dtype=torch.bfloat16)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up   = self.w3(x).float()
        if self.swiglu_limit > 0:
            up   = up.clamp(-self.swiglu_limit, self.swiglu_limit)
            gate = gate.clamp(max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.topk       = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias   = nn.Parameter(torch.zeros(args.n_routed_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        orig = scores
        scores = scores + self.bias
        indices = scores.topk(self.topk, dim=-1)[1]
        weights = orig.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights * self.route_scale, indices


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim                = args.dim
        self.n_routed_experts   = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.gate    = Gate(args)
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim, args.swiglu_limit)
            for _ in range(args.n_routed_experts)
        ])
        self.shared_experts = Expert(args.dim, args.moe_inter_dim)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            y[idx] += self.experts[i](x[idx], weights[idx, top, None])
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)


# ---------------------------------------------------------------------------
# Block (Hyper-Connections)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm_eps          = args.norm_eps
        self.hc_mult           = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps            = args.hc_eps

        self.attn      = Attention(args)
        self.ffn       = MoE(args)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm  = RMSNorm(args.dim, args.norm_eps)

        hc      = args.hc_mult
        mix_hc  = (2 + hc) * hc
        hc_dim  = hc * args.dim
        # HC params are float32 in the original (created inside set_dtype(torch.float32))
        self.hc_attn_fn    = nn.Parameter(torch.empty(mix_hc, hc_dim,  dtype=torch.float32))
        self.hc_ffn_fn     = nn.Parameter(torch.empty(mix_hc, hc_dim,  dtype=torch.float32))
        self.hc_attn_base  = nn.Parameter(torch.zeros(mix_hc,          dtype=torch.float32))
        self.hc_ffn_base   = nn.Parameter(torch.zeros(mix_hc,          dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.ones(3,                 dtype=torch.float32))
        self.hc_ffn_scale  = nn.Parameter(torch.ones(3,                 dtype=torch.float32))
        nn.init.normal_(self.hc_attn_fn, std=1e-4)
        nn.init.normal_(self.hc_ffn_fn,  std=1e-4)

    # ------------------------------------------------------------------
    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: [b, s, hc, d] → layer_input: [b, s, d], post: [b, s, hc], comb: [b, s, hc, hc]"""
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2).float()                       # [b, s, hc*d]  float32
        rsqrt  = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes  = F.linear(x_flat, hc_fn.float()) * rsqrt   # both float32 → [b, s, mix_hc]
        pre, post, comb = hc_split_sinkhorn_ref(
            mixes, hc_scale, hc_base,
            self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps,
        )
        # Weighted sum of HC copies using pre weights: [b, s, d]
        y = torch.sum(pre.unsqueeze(-1) * x.float(), dim=2)
        return y.to(dtype), post, comb

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """x: [b,s,d], residual: [b,s,hc,d] → [b,s,hc,d]"""
        return (
            post.unsqueeze(-1) * x.unsqueeze(-2)
            + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        ).type_as(x)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Attention sub-block
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos)
        x = self.hc_post(x, residual, post, comb)

        # FFN sub-block
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


# ---------------------------------------------------------------------------
# ParallelHead (simplified, no TP)
# ---------------------------------------------------------------------------

class ParallelHead(nn.Module):
    def __init__(self, vocab_size: int, dim: int, norm_eps: float, hc_eps: float):
        super().__init__()
        self.hc_eps  = hc_eps
        self.norm_eps = norm_eps
        self.weight  = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.float32))

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> torch.Tensor:
        """x: [b, s, hc, d] → [b, s, d]"""
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2).float()
        rsqrt  = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes  = F.linear(x_flat, hc_fn.float()) * rsqrt
        # hc_scale is scalar here (shape [1]), hc_base is [hc]
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        return torch.sum(pre.unsqueeze(-1) * x.float(), dim=2).to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm: RMSNorm,
    ) -> torch.Tensor:
        x = self.hc_head(x, hc_fn, hc_scale, hc_base)
        return F.linear(norm(x[:, -1]).float(), self.weight)


# ---------------------------------------------------------------------------
# MTPBlock
# ---------------------------------------------------------------------------

class MTPBlock(Block):
    """
    Multi-Token Prediction block.
    Extends Block with an extra embedding fusion step and an lm_head output.
    """

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.e_proj = nn.Linear(args.dim, args.dim, bias=False, dtype=torch.bfloat16)
        self.h_proj = nn.Linear(args.dim, args.dim, bias=False, dtype=torch.bfloat16)
        self.enorm  = RMSNorm(args.dim, args.norm_eps)
        self.hnorm  = RMSNorm(args.dim, args.norm_eps)
        self.norm   = RMSNorm(args.dim, args.norm_eps)

        hc     = args.hc_mult
        hc_dim = hc * args.dim
        # float32 to match original (created inside set_dtype(torch.float32))
        self.hc_head_fn    = nn.Parameter(torch.empty(hc, hc_dim, dtype=torch.float32))
        self.hc_head_base  = nn.Parameter(torch.zeros(hc,         dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.ones(1,           dtype=torch.float32))
        nn.init.normal_(self.hc_head_fn, std=1e-4)

        # These are set externally (shared with the main model)
        self.embed: nn.Embedding = None
        self.head:  ParallelHead = None

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:         [b, s, hc, d]  bfloat16
            start_pos: int
            input_ids: [b, s]         long

        Returns:
            logits: [b, vocab_size]   float32
        """
        assert self.embed is not None and self.head is not None, \
            "MTPBlock.embed and MTPBlock.head must be set before calling forward"

        e = self.enorm(self.embed(input_ids))           # [b, s, d]
        x = self.hnorm(x)                               # [b, s, hc, d]  (RMSNorm on last dim)
        # e_proj: [b, s, d] → [b, s, 1, d]; h_proj: [b, s, hc, d]
        x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)
        x = super().forward(x, start_pos, input_ids)   # Block (HC + attn + ffn)
        logits = self.head(
            x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm
        )
        return logits


# ---------------------------------------------------------------------------
# Wrapper Model for demo format
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """
    Self-contained wrapper for MTPBlock in the TileKernels demo format.
    Includes an embedding table and lm_head so forward() returns logits.
    """

    def __init__(self, args: ModelArgs = None):
        super().__init__()
        if args is None:
            args = ModelArgs()
        self.args    = args
        self.embed   = nn.Embedding(args.vocab_size, args.dim, dtype=torch.bfloat16)
        self.head    = ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
        self.mtp     = MTPBlock(args)
        self.mtp.embed = self.embed
        self.mtp.head  = self.head

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:         [b, s, hc, d]  bfloat16  (HC-expanded hidden state)
            input_ids: [b, s]         long

        Returns:
            logits: [b, vocab_size]   float32
        """
        return self.mtp(x, 0, input_ids)


# ---------------------------------------------------------------------------
# Demo inputs
# ---------------------------------------------------------------------------
_args = ModelArgs()

batch_size = 1
seq_len    = 8


def get_inputs():
    x         = torch.randn(batch_size, seq_len, _args.hc_mult, _args.dim, dtype=torch.bfloat16)
    input_ids = torch.randint(0, _args.vocab_size, (batch_size, seq_len))
    return [x, input_ids]


def get_init_inputs():
    return []   # Model() uses default ModelArgs internally