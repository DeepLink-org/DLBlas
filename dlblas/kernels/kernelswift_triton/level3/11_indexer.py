import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from functools import lru_cache
from contextlib import contextmanager

import triton
import triton.language as tl


world_size = 1
rank = 0
block_size = 128
fp4_block_size = 32
default_dtype = torch.bfloat16


@contextmanager
def set_dtype(dtype):
    """Temporarily override torch default dtype, restoring it on exit (even if an exception occurs)."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)

@dataclass
class ModelArgs:
    """Model hyperparameters. Field names match the config JSON keys."""
    max_batch_size: int = 4
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: Literal[None, "ue8m0"] = "ue8m0"
    expert_dtype: Literal[None, "fp4"] = None
    scale_dtype: Literal["fp32", "fp8"] = "fp8"
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 4096
    n_layers: int = 7
    n_hash_layers: int = 0
    n_mtp_layers: int = 1
    n_heads: int = 64
    # moe
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.
    swiglu_limit: float = 0.
    # mqa
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: Tuple[int] = (0, 0, 4, 128, 4, 128, 4, 0)
    # yarn
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    # index
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    # hc
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

class Linear(nn.Module):
    """Linear layer supporting BF16, FP8, and FP4 weight formats with per-block scaling."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dtype = dtype or default_dtype
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)
    
def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Dispatches to fp4_gemm / fp8_gemm / F.linear based on weight dtype.
    For quantized weights, x is first quantized to FP8 via act_quant."""
    assert bias is None
    return F.linear(x, weight)

class ColumnParallelLinear(Linear):
    """Shards output dim across TP ranks. No all-reduce needed on output."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class RowParallelLinear(Linear):
    """Shards input dim across TP ranks. All-reduce on output to sum partial results."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, None)
        if self.bias is not None:
            y += self.bias
        return y.type_as(x)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Applies rotary positional embeddings in-place. Uses conjugate for inverse (de-rotation)."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


@triton.jit
def _fused_index_score_kernel(
    Q, KV, W, OUT,
    S, H, T, D,
    stride_q_b, stride_q_s, stride_q_h, stride_q_d,
    stride_kv_b, stride_kv_t, stride_kv_d,
    stride_w_b, stride_w_s, stride_w_h,
    stride_out_b, stride_out_s, stride_out_t,
    BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_bs = tl.program_id(0)
    pid_t = tl.program_id(1)

    # Decode (b, s) from pid_bs
    b = pid_bs // S
    s = pid_bs % S

    # Offsets for T tile
    t_start = pid_t * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    t_mask = t_offsets < T
    tl.multiple_of(t_offsets, 8)

    # Base pointers
    q_bs_ptr = Q + b * stride_q_b + s * stride_q_s
    kv_b_ptr = KV + b * stride_kv_b
    w_bs_ptr = W + b * stride_w_b + s * stride_w_s
    out_bs_ptr = OUT + b * stride_out_b + s * stride_out_s

    # Accumulator over heads -> [BLOCK_T]
    acc_t = tl.zeros((BLOCK_T,), dtype=tl.float32)

    # Iterate over heads in BLOCK_H chunks
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < H

        # Accumulator for current head block producing [BLOCK_H, BLOCK_T]
        mm_acc = tl.zeros((BLOCK_H, BLOCK_T), dtype=tl.float32)

        # Double-buffered loop over D in BLOCK_D chunks
        d0_offsets = tl.arange(0, BLOCK_D)
        tl.multiple_of(d0_offsets, 8)
        d0_mask = d0_offsets < D

        # Preload first tiles
        q_ptrs0 = q_bs_ptr + h_offsets[:, None] * stride_q_h + d0_offsets[None, :] * stride_q_d
        q_mask0 = h_mask[:, None] & d0_mask[None, :]
        q_curr = tl.load(q_ptrs0, mask=q_mask0, other=0, eviction_policy="evict_last")

        kv_ptrs0 = kv_b_ptr + t_offsets[:, None] * stride_kv_t + d0_offsets[None, :] * stride_kv_d
        kv_mask0 = t_mask[:, None] & d0_mask[None, :]
        kv_curr = tl.load(kv_ptrs0, mask=kv_mask0, other=0, cache_modifier=".ca")

        # Iterate remaining tiles with prefetch of next
        for d_start in range(BLOCK_D, D, BLOCK_D):
            d_offsets_next = d0_offsets + d_start
            d_mask_next = d_offsets_next < D

            # Prefetch next tiles
            q_ptrs_next = q_bs_ptr + h_offsets[:, None] * stride_q_h + d_offsets_next[None, :] * stride_q_d
            q_mask_next = h_mask[:, None] & d_mask_next[None, :]
            q_next = tl.load(q_ptrs_next, mask=q_mask_next, other=0, eviction_policy="evict_last")

            kv_ptrs_next = kv_b_ptr + t_offsets[:, None] * stride_kv_t + d_offsets_next[None, :] * stride_kv_d
            kv_mask_next = t_mask[:, None] & d_mask_next[None, :]
            kv_next = tl.load(kv_ptrs_next, mask=kv_mask_next, other=0, cache_modifier=".ca")

            # Compute on current tiles
            mm_acc += tl.dot(q_curr, tl.trans(kv_curr))

            # Advance buffers
            q_curr = q_next
            kv_curr = kv_next

        # Final compute for the last preloaded tiles
        mm_acc += tl.dot(q_curr, tl.trans(kv_curr))

        # Apply ReLU in fp32
        mm_relu = tl.maximum(mm_acc, 0.0)

        # Load weights for current head block: [BLOCK_H] (fp32)
        w_ptrs = w_bs_ptr + h_offsets * stride_w_h
        w_tile = tl.load(w_ptrs, mask=h_mask, other=0.0).to(tl.float32)

        # Row-wise scale by weights and reduce over heads -> [BLOCK_T]
        mm_scaled = mm_relu * w_tile[:, None]
        row_sum = tl.sum(mm_scaled, axis=0)

        acc_t += row_sum

    # Store result for this (b, s) and t tile in bf16 to reduce BW
    out_ptrs = out_bs_ptr + t_offsets * stride_out_t
    tl.store(out_ptrs, acc_t.to(tl.bfloat16), mask=t_mask)


def _compute_index_score_triton(q_4d: torch.Tensor, kv: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    q_4d: [B, S, H, D] (bf16)
    kv:   [B, T, D] (bf16)
    w:    [B, S, H] (bf16 or fp32) -> promoted to fp32
    Returns: [B, S, T] (bf16)
    """
    B, S, H, D = q_4d.shape
    T = kv.shape[1]
    assert kv.shape == (B, T, D)
    assert w.shape == (B, S, H)

    if T == 0:
        return torch.empty((B, S, 0), device=q_4d.device, dtype=q_4d.dtype)

    # Ensure contiguous layouts and dtypes
    q = q_4d.contiguous()
    kv = kv.contiguous()
    w = w.contiguous().to(torch.float32)

    out = torch.empty((B, S, T), device=q.device, dtype=q.dtype)

    # Heuristic tiling tuned for small-to-medium T and typical D=64
    if T <= 32:
        BLOCK_T = 32
        num_warps = 2
    elif T <= 64:
        BLOCK_T = 64
        num_warps = 2
    else:
        BLOCK_T = 128
        num_warps = 4

    # Choose head tile as next power-of-two not exceeding 32
    if H <= 8:
        BLOCK_H = 8
    elif H <= 16:
        BLOCK_H = 16
    else:
        BLOCK_H = 32

    # Depth tile
    BLOCK_D = 64 if D >= 64 else 32

    grid = (B * S, triton.cdiv(T, BLOCK_T))

    _fused_index_score_kernel[grid](
        q, kv, w, out,
        S, H, T, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        kv.stride(0), kv.stride(1), kv.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_T=BLOCK_T, BLOCK_H=BLOCK_H, BLOCK_D=BLOCK_D,
        num_warps=num_warps, num_stages=4
    )
    return out


class ColumnParallelLinear(Linear):
    """Shards output dim across TP ranks. No all-reduce needed on output."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class RowParallelLinear(Linear):
    """Shards input dim across TP ranks. All-reduce on output to sum partial results."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, None)
        if self.bias is not None:
            y += self.bias
        return y.type_as(x)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Applies rotary positional embeddings in-place. Uses conjugate for inverse (de-rotation)."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


class ModelNew(torch.nn.Module):
    """Selects top-k compressed KV positions for sparse attention via learned scoring.
    Has its own Compressor (with Hadamard rotation) to build compressed KV for scoring."""

    def __init__(self, args: ModelArgs, freqs_cis: torch.Tensor, kv_cache: torch.Tensor, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio
        self.kv_cache = kv_cache
        self.freqs_cis = freqs_cis

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen

        # Queries: [B, S, H, D]
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        # Per-head weights [B, S, H]; promotion to fp32 in kernel wrapper
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)

        # KV slice: [B, T, D] where T = end_pos // ratio
        T = end_pos // ratio
        kv_slice = self.kv_cache[:bsz, :T]

        # Compute index_score via Triton kernel: [B, S, T] bf16
        index_score = _compute_index_score_triton(q, kv_slice, weights)

        # Mask for initial step
        if start_pos == 0:
            device = x.device
            mask = torch.arange(seqlen // ratio, device=device).repeat(seqlen, 1) >= \
                   (torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio)
            index_score += torch.where(mask, float("-inf"), 0.0)

        # Top-k indices
        topk_idxs = index_score.topk(min(self.index_topk, T), dim=-1)[1]
        if start_pos == 0:
            device = x.device
            mask = topk_idxs >= (torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio)
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs += offset
        return topk_idxs
    

args = ModelArgs(
        max_batch_size=2,
        max_seq_len=1024,
        dim=1024,
        index_n_heads=16,
        index_head_dim=64,
        index_topk=128,
        q_lora_rank=256,
        rope_head_dim=32
    )

def get_inputs():
    batch_size = 2
    seq_len = 64
    x = torch.randn(batch_size, seq_len, args.dim, dtype=torch.bfloat16).cuda()
    qr = torch.randn(batch_size, seq_len, args.q_lora_rank, dtype=torch.bfloat16).cuda()
    start_pos = 0
    offset = 0
    return [x, qr, start_pos, offset]


def get_init_inputs():
    compress_ratio=4
    max_seq_len = args.max_seq_len
    rope_theta = 10000.0
    # freqs_cis = torch.zeros(max_seq_len, args.rope_head_dim).cuda()
    freqs = 1.0 / (rope_theta ** (torch.arange(0, args.rope_head_dim, 2)[:args.rope_head_dim//2].float() / args.rope_head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float().cuda()
    freqs_cis = torch.polar(torch.ones_like(freqs).cuda(), freqs).view(max_seq_len, -1)
    kv_cache = torch.zeros(args.max_batch_size, args.max_seq_len // compress_ratio, args.index_head_dim, dtype=default_dtype).cuda()
    return [args, freqs_cis, kv_cache, compress_ratio]


if __name__ == "__main__":
    result = ModelNew(*get_init_inputs()).cuda().forward(*get_inputs())
    print(f"Forward pass successful! Output shape: {result.shape}, dtype: {result.dtype}")
    print(f"Sample output values: {result[0, :5]}")