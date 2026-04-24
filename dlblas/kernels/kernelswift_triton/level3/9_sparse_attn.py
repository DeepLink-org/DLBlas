import torch
import torch.nn as nn
import triton
import triton.language as tl


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


@triton.jit
def sparse_attn_kernel(
    q_ptr,            # bf16  [B, M, H, D]
    kv_ptr,           # bf16  [B, N, D]
    topk_ptr,         # int32 [B, M, TOPK]
    sink_ptr,         # f32   [H]
    o_ptr,            # bf16  [B, M, H, D]
    B: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    softmax_scale: tl.float32,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    h = pid % H
    tmp = pid // H
    m = tmp % M
    b = tmp // M

    offs_d = tl.arange(0, BLOCK_D)
    offs_t = tl.arange(0, BLOCK_T)
    tl.multiple_of(offs_d, 8)

    # Base offsets
    q_base = ((b * M + m) * H + h) * D
    o_base = q_base
    topk_base = (b * M + m) * TOPK

    # Load per-head sink
    sink_val = tl.load(sink_ptr + h).to(tl.float32)

    # Load topk indices for (b, m)
    t_in_range = offs_t < TOPK
    idxs = tl.load(topk_ptr + topk_base + offs_t, mask=t_in_range, other=0).to(tl.int32)
    valid_t = t_in_range & (idxs >= 0)
    # Clamp invalid idxs to 0 to avoid OOB, also clamp to N-1 just in case
    safe_idx = tl.where(valid_t, idxs, 0)
    safe_idx = tl.minimum(safe_idx, N - 1)

    # Early-exit: if no valid indices, write zeros and return
    valid_count = tl.sum(valid_t.to(tl.int32), axis=0)
    if valid_count == 0:
        for d0 in tl.static_range(0, D, BLOCK_D):
            d_idx = d0 + offs_d
            mask_d = d_idx < D
            tl.store(o_ptr + o_base + d_idx, tl.zeros([BLOCK_D], dtype=tl.bfloat16), mask=mask_d)
        return

    # Precompute KV row offsets for these indices (row-major contiguous)
    kv_row_offsets = ((b * N) + safe_idx) * D  # [BLOCK_T]

    # Fast path: entire head-dim fits in one tile
    if D <= BLOCK_D:
        d_idx = offs_d
        mask_d = d_idx < D

        # Load q[b,m,h,:]
        q_vec = tl.load(q_ptr + q_base + d_idx, mask=mask_d, other=0.0).to(tl.float32)

        # Gather kv tile [BLOCK_T, BLOCK_D]
        kv_ptrs = kv_ptr + kv_row_offsets[:, None] + d_idx[None, :]
        kv_mask = valid_t[:, None] & mask_d[None, :]
        kv_tile = tl.load(kv_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        # Compute scores and scaled softmax with sink in denominator only
        scores = tl.sum(kv_tile * q_vec[None, :], axis=1) * softmax_scale
        neg_inf = -float("inf")
        scores = tl.where(valid_t, scores, neg_inf)

        max_scores = tl.max(scores, axis=0)
        max_scores = tl.maximum(max_scores, sink_val)

        exp_scores = tl.exp(scores - max_scores)
        sum_exp_scores = tl.sum(exp_scores, axis=0)
        exp_sink = tl.exp(sink_val - max_scores)
        denom = sum_exp_scores + exp_sink

        attn_w = exp_scores / denom  # [BLOCK_T]

        # Weighted sum to produce output vector over D
        o_vec = tl.sum(kv_tile * attn_w[:, None], axis=0)  # [BLOCK_D] in f32
        tl.store(o_ptr + o_base + d_idx, o_vec.to(tl.bfloat16), mask=mask_d)
        return

    # General path: two-phase over D in tiles

    # Phase 1: compute scores for all TOPK entries across D in tiles
    scores = tl.zeros([BLOCK_T], dtype=tl.float32)
    for d0 in tl.static_range(0, D, BLOCK_D):
        d_idx = d0 + offs_d
        mask_d = d_idx < D

        # Load q[b,m,h,d0:d0+BLOCK_D]
        q_vec = tl.load(q_ptr + q_base + d_idx, mask=mask_d, other=0.0).to(tl.float32)

        # Gather kv rows tile [BLOCK_T, BLOCK_D]
        kv_ptrs = kv_ptr + kv_row_offsets[:, None] + d_idx[None, :]
        kv_mask = valid_t[:, None] & mask_d[None, :]
        kv_tile = tl.load(kv_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        # Accumulate partial dot-products
        scores += tl.sum(kv_tile * q_vec[None, :], axis=1)

    # Scale and mask invalid to -inf to exclude from softmax
    scores = scores * softmax_scale
    neg_inf = -float("inf")
    scores = tl.where(valid_t, scores, neg_inf)

    # Numerically stable softmax with sink only in denominator
    max_scores = tl.max(scores, axis=0)
    max_scores = tl.maximum(max_scores, sink_val)

    exp_scores = tl.exp(scores - max_scores)
    sum_exp_scores = tl.sum(exp_scores, axis=0)
    exp_sink = tl.exp(sink_val - max_scores)
    denom = sum_exp_scores + exp_sink
    attn_w = exp_scores / denom  # [BLOCK_T]

    # Phase 2: weighted sum to produce output vector over D
    for d0 in tl.static_range(0, D, BLOCK_D):
        d_idx = d0 + offs_d
        mask_d = d_idx < D

        kv_ptrs = kv_ptr + kv_row_offsets[:, None] + d_idx[None, :]
        kv_mask = valid_t[:, None] & mask_d[None, :]
        kv_tile = tl.load(kv_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        o_vec = tl.sum(kv_tile * attn_w[:, None], axis=0)  # [BLOCK_D] in f32
        tl.store(o_ptr + o_base + d_idx, o_vec.to(tl.bfloat16), mask=mask_d)


def _next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def sparse_attn_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """
    Triton implementation matching sparse_attn_ref.
    """
    assert q.is_cuda and kv.is_cuda and topk_idxs.is_cuda and attn_sink.is_cuda, "Inputs must be CUDA tensors"
    # Ensure contiguous tensors for pointer arithmetic
    q_c = q.contiguous()
    kv_c = kv.contiguous()
    topk_c = topk_idxs.contiguous()
    sink_c = attn_sink.contiguous()

    B, M, H, D = q_c.shape
    _, N, D_kv = kv_c.shape
    assert D == D_kv, "Head dim mismatch"
    TOPK = topk_c.shape[-1]

    o = torch.empty_like(q_c)

    # Meta-parameters
    BLOCK_D = min(128, _next_power_of_2(D))
    BLOCK_T = _next_power_of_2(TOPK)
    # Heuristic: fewer warps for small tiles to reduce overhead, more for larger D
    num_warps = 2 if (BLOCK_D <= 64 and BLOCK_T <= 16) else 4

    grid = (B * M * H,)

    sparse_attn_kernel[grid](
        q_c, kv_c, topk_c, sink_c, o,
        B=B, M=M, H=H, D=D, N=N, TOPK=TOPK,
        softmax_scale=float(softmax_scale),
        BLOCK_D=BLOCK_D, BLOCK_T=BLOCK_T,
        num_warps=num_warps, num_stages=2,
    )
    return o


class ModelNew(nn.Module):
    """
    Triton-accelerated implementation of sparse attention with an attention sink.
    Falls back to the reference PyTorch implementation on CPU.
    """

    def __init__(self, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.softmax_scale = head_dim ** -0.5
        # Learnable per-head sink bias (only affects softmax denominator)
        self.attn_sink = nn.Parameter(torch.zeros(n_heads, dtype=torch.float32))

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_idxs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q:         [b, m, h, d]  bfloat16
            kv:        [b, n, d]     bfloat16
            topk_idxs: [b, m, topk]  int32, -1 for invalid positions

        Returns:
            o: [b, m, h, d] bfloat16
        """
        # Use Triton kernel on CUDA; otherwise fall back to reference implementation
        if q.is_cuda and kv.is_cuda and topk_idxs.is_cuda and self.attn_sink.is_cuda:
            return sparse_attn_triton(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            return sparse_attn_ref(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)


# ---------------------------------------------------------------------------
# Default config for get_inputs / get_init_inputs
# ---------------------------------------------------------------------------
batch_size = 2
seq_len    = 16
n_kv       = 32
n_heads    = 8
head_dim   = 64
topk       = 16


def get_inputs():
    q         = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    kv        = torch.randn(batch_size, n_kv,   head_dim,           dtype=torch.bfloat16, device="cuda")
    # allow -1 to appear to exercise masking path
    topk_idxs = torch.randint(-1, n_kv, (batch_size, seq_len, topk), dtype=torch.int32, device="cuda")
    return [q, kv, topk_idxs]


def get_init_inputs():
    return [n_heads, head_dim]