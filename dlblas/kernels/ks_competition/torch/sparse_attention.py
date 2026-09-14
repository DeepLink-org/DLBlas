# fmt: off
import torch
import torch.nn as nn


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


class Model(nn.Module):
    """
    Pure PyTorch implementation of sparse_attn from
    DeepSeek-V4-Pro/inference/kernel.py.

    Wraps sparse_attn_ref with a learnable attn_sink parameter.
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
        return sparse_attn_ref(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)


# ---------------------------------------------------------------------------
# Default config for get_inputs / get_init_inputs
# ---------------------------------------------------------------------------
batch_size = 8
seq_len    = 2600
n_kv       = 32
n_heads    = 64
head_dim   = 128
topk       = 16


def get_inputs():
    q         = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=torch.bfloat16)
    kv        = torch.randn(batch_size, n_kv,   head_dim,           dtype=torch.bfloat16)
    topk_idxs = torch.randint(0, n_kv, (batch_size, seq_len, topk), dtype=torch.int32)
    return [q, kv, topk_idxs]


def get_init_inputs():
    return [n_heads, head_dim]
# fmt: on
