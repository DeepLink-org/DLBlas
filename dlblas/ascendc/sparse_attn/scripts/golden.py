# ----------------------------------------------------------------------------------------------------------
# golden.py - PyTorch golden reference for sparse_attn
# ----------------------------------------------------------------------------------------------------------
#
# This file is shared by gen_data.py (direct-invoke path) and test_torch.py (PyTorch path).
# Both paths use the same reference implementation for consistency.

import torch
import torch.nn as nn


def sparse_attn_ref(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Reference sparse attention implementation.

    Args:
        q:         [b, m, h, d]  bf16
        kv:        [b, n, d]     bf16  (shared key-value, head-agnostic)
        attn_sink: [h]           fp32
        topk_idxs: [b, m, topk]  int32  (-1 = invalid)
        softmax_scale: float

    Returns:
        o: [b, m, h, d] bf16
    """
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]

    valid_mask = topk_idxs >= 0
    safe_idxs = topk_idxs.clamp(min=0).long()

    b_idx = torch.arange(b, device=q.device)[:, None, None].expand(b, m, topk)
    gathered_kv = kv[b_idx, safe_idxs]
    gathered_kv = gathered_kv.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

    scores = torch.einsum("bmhd,bmtd->bmht",
                          q.float(), gathered_kv.float()) * softmax_scale
    scores = scores.masked_fill(~valid_mask.unsqueeze(2), float("-inf"))

    sink = attn_sink.float().view(1, 1, h, 1)
    max_scores = torch.amax(scores, dim=-1, keepdim=True)
    max_scores = torch.maximum(max_scores, sink)

    exp_scores = torch.exp(scores - max_scores)
    exp_scores = exp_scores.masked_fill(~valid_mask.unsqueeze(2), 0.0)

    exp_sink = torch.exp(sink - max_scores)
    sum_exp = exp_scores.sum(dim=-1, keepdim=True) + exp_sink

    attn_weights = exp_scores / sum_exp

    output = torch.einsum("bmht,bmtd->bmhd",
                          attn_weights, gathered_kv.float())
    return output.to(q.dtype)


def compute_golden(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Compute golden output (numpy array or torch.Tensor interface).

    If inputs are numpy arrays, convert to torch, compute, convert back.
    """
    if isinstance(q, torch.Tensor):
        return sparse_attn_ref(q, kv, attn_sink, topk_idxs, softmax_scale)

    import numpy as np
    q_t = torch.from_numpy(q)
    kv_t = torch.from_numpy(kv)
    sink_t = torch.from_numpy(attn_sink)
    idxs_t = torch.from_numpy(topk_idxs)
    result = sparse_attn_ref(q_t, kv_t, sink_t, idxs_t, softmax_scale)
    return result.numpy()
