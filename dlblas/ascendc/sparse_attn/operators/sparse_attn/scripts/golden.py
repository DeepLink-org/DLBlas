# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# SparseAttn Golden 计算（参考实现）
# ============================================================================

import numpy as np


def compute_golden(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Compute sparse attention reference output.

    Pure numpy implementation matching the PyTorch reference:
    sparse_attn_ref() in /origin/sparse_attn.py

    Args:
        q:           [b, m, h, d]  float32
        kv:          [b, n, d]     float32
        attn_sink:   [h]           float32
        topk_idxs:   [b, m, topk]  int32  (-1 = invalid)
        softmax_scale: float

    Returns:
        o: [b, m, h, d] float32
    """
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]

    valid_mask = topk_idxs >= 0  # [b, m, topk]
    safe_idxs = np.maximum(topk_idxs, 0)  # clamp to 0

    # Gather KV: [b, m, topk, d]
    # kv is [b, n, d]
    b_idx = np.arange(b)[:, None, None]  # [b, 1, 1]
    b_idx = np.broadcast_to(b_idx, (b, m, topk))  # [b, m, topk]
    gathered_kv = kv[b_idx, safe_idxs]  # [b, m, topk, d]

    # Zero out invalid positions
    gathered_kv = gathered_kv * valid_mask[..., None]

    # Attention scores: [b, m, h, topk]
    # einsum("bmhd,bmtd->bmht", q, gathered_kv) * softmax_scale
    scores = np.einsum("bmhd,bmtd->bmht", q.astype(np.float32),
                       gathered_kv.astype(np.float32)) * softmax_scale

    # Mask invalid to -inf
    scores = np.where(valid_mask[:, :, None, :], scores, -np.inf)

    # Softmax with sink
    sink = attn_sink.reshape(1, 1, h, 1)  # [1, 1, h, 1]

    max_scores = np.max(scores, axis=-1, keepdims=True)  # [b, m, h, 1]
    max_scores = np.maximum(max_scores, sink)

    exp_scores = np.exp(scores - max_scores)
    exp_scores = np.where(valid_mask[:, :, None, :], exp_scores, 0.0)

    exp_sink = np.exp(sink - max_scores)  # [b, m, h, 1]
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True) + exp_sink

    attn_weights = exp_scores / np.maximum(sum_exp, 1e-10)

    # Weighted sum
    output = np.einsum("bmht,bmtd->bmhd", attn_weights,
                       gathered_kv.astype(np.float32))

    return output
