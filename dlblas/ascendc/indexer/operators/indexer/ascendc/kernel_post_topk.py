# Kernel 4: postprocess_topk — ReLU + Weighted Sum + Causal Mask + TopK + Post-process
# Vector operations on Ascend910B2
#
# Architecture:
#   Steps 1-2: ReLU + weighted sum over H — PyTorch NPU ops (production path)
#   Step 3:    Causal mask — PyTorch NPU ops
#   Step 4:    TopK selection — PyTorch NPU op (no efficient Triton topk primitive)
#   Step 5:    Post-processing (mask invalid idxs, add offset) — PyTorch NPU ops
#
# A Triton fusion kernel (score_aggregate_kernel) is available for Steps 1-3
# via use_triton_aggregate=True. It provides fp32 accumulation precision but
# currently lower throughput than PyTorch NPU ops for large S. Future optimization
# of block sizes and memory access patterns may close this gap.
#
# NOTE: TopK remains a PyTorch op because Triton-Ascend does not provide an
# efficient topk / partial-sort primitive. A future Triton implementation could
# use a k-pass argmax approach for small k.

import torch
import triton
import triton.language as tl


# ==============================================================================
# Triton Fusion Kernel: ReLU + Weighted Sum + Causal Mask
# Functional alternative to PyTorch ops — see file-level notes.
# ==============================================================================

@triton.jit
def score_aggregate_kernel(
    scores_ptr,         # (B, H, S, kv_len) bf16
    weights_ptr,        # (B, S, H) bf16
    index_score_ptr,    # (B, S, kv_len) bf16 (output)
    S, H, kv_len,
    start_pos,
    compress_ratio,
    stride_sc_b, stride_sc_h, stride_sc_s, stride_sc_kv,
    stride_w_b, stride_w_s, stride_w_h,
    stride_is_b, stride_is_s, stride_is_kv,
    BLOCK_KV: tl.constexpr,
):
    """Fused ReLU + weighted sum over H + causal mask.

    Launch grid: (B, S) — 2D grid with inner kv_len loop.
    Each block handles one (b, s) pair, iterating over kv_len in BLOCK_KV chunks.

    For each (b, s):
      Iterate over kv_len in BLOCK_KV chunks:
        1. acc = sum_h( relu(scores[b,h,s, kv_chunk]) * weights[b,s,h] )
        2. if start_pos == 0:
             mask positions where kv_idx >= (s+1)//compress_ratio → -1e30
        3. store → index_score[b, s, kv_chunk]
    """
    b = tl.program_id(0)
    s = tl.program_id(1)

    for kv_start in range(0, kv_len, BLOCK_KV):
        offs_kv = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = offs_kv < kv_len

        # Accumulate ReLU(scores) * weights over H
        acc = tl.zeros((BLOCK_KV,), dtype=tl.float32)
        for h in range(H):
            sc_ptrs = (scores_ptr + b * stride_sc_b + h * stride_sc_h
                       + s * stride_sc_s + offs_kv * stride_sc_kv)
            sc = tl.load(sc_ptrs, mask=kv_mask, other=0.0)
            sc_relu = tl.maximum(sc, 0.0)

            w = tl.load(weights_ptr + b * stride_w_b + s * stride_w_s + h * stride_w_h)
            acc += sc_relu.to(tl.float32) * w.to(tl.float32)

        # Causal mask (prefill only)
        # floor((s+1) / ratio) — matching origin/indexer.py line 171
        if start_pos == 0:
            threshold = (s + 1) // compress_ratio
            is_masked = offs_kv >= threshold
            NEG_LARGE = -1e30
            acc = tl.where(is_masked & kv_mask, NEG_LARGE, acc.to(tl.float32))

        # Store (bf16, matching reference precision)
        is_ptrs = (index_score_ptr + b * stride_is_b + s * stride_is_s
                   + offs_kv * stride_is_kv)
        tl.store(is_ptrs, acc.to(tl.bfloat16), mask=kv_mask)


# ==============================================================================
# Host-side entry point
# ==============================================================================

def postprocess_topk(
    scores: torch.Tensor,
    weights: torch.Tensor,
    start_pos: int,
    offset: int,
    index_topk: int,
    compress_ratio: int,
    use_triton_aggregate: bool = False,
) -> torch.Tensor:
    """Kernel 4: Postprocessing and TopK selection.

    Operations:
      1. ReLU(scores) * weights (PyTorch NPU op, or Triton if use_triton_aggregate=True)
      2. Sum over head dimension → index_score
      3. Causal mask if start_pos == 0
      4. TopK selection (PyTorch NPU op)
      5. Post-processing: mask invalid indices, add offset (PyTorch)

    Args:
        scores: (B, H, S, kv_len) bf16 — from Kernel 3
        weights: (B*S, H) bf16 — from Kernel 2 (flattened batch+seq)
        start_pos: int — chunk start position
        offset: int — index offset to add
        index_topk: int — number of top indices to select
        compress_ratio: int — KV compression ratio
        use_triton_aggregate: if True, use score_aggregate_kernel for Steps 1-3

    Returns:
        topk_idxs: (B, S, K) int64 — selected KV position indices
    """
    B, H, S, kv_len = scores.shape
    device = scores.device

    if use_triton_aggregate:
        # --- Steps 1-3: ReLU + Weighted sum + Causal mask (Triton kernel) ---
        w_3d = weights.reshape(B, S, H).contiguous()
        index_score = torch.empty(B, S, kv_len, dtype=scores.dtype, device=device)
        grid = (B, S)

        score_aggregate_kernel[grid](
            scores, w_3d, index_score,
            S, H, kv_len,
            start_pos, compress_ratio,
            scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
            w_3d.stride(0), w_3d.stride(1), w_3d.stride(2),
            index_score.stride(0), index_score.stride(1), index_score.stride(2),
            BLOCK_KV=128,
        )
    else:
        # --- Steps 1-3: ReLU + Weighted sum + Causal mask (PyTorch) ---
        # Production path: uses optimized NPU ops.
        w_4d = weights.reshape(B, S, H).permute(0, 2, 1).unsqueeze(-1)  # (B, H, S, 1)
        scores_act = torch.relu(scores)  # (B, H, S, kv_len)
        index_score = (scores_act * w_4d).sum(dim=1)  # (B, S, kv_len) — bf16

        # Causal mask (prefill only)
        if start_pos == 0:
            row_idx = torch.arange(S, device=device).unsqueeze(1)  # (S, 1)
            col_idx = torch.arange(kv_len, device=device).unsqueeze(0)  # (1, kv_len)
            # threshold = floor((s+1) / ratio) — matching origin/indexer.py line 171
            threshold = (row_idx + 1) // compress_ratio  # (S, 1)
            causal_mask = col_idx >= threshold  # (S, kv_len)
            index_score = index_score + torch.where(causal_mask,
                                                    torch.tensor(float("-inf"), dtype=torch.float32, device=device),
                                                    torch.tensor(0.0, dtype=torch.float32, device=device))

    # --- Step 4: TopK selection (PyTorch) ---
    k = min(index_topk, kv_len)
    # Convert to fp32 for stable topk comparison
    topk_values, topk_idxs = torch.topk(index_score.float(), k=k, dim=-1)  # (B, S, k)

    # --- Step 5: Post-processing (PyTorch) ---
    if start_pos == 0:
        row_idx = torch.arange(S, device=device).unsqueeze(1)  # (S, 1)
        threshold = (row_idx + 1) // compress_ratio  # floor((s+1)/ratio)
        mask = topk_idxs >= threshold
        neg_one = torch.tensor(-1, dtype=torch.int64, device=device)
        topk_idxs = torch.where(mask, neg_one, topk_idxs.to(torch.int64) + offset)
    else:
        topk_idxs = topk_idxs.to(torch.int64) + offset

    return topk_idxs
