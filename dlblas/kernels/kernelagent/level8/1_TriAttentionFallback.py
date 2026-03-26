"""
TriAttention optimized with Triton FlashAttention-style kernel
Functionally equivalent to the original PyTorch implementation.

From: protenix/openfold_local/model/primitives.py:_tri_attention()
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qg, stride_qm, stride_qd,
    stride_kg, stride_km, stride_kd,
    stride_vg, stride_vm, stride_vd,
    stride_og, stride_om, stride_od,
    L, D, scale,
    BLOCK_M: tl.constexpr,  # rows of queries
    BLOCK_N: tl.constexpr,  # columns of keys/values
    BLOCK_D: tl.constexpr,  # feature dim (D) tile
):
    # Program IDs:
    #   pid_g indexes the [B*N*H] groups
    #   pid_m indexes blocks along the sequence length (queries)
    pid_g = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Hints to help codegen/vectorization
    tl.multiple_of(offs_d, 8)
    tl.multiple_of(offs_m, 16)

    row_mask = offs_m < L
    d_mask = offs_d < D

    # Load Q tile [BLOCK_M, BLOCK_D], compute in fp32 and pre-scale
    q_ptrs = Q_ptr + pid_g * stride_qg + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=row_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    q = q * scale

    # Online softmax state
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    # Iterate over K/V blocks across sequence length
    for start_n in range(0, L, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < L

        k_ptrs = K_ptr + pid_g * stride_kg + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd
        v_ptrs = V_ptr + pid_g * stride_vg + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd

        # Stream K/V via L2 to reduce L1 pollution
        k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0, cache_modifier=".cg").to(tl.float32)
        v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0, cache_modifier=".cg").to(tl.float32)

        # Scores [BLOCK_M, BLOCK_N] = (Q @ K^T), Q already scaled
        scores = tl.dot(q, tl.trans(k))
        # Mask invalid columns to -inf so they don't contribute to softmax
        scores = tl.where(n_mask[None, :], scores, -float("inf"))

        # Online softmax update
        max_scores = tl.max(scores, axis=1)                      # [BLOCK_M]
        new_m_i = tl.maximum(m_i, max_scores)                    # [BLOCK_M]
        p = tl.exp(scores - new_m_i[:, None])                    # [BLOCK_M, BLOCK_N]
        alpha = tl.exp(m_i - new_m_i)                            # [BLOCK_M]

        l_i = l_i * alpha + tl.sum(p, axis=1)                    # update denominators
        acc = acc * alpha[:, None] + tl.dot(p, v)                # update numerators
        m_i = new_m_i

    # Normalize
    out = acc / l_i[:, None]
    # Store with bounds mask
    o_ptrs = O_ptr + pid_g * stride_og + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out, mask=row_mask[:, None] & d_mask[None, :])


def tri_attention_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias1: torch.Tensor = None,
    bias2: torch.Tensor = None,
) -> torch.Tensor:
    """
    Args:
        q,k,v: [B, N, S, H, D]
        bias1: [B, N, S, 1, S] 或 None
        bias2: [B, N, 1, H, S, S] 或 None（这里简化为可 broadcast 的形式）
    Returns:
        out: [B, N, S, H, D]
    """
    B, N, S, H, D = q.shape
    # reshape to [B*N*H, S, D]
    q2 = q.permute(0, 1, 3, 2, 4).reshape(B * N * H, S, D).contiguous()
    k2 = k.permute(0, 1, 3, 2, 4).reshape(B * N * H, S, D).contiguous()
    v2 = v.permute(0, 1, 3, 2, 4).reshape(B * N * H, S, D).contiguous()

    # If any bias provided, fall back to PyTorch to preserve semantics
    if bias1 is not None or bias2 is not None:
        attn_bias = 0.0
        if bias1 is not None:
            # bias1: [B,N,S,1,S] -> [B,N,H,S,S]
            b1 = bias1.expand(B, N, S, H, S).permute(0, 1, 3, 2, 4)  # [B,N,H,S,S]
            b1 = b1.reshape(B * N * H, S, S)
            attn_bias = attn_bias + b1
        if bias2 is not None:
            # allow bias2 to be [B,N,H,S,S] or broadcastable
            if bias2.dim() == 5:
                b2 = bias2
            else:
                # [B,N,1,H,S,S] -> [B,N,H,S,S]
                b2 = bias2.squeeze(2)
            b2 = b2.reshape(B * N * H, S, S)
            attn_bias = attn_bias + b2

        out = F.scaled_dot_product_attention(q2, k2, v2, attn_mask=attn_bias, dropout_p=0.0, is_causal=False)
        out = out.reshape(B, N, H, S, D).permute(0, 1, 3, 2, 4)
        return out

    # Fast path: Triton kernel (no bias)
    G = B * N * H
    out2 = torch.empty_like(q2)

    # Tile sizes tuned for H100/H200; dynamic BLOCK_D to avoid extra work for small D
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 32 if D <= 32 else 64
    scale = 1.0 / math.sqrt(float(D))

    grid = (G, triton.cdiv(S, BLOCK_M))
    attention_fwd_kernel[grid](
        q2, k2, v2, out2,
        q2.stride(0), q2.stride(1), q2.stride(2),
        k2.stride(0), k2.stride(1), k2.stride(2),
        v2.stride(0), v2.stride(1), v2.stride(2),
        out2.stride(0), out2.stride(1), out2.stride(2),
        S, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=3
    )

    out = out2.reshape(B, N, H, S, D).permute(0, 1, 3, 2, 4)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return tri_attention_fallback(q, k, v, None, None)


# ==========================================
# Hyperparameters & Data Generation
# ==========================================

B = 1
N = 2
S = 128
H = 4
D = 32


def get_inputs():
    device = 'cuda'
    torch.manual_seed(42)

    q = torch.randn(B, N, S, H, D, device=device)
    k = torch.randn(B, N, S, H, D, device=device)
    v = torch.randn(B, N, S, H, D, device=device)

    return [q, k, v]


def get_init_inputs():
    return []