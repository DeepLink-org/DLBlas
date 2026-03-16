import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Baseline configs
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_DMODEL": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_DMODEL": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_DMODEL": 128}, num_warps=2, num_stages=2),
        # Larger tiles to reduce loop iters and improve bandwidth utilization on H200
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_DMODEL": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_DMODEL": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_DMODEL": 128}, num_warps=8, num_stages=3),
        # Alternative pipeline depths
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_DMODEL": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_DMODEL": 128}, num_warps=8, num_stages=2),
    ],
    key=["T"],
)
@triton.jit
def _flash_attn_fwd(
    q_ptr, k_ptr, v_ptr, o_ptr,
    Z, T, D,
    stride_qz, stride_qt, stride_qd,
    stride_kz, stride_kt, stride_kd,
    stride_vz, stride_vt, stride_vd,
    stride_oz, stride_ot, stride_od,
    scale,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    pid_z = tl.program_id(0)
    pid_m = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    tl.multiple_of(offs_d, 16)
    tl.max_contiguous(offs_d, BLOCK_DMODEL)

    # Masks
    m_mask = offs_m < T
    d_mask = offs_d < D

    # Pointers to this (z) batch-head slice
    q_block_ptr = q_ptr + pid_z * stride_qz + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    o_block_ptr = o_ptr + pid_z * stride_oz + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od

    # Load Q block once
    q = tl.load(q_block_ptr, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    # Pre-scale Q to avoid per-iteration multiply
    q = q * scale

    # Initialize running stats
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    # For out-of-bounds rows, avoid NaNs by setting (m_i=0, l_i=1)
    m_i = tl.where(m_mask, m_i, 0.0)
    l_i = tl.where(m_mask, l_i, 1.0)

    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    start_n = 0
    if CAUSAL:
        # Only iterate over necessary key blocks for causal attention: j <= i
        while (start_n < T) and (start_n < m_start + BLOCK_M):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < T

            k_block_ptr = k_ptr + pid_z * stride_kz + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
            v_block_ptr = v_ptr + pid_z * stride_vz + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd

            # Stream K/V via L2 to reduce L1 thrash
            k = tl.load(k_block_ptr, mask=n_mask[:, None] & d_mask[None, :], other=0.0, cache_modifier=".cg")
            v = tl.load(v_block_ptr, mask=n_mask[:, None] & d_mask[None, :], other=0.0, cache_modifier=".cg")

            # Scores
            scores = tl.dot(q, tl.trans(k))  # [BM, BN]

            # Causal + bounds mask
            causal = offs_n[None, :] <= offs_m[:, None]
            valid = m_mask[:, None] & n_mask[None, :] & causal
            scores = tl.where(valid, scores, -float("inf"))

            # Update running max/sum using the online softmax merge
            m_ij = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)

            p = tl.exp(scores - m_new[:, None])
            l_new = l_i * alpha + tl.sum(p, axis=1)

            # Accumulate numerator for output: sum_j exp(s_ij - m_new) * V_j
            acc = acc * alpha[:, None] + tl.dot(p, v)

            m_i = m_new
            l_i = l_new

            start_n += BLOCK_N
    else:
        while start_n < T:
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < T

            k_block_ptr = k_ptr + pid_z * stride_kz + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
            v_block_ptr = v_ptr + pid_z * stride_vz + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd

            k = tl.load(k_block_ptr, mask=n_mask[:, None] & d_mask[None, :], other=0.0, cache_modifier=".cg")
            v = tl.load(v_block_ptr, mask=n_mask[:, None] & d_mask[None, :], other=0.0, cache_modifier=".cg")

            scores = tl.dot(q, tl.trans(k))  # [BM, BN]
            valid = m_mask[:, None] & n_mask[None, :]
            scores = tl.where(valid, scores, -float("inf"))

            m_ij = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)

            p = tl.exp(scores - m_new[:, None])
            l_new = l_i * alpha + tl.sum(p, axis=1)

            acc = acc * alpha[:, None] + tl.dot(p, v)

            m_i = m_new
            l_i = l_new

            start_n += BLOCK_N

    # Normalize
    inv_l = 1.0 / l_i
    o = acc * inv_l[:, None]

    # Store
    tl.store(o_block_ptr, o, mask=m_mask[:, None] & d_mask[None, :])


class ModelNew(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head

        # If dropout on attention is active (p>0 and training), fallback to reference PyTorch path
        # to preserve exact dropout semantics/RNG behavior.
        if self.training and getattr(self.attn_dropout, "p", 0.0) > 0.0:
            k_ = k.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
            q_ = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
            v_ = v.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)

            att = (q_ @ k_.transpose(-2, -1)) * (1.0 / math.sqrt(k_.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v_  # (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
            y = self.resid_dropout(self.c_proj(y))
            return y

        # Triton-accelerated path (no attention dropout)
        # Reshape to [B, nh, T, hs] then merge B*nh into one dimension for kernel
        q = q.view(B, T, self.n_head, hs).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, T, self.n_head, hs).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, T, self.n_head, hs).permute(0, 2, 1, 3).contiguous()

        # Compute in float32 for numerical stability, matching the reference PyTorch behavior on FP32 inputs
        q32 = q.to(torch.float32)
        k32 = k.to(torch.float32)
        v32 = v.to(torch.float32)

        Z = B * self.n_head
        D = hs

        # Flatten (B, H, T, D) -> (Z, T, D) where Z = B*H
        def merge_bh(t):
            return t.view(Z, T, D).contiguous()

        qz = merge_bh(q32)
        kz = merge_bh(k32)
        vz = merge_bh(v32)
        oz = torch.empty_like(qz)

        stride_qz, stride_qt, stride_qd = qz.stride()
        stride_kz, stride_kt, stride_kd = kz.stride()
        stride_vz, stride_vt, stride_vd = vz.stride()
        stride_oz, stride_ot, stride_od = oz.stride()

        scale = 1.0 / math.sqrt(D)

        grid = lambda META: (Z, triton.cdiv(T, META["BLOCK_M"]))
        _flash_attn_fwd[grid](
            qz, kz, vz, oz,
            Z, T, D,
            stride_qz, stride_qt, stride_qd,
            stride_kz, stride_kt, stride_kd,
            stride_vz, stride_vt, stride_vd,
            stride_oz, stride_ot, stride_od,
            scale,
            CAUSAL=True,
        )

        y_heads = oz.view(B, self.n_head, T, D).permute(0, 2, 1, 3).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y_heads))
        return y


batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [torch.randn(batch_size, seq_len, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]