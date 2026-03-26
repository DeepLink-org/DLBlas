import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _conti_input(data, seq_lens):
    data = [x[:l] for x, l in zip(data, seq_lens)]
    data = torch.cat(data, dim=0)
    return data


@triton.jit
def _attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, bias_ptr, o_ptr,
    B, Hq, Hk, S, S_full, D, Dv,
    stride_q_b, stride_q_s, stride_q_h, stride_q_d,
    stride_k_b, stride_k_s, stride_k_h, stride_k_d,
    stride_v_b, stride_v_s, stride_v_h, stride_v_d,
    stride_bias_b, stride_bias_s, stride_bias_n,
    stride_o_b, stride_o_s, stride_o_h, stride_o_d,
    group, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_DV: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    b = pid_bh // Hq
    hq = pid_bh % Hq
    hk = hq // group

    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, BLOCK_D)
    dv_offsets = tl.arange(0, BLOCK_DV)

    # Load Q tile [BLOCK_M, BLOCK_D]
    q_ptrs = q_ptr + b * stride_q_b + m_offsets[:, None] * stride_q_s + hq * stride_q_h + d_offsets[None, :] * stride_q_d
    q_mask = (m_offsets[:, None] < S) & (d_offsets[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
    q = q * scale

    # Initialize softmax stats and accumulator
    NEG_INF = -float("inf")
    m_i = tl.full([BLOCK_M], NEG_INF, tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    o_acc = tl.zeros([BLOCK_M, BLOCK_DV], tl.float32)

    # Iterate over K/V sequence dimension in BLOCK_N steps
    for start_n in range(0, S_full, BLOCK_N):
        n_offsets = start_n + tl.arange(0, BLOCK_N)

        # Load K tile as [BLOCK_D, BLOCK_N] for efficient dot
        k_ptrs = k_ptr + b * stride_k_b + hk * stride_k_h + d_offsets[:, None] * stride_k_d + n_offsets[None, :] * stride_k_s
        k_mask = (d_offsets[:, None] < D) & (n_offsets[None, :] < S_full)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Compute logits [BLOCK_M, BLOCK_N] = q @ k
        logits = tl.dot(q, k)

        # Load bias tile [BLOCK_M, BLOCK_N] and add
        bias_ptrs = bias_ptr + b * stride_bias_b + m_offsets[:, None] * stride_bias_s + n_offsets[None, :] * stride_bias_n
        valid_mn = (m_offsets[:, None] < S) & (n_offsets[None, :] < S_full)
        bias_tile = tl.load(bias_ptrs, mask=valid_mn, other=0.0).to(tl.float32)
        logits = logits + bias_tile

        # Mask invalid positions with -inf to avoid NaNs
        logits = tl.where(valid_mn, logits, NEG_INF)

        # Numerically stable softmax update
        max_curr = tl.max(logits, 1)
        m_new = tl.maximum(m_i, max_curr)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(logits - m_new[:, None])

        # Load V tile [BLOCK_N, BLOCK_DV]
        v_ptrs = v_ptr + b * stride_v_b + hk * stride_v_h + n_offsets[:, None] * stride_v_s + dv_offsets[None, :] * stride_v_d
        v_mask = (n_offsets[:, None] < S_full) & (dv_offsets[None, :] < Dv)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)

        # Update accumulators
        o_acc = o_acc * alpha[:, None] + tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    # Normalize and store
    o = o_acc / l_i[:, None]
    o_ptrs = o_ptr + b * stride_o_b + m_offsets[:, None] * stride_o_s + hq * stride_o_h + dv_offsets[None, :] * stride_o_d
    out_mask = (m_offsets[:, None] < S) & (dv_offsets[None, :] < Dv)
    tl.store(o_ptrs, o.to(tl.float16), mask=out_mask)


def _triton_attention(batched_q, batched_kv, bias):
    batched_k, batched_v = batched_kv
    B, S, Hq, D = batched_q.shape
    _, S_full, Hk, Dk = batched_k.shape
    Dv = batched_v.shape[-1]
    assert D == Dk, "Q and K head dimensions must match"
    group = Hq // Hk

    # Reorder K/V for coalesced memory access
    # K: [B, Hk, D, S_full], V: [B, Hk, S_full, Dv]
    k_perm = batched_k.permute(0, 2, 3, 1)
    v_perm = batched_v.transpose(1, 2)

    # Allocate output [B, S, Hq, Dv]
    o = torch.empty((B, S, Hq, Dv), dtype=batched_q.dtype, device=batched_q.device)

    # Tuned block sizes with head-dim padding to multiples of 16 (<=64)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = min(64, ((D + 15) // 16) * 16)
    BLOCK_DV = min(64, ((Dv + 15) // 16) * 16)

    # Fallback to reference implementation if head dims exceed block capacity
    if D > BLOCK_D or Dv > BLOCK_DV:
        num_heads_q = batched_q.shape[2]
        num_heads_k = batched_k.shape[2]
        head_dim = batched_q.shape[-1]
        group_f = num_heads_q // num_heads_k

        q = batched_q.transpose(1, 2)
        k = batched_k.permute(0, 2, 3, 1)
        v = batched_v.transpose(1, 2)
        k = k.unsqueeze(2).expand(-1, -1, group_f, -1, -1).flatten(1, 2)
        v = v.unsqueeze(2).expand(-1, -1, group_f, -1, -1).flatten(1, 2)
        qk = torch.matmul(q, k) / math.sqrt(head_dim)
        attn_weight = qk + bias[:, None]
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
        attn_weight = attn_weight.to(q.dtype)
        attn_output = torch.matmul(attn_weight, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output

    grid = (B * Hq, triton.cdiv(S, BLOCK_M))
    scale = 1.0 / math.sqrt(D)

    _attn_fwd_kernel[grid](
        batched_q, k_perm, v_perm, bias, o,
        B, Hq, Hk, S, S_full, D, Dv,
        batched_q.stride(0), batched_q.stride(1), batched_q.stride(2), batched_q.stride(3),
        k_perm.stride(0), k_perm.stride(3), k_perm.stride(1), k_perm.stride(2),
        v_perm.stride(0), v_perm.stride(2), v_perm.stride(1), v_perm.stride(3),
        bias.stride(0), bias.stride(1), bias.stride(2),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        group, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_DV=BLOCK_DV,
        num_warps=4, num_stages=2,
    )
    return o


def _naive_attention(batched_q, batched_kv, bias):
    batched_k, batched_v = batched_kv

    num_heads_q = batched_q.shape[2]
    num_heads_k = batched_k.shape[2]
    head_dim = batched_q.shape[-1]
    group = num_heads_q // num_heads_k

    q = batched_q.transpose(1, 2)
    k = batched_k.permute(0, 2, 3, 1)
    v = batched_v.transpose(1, 2)

    # expand group
    k = k.unsqueeze(2).expand(-1, -1, group, -1, -1).flatten(1, 2)
    v = v.unsqueeze(2).expand(-1, -1, group, -1, -1).flatten(1, 2)

    qk = torch.matmul(q, k) / math.sqrt(head_dim)
    attn_weight = qk + bias[:, None]
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    attn_weight = attn_weight.to(q.dtype)
    attn_output = torch.matmul(attn_weight, v)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output


class ModelNew(nn.Module):
    def __init__(self, dtype=torch.float16, device='cuda'):
        super().__init__()
        self.dtype = dtype
        self.device = device

    def forward(self, batched_q, batched_kv, seq_lens, history_lens):
        mask = self._mask(seq_lens, history_lens)
        # Triton-accelerated attention with fallback as needed
        attn_out = _triton_attention(batched_q, batched_kv, mask)
        return _conti_input(attn_out, seq_lens)

    def _make_bias(self, seq_lens, history_lens, neg_val):
        full_seq_lens = seq_lens + history_lens
        max_seq_len = seq_lens.max().item()
        max_full_len = full_seq_lens.max().item()
        seq_ranges = [torch.arange(max_seq_len) for _ in seq_lens]
        for r, l in zip(seq_ranges, seq_lens):
            r[l:] = -max_full_len
        seq_ranges = torch.stack(seq_ranges, dim=0).to(self.device)
        kv_ranges = [torch.arange(max_full_len) for _ in full_seq_lens]
        kv_ranges = torch.stack(kv_ranges, 0).to(self.device)
        mask = kv_ranges[:, None, :] - seq_ranges[:, :, None] > history_lens[:, None, None]
        return mask.float() * neg_val

    def _mask(self, seq_lens, history_lens):
        neg_val = -1e30
        return self._make_bias(seq_lens, history_lens, neg_val)


device = 'cuda'
dtype = torch.float16
feat_dim = 16
feat_dim_v = 16
num_heads_q = 4
num_heads_k = 2
seq_lens = torch.tensor([128], device=device)
history_lens = torch.tensor([128], device=device)


def _batched_q(seq_lens, num_heads_q, feat_dim, dtype):
    torch.manual_seed(123)
    batch_size = len(seq_lens)
    max_seq_len = seq_lens.max().item()
    return torch.randn(batch_size, max_seq_len, num_heads_q, feat_dim, dtype=dtype, device=device)


def _batched_kv(seq_lens, history_lens, num_heads_k, feat_dim, feat_dim_v, dtype):
    torch.manual_seed(123)
    batch_size = len(seq_lens)
    full_seq_lens = seq_lens + history_lens
    max_seq_len = full_seq_lens.max().item()
    k = torch.rand(batch_size, max_seq_len, num_heads_k, feat_dim, dtype=dtype, device=device)
    v = torch.rand(batch_size, max_seq_len, num_heads_k, feat_dim_v, dtype=dtype, device=device)
    return k, v


def get_inputs():
    batched_q = _batched_q(seq_lens, num_heads_q, feat_dim, dtype)
    batched_kv = _batched_kv(seq_lens, history_lens, num_heads_k, feat_dim, feat_dim_v, dtype)
    return (batched_q, batched_kv, seq_lens, history_lens)


def get_init_inputs():
    return (dtype, device)