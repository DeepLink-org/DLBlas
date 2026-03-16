import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

# From https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(NewGELU, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=4),
    ],
    key=["M", "N"],
)
@triton.jit
def relu_causal_attn_fwd(
    Q, K, V, Out, scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    B, H, M, N, KD,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_bh = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    b = pid_bh // H
    h = pid_bh % H

    m0 = pid_m * BLOCK_M
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_K)

    # Prepare masks for M dimension
    m_mask = offs_m < M

    # Pointer bases per (b, h)
    q_head_ptr = Q + b * stride_qb + h * stride_qh
    k_head_ptr = K + b * stride_kb + h * stride_kh
    v_head_ptr = V + b * stride_vb + h * stride_vh
    o_head_ptr = Out + b * stride_ob + h * stride_oh

    # Accumulator for output [BLOCK_M, BLOCK_K]
    acc_o = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    # For causal early-skip: last valid query index handled by this CTA
    max_valid_m = tl.minimum(m0 + (BLOCK_M - 1), M - 1)

    n0 = 0
    while n0 < N:
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        # Skip tiles strictly above causal diagonal: no contribution if keys start after the last query
        tile_has_effect = n0 <= max_valid_m

        if tile_has_effect:
            # Compute S = (Q @ K^T) for this (m-tile, n-tile)
            S = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            # Accumulate across head dimension KD by chunks of BLOCK_K
            k0 = 0
            while k0 < KD:
                offs_kk = k0 + offs_k

                # Load Q chunk: [BLOCK_M, BLOCK_K] -> pre-scale then cast to fp16 for tensor cores
                q_ptrs = q_head_ptr + (offs_m[:, None] * stride_qm) + (offs_kk[None, :] * stride_qk)
                q_mask = m_mask[:, None] & (offs_kk[None, :] < KD)
                q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)
                q_block = (q_block * scale).to(tl.float16)

                # Load K chunk: [BLOCK_N, BLOCK_K] (then transpose in dot)
                k_ptrs = k_head_ptr + (offs_n[:, None] * stride_kn) + (offs_kk[None, :] * stride_kk)
                k_mask2 = n_mask[:, None] & (offs_kk[None, :] < KD)
                k_block = tl.load(k_ptrs, mask=k_mask2, other=0.0, cache_modifier=".cg").to(tl.float16)

                # Tensor core GEMM for partial S
                S += tl.dot(q_block, tl.trans(k_block))

                k0 += BLOCK_K

            # Apply causal mask and bounds; then ReLU
            causal = (offs_n[None, :] <= offs_m[:, None])
            valid = m_mask[:, None] & n_mask[None, :] & causal

            # ReLU then mask (equivalent to masked_fill(-inf) then relu)
            S = tl.maximum(S, 0.0)
            S = tl.where(valid, S, 0.0)

            # Multiply by V and accumulate into output chunk over N
            v_ptrs = v_head_ptr + (offs_n[:, None] * stride_vn) + (offs_d[None, :] * stride_vk)
            v_mask2 = n_mask[:, None] & (offs_d[None, :] < KD)
            v_block = tl.load(v_ptrs, mask=v_mask2, other=0.0, cache_modifier=".cg").to(tl.float16)

            acc_o += tl.dot(S.to(tl.float16), v_block)

        n0 += BLOCK_N

    # Store result [BLOCK_M, BLOCK_K]
    o_ptrs = o_head_ptr + (offs_m[:, None] * stride_om) + (offs_d[None, :] * stride_ok)
    o_mask = m_mask[:, None] & (offs_d[None, :] < KD)
    tl.store(o_ptrs, acc_o, mask=o_mask)


class ModelNew(nn.Module):
    """
    A multi-head masked self-attention layer with a projection at the end that uses ReLU instead of Softmax.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        hs = C // self.n_head
        scale = 1.0 / math.sqrt(hs)

        use_triton = x.is_cuda and q.dtype == torch.float32 and k.dtype == torch.float32 and v.dtype == torch.float32
        if use_triton:
            # Allocate output (B, nh, T, hs)
            y = torch.empty_like(q)

            # Strides in elements
            sqb, sqh, sqm, sqk = q.stride()
            skb, skh, skn, skk = k.stride()
            svb, svh, svn, svk = v.stride()
            sob, soh, som, sok = y.stride()

            # Kernel launch parameters
            BLOCK_M = 64
            BLOCK_N = 128  # larger K/V tile to reduce loops and improve tensor core utilization
            BLOCK_K = 64
            grid = (B * self.n_head, triton.cdiv(T, BLOCK_M))

            relu_causal_attn_fwd[grid](
                q, k, v, y, scale,
                sqb, sqh, sqm, sqk,
                skb, skh, skn, skk,
                svb, svh, svn, svk,
                sob, soh, som, sok,
                B, self.n_head, T, T, hs,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )
        else:
            # Fallback to PyTorch reference
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.relu(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        return y


batch_size = 16
max_seqlen = 1024
n_embd = 768  # Hidden dimension, typical for BERT-base size
n_head = 12   # Number of attention heads, typical for BERT-base size

def get_inputs():
    return [torch.randn(batch_size, max_seqlen, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, max_seqlen]