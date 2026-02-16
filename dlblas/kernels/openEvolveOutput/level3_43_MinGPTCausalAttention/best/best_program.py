# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

@triton.jit
def _attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_b, stride_t, stride_d,
    T, scale,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    D_power2: tl.constexpr,   # Power-of-two dimension
):
    pid = tl.program_id(0)
    pid_batch = pid // T
    pid_row = pid % T

    # Column mask for padding
    col_mask = tl.arange(0, D_power2) < D
    # Load query vector with padding
    q_row_ptr = q_ptr + pid_batch * stride_b + pid_row * stride_t
    q = tl.load(q_row_ptr + tl.arange(0, D_power2) * stride_d, 
                mask=col_mask, other=0.0)

    # Initialize accumulators with proper scalar types
    acc = tl.zeros([D_power2], dtype=tl.float32)
    max_score = tl.full([], float('-inf'), dtype=tl.float32)  # Scalar tensor
    normalizer = tl.zeros([], dtype=tl.float32)  # Scalar tensor

    for j in range(0, pid_row + 1, BLOCK):
        j_offs = j + tl.arange(0, BLOCK)
        row_mask = j_offs < (pid_row + 1)
        
        # Load key block with double masking
        k_ptrs = k_ptr + pid_batch * stride_b + j_offs[:, None] * stride_t + tl.arange(0, D_power2)[None, :] * stride_d
        k_block = tl.load(k_ptrs, 
                         mask=row_mask[:, None] & col_mask[None, :], 
                         other=0.0)

        # Compute scores
        s = tl.sum(q[None, :] * k_block, axis=1) * scale
        s = tl.where(row_mask, s, float('-inf'))

        # Online softmax updates
        cur_max = tl.max(s, axis=0)
        new_max = tl.maximum(max_score, cur_max)
        exp_s = tl.exp(s - new_max)
        exp_s = tl.where(row_mask, exp_s, 0.0)
        alpha = tl.exp(max_score - new_max)

        # Load value block with double masking
        v_ptrs = v_ptr + pid_batch * stride_b + j_offs[:, None] * stride_t + tl.arange(0, D_power2)[None, :] * stride_d
        v_block = tl.load(v_ptrs, 
                         mask=row_mask[:, None] & col_mask[None, :], 
                         other=0.0)

        # Update accumulators
        v_update = tl.sum(exp_s[:, None] * v_block, axis=0)
        acc = acc * alpha + v_update
        normalizer = normalizer * alpha + tl.sum(exp_s, axis=0)
        max_score = new_max

    # Compute output and store
    output_row = acc / normalizer
    o_ptrs = o_ptr + pid_batch * stride_b + pid_row * stride_t + tl.arange(0, D_power2) * stride_d
    tl.store(o_ptrs, output_row, mask=col_mask)

class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = q.contiguous().view(B * self.n_head, T, -1)
        k = k.contiguous().view(B * self.n_head, T, -1)
        v = v.contiguous().view(B * self.n_head, T, -1)
        y = torch.empty_like(q)

        scale = 1.0 / math.sqrt(q.size(-1))
        D = q.size(-1)
        D_power2 = 2 ** math.ceil(math.log2(D))  # Next power of two
        grid = (B * self.n_head * T,)
        BLOCK = 64
        _attn_kernel[grid](
            q, k, v, y,
            q.stride(0), q.stride(1), q.stride(2),
            T, scale,
            D, BLOCK, D_power2,
        )

        y = y.view(B, self.n_head, T, -1).transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
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
# =================== EVOLVE-BLOCK-END ===================