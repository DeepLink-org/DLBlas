# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

@triton.jit
def attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    scale, seq_len,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + (offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd)
    k_ptrs = k_ptr + pid_b * stride_kb + pid_h * stride_kh + (offs_n[None, :] * stride_kt + offs_d[:, None] * stride_kd)
    v_ptrs = v_ptr + pid_b * stride_vb + pid_h * stride_vh + (offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd)
    
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for block_n in range(0, seq_len, BLOCK_N):
        offs_n_curr = block_n + offs_n
        
        mask_n = offs_n_curr < seq_len
        mask_mn = (offs_m[:, None] >= offs_n_curr[None, :]) & (offs_m[:, None] < seq_len)
        
        mask_q = (offs_m[:, None] < seq_len) & (offs_d[None, :] < BLOCK_D)
        q = tl.load(q_ptrs, mask=mask_q, other=0.0)
        
        mask_k = mask_n[None, :] & (offs_d[:, None] < BLOCK_D)
        k = tl.load(k_ptrs + block_n * stride_kt, mask=mask_k, other=0.0)
        
        att = tl.dot(q, k) * scale
        att = tl.where(mask_mn, att, float('-inf'))
        att = tl.maximum(att, 0)
        
        mask_v = mask_n[:, None] & (offs_d[None, :] < BLOCK_D)
        v = tl.load(v_ptrs + block_n * stride_vt, mask=mask_v, other=0.0)
        
        att = tl.dot(att, v)
        acc += att
    
    mask_o = (offs_m[:, None] < seq_len) & (offs_d[None, :] < BLOCK_D)
    o_ptrs = o_ptr + pid_b * stride_ob + pid_h * stride_oh + (offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc, mask=mask_o)

class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        self.max_seqlen = max_seqlen

    def forward(self, x):
        B, T, C = x.size()
        hs = C // self.n_head
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        
        scale = 1.0 / math.sqrt(hs)
        output = torch.empty_like(v)
        
        grid = (B, self.n_head, triton.cdiv(T, 64))
        attention_kernel[grid](
            q, k, v, output,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            scale, T,
            BLOCK_M=64, BLOCK_N=64, BLOCK_D=hs
        )
        
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return output

batch_size = 16
max_seqlen = 1024
n_embd = 768
n_head = 12

def get_inputs():
    return [torch.randn(batch_size, max_seqlen, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, max_seqlen]
# =================== EVOLVE-BLOCK-END ===================