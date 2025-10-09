# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def _attn_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    B, H, M, N, K_dim,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
    
    q = tl.load(q_ptr, mask=offs_m[:, None] < M, other=0.0)
    k = tl.load(k_ptr, mask=offs_n[:, None] < N, other=0.0)
    v = tl.load(v_ptr, mask=offs_n[:, None] < N, other=0.0)
    
    q = q.to(tl.float32)
    k = k.to(tl.float32)
    v = v.to(tl.float32)
    
    S = tl.dot(q, tl.trans(k)) * scale
    S = tl.where(S == 0, -float('inf'), S)
    m = tl.max(S, axis=1)
    S = S - m[:, None]
    P = tl.exp(S)
    l = tl.sum(P, axis=1)
    P = P / l[:, None]
    
    acc = tl.dot(P, v)
    acc = acc.to(q.dtype)
    
    out_ptr = Out + pid_b * stride_ob + pid_h * stride_oh + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    tl.store(out_ptr, acc, mask=offs_m[:, None] < M)

class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        M = H * W
        x = x.view(B, C, M).permute(2, 0, 1)  # (M, B, C)
        
        q = self.q_proj(x)  # (M, B, C)
        k = self.k_proj(x)  # (M, B, C)
        v = self.v_proj(x)  # (M, B, C)
        
        # Reshape to (B, H, M, D)
        q = q.permute(1, 0, 2).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.permute(1, 0, 2).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.permute(1, 0, 2).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        out = torch.empty_like(q)
        scale = 1.0 / math.sqrt(self.head_dim)
        
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = self.head_dim
        
        grid = lambda META: (B, self.num_heads, triton.cdiv(M, BLOCK_M))
        _attn_kernel[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B, self.num_heads, M, M, self.head_dim,
            scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )
        
        out = out.permute(0, 2, 1, 3).reshape(B, M, C)
        out = self.out_proj(out)
        
        residual = x.permute(1, 0, 2)
        out = self.norm(out + residual)
        out = out.permute(1, 0, 2).reshape(B, C, H, W)
        return out

embed_dim = 128
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 128
image_width = 128

def get_inputs():
    return [torch.randn(batch_size, num_channels, image_height, image_width)]

def get_init_inputs():
    return [embed_dim, num_heads]
# =================== EVOLVE-BLOCK-END ===================