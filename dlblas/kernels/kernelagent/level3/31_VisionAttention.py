import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _residual_layernorm_fwd(
    x_ptr,          # attn output [rows, cols]
    r_ptr,          # residual     [rows, cols]
    gamma_ptr,      # ln weight    [cols]
    beta_ptr,       # ln bias      [cols]
    y_ptr,          # output       [rows, cols]
    n_cols,         # E
    eps,            # epsilon
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    row_offset = row_id * n_cols

    # Load as fp32 for numerical stability
    x = tl.load(x_ptr + row_offset + offs, mask=mask, other=0).to(tl.float32)
    r = tl.load(r_ptr + row_offset + offs, mask=mask, other=0).to(tl.float32)
    z = x + r

    # Compute mean
    mean = tl.sum(z, axis=0) / n_cols
    z_centered = z - mean

    # Compute variance
    var = tl.sum(z_centered * z_centered, axis=0) / n_cols
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine
    gamma = tl.load(gamma_ptr + offs, mask=mask, other=0).to(tl.float32)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0).to(tl.float32)
    y = z_centered * inv_std
    y = y * gamma + beta

    # Store (assume fp32 output tensor; matches input dtype in this workload)
    tl.store(y_ptr + row_offset + offs, y, mask=mask)


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _fused_residual_layernorm(attn_out: torch.Tensor,
                              residual: torch.Tensor,
                              weight: torch.Tensor,
                              bias: torch.Tensor,
                              eps: float) -> torch.Tensor:
    # attn_out/residual: (S, B, E) contiguous -> flatten to (rows, E)
    assert attn_out.is_contiguous()
    assert residual.is_contiguous()
    S, B, E = attn_out.shape
    rows = S * B
    x_flat = attn_out.view(rows, E)
    r_flat = residual.view(rows, E)
    y_flat = torch.empty_like(x_flat)

    # ensure parameters are contiguous on same device
    gamma = weight.contiguous()
    beta = bias.contiguous()

    BLOCK = min(1024, _next_power_of_2(E))
    if BLOCK <= 64:
        num_warps = 1
    elif BLOCK <= 128:
        num_warps = 2
    elif BLOCK <= 256:
        num_warps = 4
    else:
        num_warps = 8

    grid = (rows,)
    _residual_layernorm_fwd[grid](
        x_flat, r_flat, gamma, beta, y_flat,
        E, eps,
        BLOCK_SIZE=BLOCK,
        num_warps=num_warps,
        num_stages=2,
    )
    return y_flat.view(S, B, E)


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Attention Block using Multihead Self-Attention.
        :param embed_dim: Embedding dimension (the number of channels)
        :param num_heads: Number of attention heads
        """
        super(ModelNew, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass of the AttentionBlock.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of the same shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        S = H * W
        E = C
        x_seq = x.view(B, C, S).permute(2, 0, 1).contiguous()  # (S, B, E)
        residual = x_seq

        # Manual MHA using the module's weights to preserve semantics
        h = self.attn.num_heads
        d = E // h

        # In-projection
        if self.attn.in_proj_weight is not None:
            qkv = F.linear(x_seq, self.attn.in_proj_weight, self.attn.in_proj_bias)  # (S, B, 3E)
            q, k, v = qkv.split(E, dim=-1)
        else:
            # Fallback path if separate q/k/v weights exist (unlikely with default ctor)
            bias = self.attn.in_proj_bias
            bq, bk, bv = None, None, None
            if bias is not None:
                bq, bk, bv = bias.split(E, dim=0)
            q = F.linear(x_seq, self.attn.q_proj_weight, bq)
            k = F.linear(x_seq, self.attn.k_proj_weight, bk)
            v = F.linear(x_seq, self.attn.v_proj_weight, bv)

        # Shape to (B, H, S, d)
        q = q.view(S, B, h, d).permute(1, 2, 0, 3).contiguous()
        k = k.view(S, B, h, d).permute(1, 2, 0, 3).contiguous()
        v = v.view(S, B, h, d).permute(1, 2, 0, 3).contiguous()

        # Scaled Dot-Product Attention (dropout_p=0.0 to match default MHA)
        attn_ctx = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)  # (B, H, S, d)

        # Merge heads back to (S, B, E)
        attn_ctx = attn_ctx.permute(2, 0, 1, 3).contiguous().view(S, B, E)

        # Out projection
        attn_output = F.linear(attn_ctx, self.attn.out_proj.weight, self.attn.out_proj.bias)  # (S, B, E)

        # Fused residual + layernorm using Triton when CUDA, otherwise fallback
        if attn_output.is_cuda and residual.is_cuda:
            y = _fused_residual_layernorm(attn_output.contiguous(),
                                          residual.contiguous(),
                                          self.norm.weight,
                                          self.norm.bias,
                                          self.norm.eps)
        else:
            y = self.norm(attn_output + residual)

        # Back to (B, C, H, W)
        y = y.permute(1, 2, 0).contiguous().view(B, C, H, W)
        return y


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