"""
Task 05: TriAttentionFallback - Ascend NPU Implementation

Scaled dot-product attention with dimension reshaping for 5D inputs.
Uses F.scaled_dot_product_attention for efficient NPU execution.

Hardware: Huawei Ascend 910B2C
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        bias1: [B, N, S, 1, S] or None
        bias2: [B, N, 1, H, S, S] or None
    Returns:
        out: [B, N, S, H, D]
    """
    B, N, S, H, D = q.shape
    # Reshape to [B*N*H, S, D] for SDPA
    q2 = q.permute(0, 1, 3, 2, 4).reshape(B * N * H, S, D)
    k2 = k.permute(0, 1, 3, 2, 4).reshape(B * N * H, S, D)
    v2 = v.permute(0, 1, 3, 2, 4).reshape(B * N * H, S, D)

    attn_bias = None
    if bias1 is not None or bias2 is not None:
        attn_bias = 0.0
        if bias1 is not None:
            b1 = bias1.expand(B, N, S, H, S).permute(0, 1, 3, 2, 4)
            b1 = b1.reshape(B * N * H, S, S)
            attn_bias = attn_bias + b1
        if bias2 is not None:
            if bias2.dim() == 5:
                b2 = bias2
            else:
                b2 = bias2.squeeze(2)
            b2 = b2.reshape(B * N * H, S, S)
            attn_bias = attn_bias + b2

    out = F.scaled_dot_product_attention(
        q2, k2, v2, attn_mask=attn_bias, dropout_p=0.0, is_causal=False
    )
    out = out.reshape(B, N, H, S, D).permute(0, 1, 3, 2, 4)
    return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
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
    device = "npu:0"
    torch.manual_seed(42)

    q = torch.randn(B, N, S, H, D, device=device)
    k = torch.randn(B, N, S, H, D, device=device)
    v = torch.randn(B, N, S, H, D, device=device)

    return [q, k, v]


def get_init_inputs():
    return []


if __name__ == "__main__":
    torch.npu.set_device(0)
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    out = model(*inputs)
    print(f"Output shape: {out.shape}")
    print(f"Output device: {out.device}")
    print(out)
