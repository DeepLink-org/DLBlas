# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_npu

    _NPU_FA = torch_npu.npu_fusion_attention
except Exception:
    _NPU_FA = None


_SDPA = F.scaled_dot_product_attention
_NPU_FA_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


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
        bias2: [B, N, 1, H, S, S] or [B, N, H, S, S] or None

    Returns:
        out: [B, N, S, H, D]
    """
    B, N, S, H, D = q.shape

    # Fast path: no bias + NPU fused attention.
    # Original layout [B, N, S, H, D]
    # Merge B and N -> [B*N, S, H, D], which matches BSND layout.
    if bias1 is None and bias2 is None and _NPU_FA is not None:
        if q.device.type == "npu" and q.dtype in _NPU_FA_DTYPES:
            try:
                out = _NPU_FA(
                    q.flatten(0, 1),
                    k.flatten(0, 1),
                    v.flatten(0, 1),
                    H,
                    "BSND",
                    scale=D**-0.5,
                    keep_prob=1.0,
                )[0]

                return out.reshape(B, N, S, H, D)
            except Exception:
                pass

    # General fallback path.
    # SDPA expects [..., H, S, D], so move H before S.
    q = q.transpose(2, 3)
    k = k.transpose(2, 3)
    v = v.transpose(2, 3)

    attn_bias = None

    if bias1 is not None:
        # [B, N, S, 1, S] -> [B, N, 1, S, S]
        # Let SDPA broadcast over H.
        attn_bias = bias1.transpose(2, 3)

    if bias2 is not None:
        # [B, N, 1, H, S, S] -> [B, N, H, S, S]
        b2 = bias2.squeeze(2) if bias2.dim() == 6 else bias2
        attn_bias = b2 if attn_bias is None else attn_bias + b2

    out = _SDPA(
        q,
        k,
        v,
        attn_mask=attn_bias,
        dropout_p=0.0,
        is_causal=False,
    )

    return out.transpose(2, 3)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

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
