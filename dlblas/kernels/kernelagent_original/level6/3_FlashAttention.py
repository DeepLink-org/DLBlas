import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Simple model that performs Flash Attention (scaled dot-product attention).
    Uses PyTorch's fused SDPA kernel when available (FlashAttention / memory-efficient).
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies scaled dot-product attention to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
                              This tensor is used as Q, K, V (self-attention).

        Returns:
            torch.Tensor: Attention output of shape (batch_size, seq_len, dim).
        """
        # Self-attention: Q = K = V = x
        q = x
        k = x
        v = x

        # Prefer PyTorch fused kernels (FlashAttention) on supported GPUs/dtypes.
        # For maximum compatibility, keep dropout_p=0.0 and is_causal=False.
        try:
            with torch.backends.cpu.sdp_kernel(
                enable_flash=True,
                enable_math=True,
                enable_mem_efficient=True,
            ):
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )
        except Exception:
            # Fallback (math attention) if SDPA kernel selection/context isn't available.
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )

        return out

batch_size = 16
seq_len = 128
dim = 512

def get_inputs():
    # SDPA/FlashAttention is most effective on cpu with fp16/bf16.
    # Keep this generic; caller can move to cpu and cast as needed.
    x = torch.randn(batch_size, seq_len, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed