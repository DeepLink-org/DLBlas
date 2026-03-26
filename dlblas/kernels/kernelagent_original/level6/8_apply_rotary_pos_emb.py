import torch
import torch.nn as nn


def _rope_ref(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: (S, H, D), cos/sin: (S, D)
    S = cos.size(0)
    rope_dim = cos.size(1)
    D = q.size(-1)
    if D < rope_dim:
        raise ValueError("head_dim < rope_dim is not supported")
    half = rope_dim // 2
    ql, qh = q[..., :half], q[..., half:rope_dim]
    kl, kh = k[..., :half], k[..., half:rope_dim]
    # reshape cos/sin to broadcast across head dimension
    cos_l, cos_h = cos[..., :half].unsqueeze(1), cos[..., half:].unsqueeze(1)
    sin_l, sin_h = sin[..., :half].unsqueeze(1), sin[..., half:].unsqueeze(1)
    ql2 = ql * cos_l - qh * sin_l
    qh2 = qh * cos_h + ql * sin_h
    kl2 = kl * cos_l - kh * sin_l
    kh2 = kh * cos_h + kl * sin_h
    q_out = torch.cat([ql2, qh2, q[..., rope_dim:]], dim=-1)
    k_out = torch.cat([kl2, kh2, k[..., rope_dim:]], dim=-1)
    return q_out, k_out


class Model(nn.Module):
    """Reference PyTorch rotary positional embedding for q, k."""

    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        return _rope_ref(q, k, cos, sin)


# Hyperparameters (mirroring test shapes/dtype)
S = 128
Hq = 8
Hk = 8
D = 128
rope_dim = 128
dtype = torch.float16


def get_inputs():
    q = (torch.rand(S, Hq, D, dtype=dtype) - 0.5) / 2
    k = (torch.rand(S, Hk, D, dtype=dtype) - 0.5) / 2
    cos = torch.rand(S, rope_dim, dtype=dtype)
    sin = torch.rand(S, rope_dim, dtype=dtype)
    return [q, k, cos, sin]


def get_init_inputs():
    return []
