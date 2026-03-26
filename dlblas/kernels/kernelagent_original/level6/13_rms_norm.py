import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Reference PyTorch RMSNorm.
    Inputs:
        hidden_states: (M, N)
        weight: (N)
        eps: float
        residual: optional (M, N)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor = None):
        x = hidden_states if residual is None else hidden_states + residual
        input_dtype = x.dtype
        xf = x.to(torch.float32)
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        y = xf * torch.rsqrt(var + self.eps)
        y = y.to(input_dtype)
        return weight * y if residual is None else (weight * y, x)


# Hyperparameters
seq_len = 65536
feat_size = 4096
dtype = torch.float16

def get_inputs():
    hidden_states = (torch.rand(seq_len, feat_size, dtype=dtype) - 0.5) / 2
    weight = (torch.rand(feat_size, dtype=dtype) - 0.5) / 2
    residual = (torch.rand(seq_len, feat_size, dtype=dtype) - 0.5) / 2
    return [hidden_states, weight, residual]

def get_init_inputs():
    return []


