import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Silu and Mul
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        gate = F.silu(gate)
        out = gate * up
        return out

# Hyperparameters
seqlen = 65536
feat_size = 4096

def get_inputs():
    return [torch.rand(seqlen, feat_size)]

def get_init_inputs():
    return []