from __future__ import annotations

import torch
from torch import nn

from clike_910b import load_library


class ModelNew(nn.Module):
    def __init__(self, repeat: int = 10, eps: float = 1e-6):
        super().__init__()
        self.repeat = repeat
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        load_library()
        return torch.ops.dlblas_ks_ascendc.sinkhorn(
            x.contiguous(), self.repeat, self.eps
        )


n0 = 1
n1 = 1024
mhc = 4


def get_inputs():
    return [torch.randn(n0, n1, mhc, mhc)]


def get_init_inputs():
    return []
