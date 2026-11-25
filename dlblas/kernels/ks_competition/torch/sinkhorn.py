# fmt: off
import torch
import torch.nn as nn
class Model(nn.Module):
    """
    Pure PyTorch implementation of sinkhorn_normalize.
    Iteratively normalizes a matrix to be doubly stochastic:
      1. softmax(x, dim=-1) + eps
      2. column-normalize: x / (x.sum(-2) + eps)
      3. repeat (row-normalize then column-normalize) for repeat-1 iterations
    """
    def __init__(self, repeat: int = 10, eps: float = 1e-6):
        super().__init__()
        self.repeat = repeat
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., mhc, mhc] float32
        Returns:
            [..., mhc, mhc] float32  (doubly stochastic)
        """
        x = x.softmax(-1) + self.eps
        x = x / (x.sum(-2, keepdim=True) + self.eps)
        for _ in range(self.repeat - 1):
            x = x / (x.sum(-1, keepdim=True) + self.eps)
            x = x / (x.sum(-2, keepdim=True) + self.eps)
        return x
n0 = 1
n1 = 1024
mhc = 4
def get_inputs():
    x = torch.randn(n0, n1, mhc, mhc)
    return [x]
def get_init_inputs():
    return []
# fmt: on
