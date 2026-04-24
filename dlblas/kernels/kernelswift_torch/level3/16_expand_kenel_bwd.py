# model.py

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that simulates the backward of expand_to_mhc operation.
    It reduces (sums) along the broadcasted dimension.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, o_grad: torch.Tensor) -> torch.Tensor:
        """
        Simulated backward of expand operation.

        Args:
            o_grad (torch.Tensor): Gradient tensor of shape
                                  (n0, n1, mhc_mult, h)

        Returns:
            torch.Tensor: Reduced gradient of shape (n0, n1, h)
        """
        return o_grad.sum(dim=-2)


# ----------------------------
# Test input configuration
# ----------------------------
batch_n0 = 2
batch_n1 = 1024
mhc_mult = 4
hidden_dim = 1280


def get_inputs():
    o_grad = torch.randn(batch_n0, batch_n1, mhc_mult, hidden_dim)
    return [o_grad]


def get_init_inputs():
    return []