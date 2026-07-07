import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Model that computes manual backward of mhc_head_compute_mix.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        input_mix: torch.Tensor,
        mhc_scale: torch.Tensor,
        mhc_base: torch.Tensor,
        grad_out: torch.Tensor,
    ):
        """
        Manual backward computation.

        Args:
            input_mix: (n0, n1, mhc_mult)
            mhc_scale: (1,)
            mhc_base: (mhc_mult,)
            grad_out: same shape as input_mix

        Returns:
            grad_input_mix, grad_mhc_scale, grad_mhc_base
        """

        # ---- forward intermediate ----
        z = input_mix * mhc_scale + mhc_base
        sigmoid = torch.sigmoid(z)

        # ---- sigmoid backward ----
        grad_z = grad_out * sigmoid * (1 - sigmoid)

        # ---- grad_input_mix ----
        grad_input_mix = grad_z * mhc_scale

        # ---- grad_mhc_base ----
        grad_mhc_base = grad_z.sum(dim=(0, 1), keepdim=True).view(-1)

        # ---- grad_mhc_scale ----
        grad_mhc_scale = (grad_z * input_mix).sum(dim=(0, 1, 2), keepdim=True).view(1)

        return grad_input_mix, grad_mhc_scale, grad_mhc_base

batch0 = 2
batch1 = 1024
mhc_mult = 4


def get_inputs():
    input_mix = torch.randn(batch0, batch1, mhc_mult, dtype=torch.float32)
    mhc_scale = torch.randn(1, dtype=torch.float32)
    mhc_base = torch.randn(mhc_mult, dtype=torch.float32)
    grad_out = torch.randn(batch0, batch1, mhc_mult, dtype=torch.float32)

    return [input_mix, mhc_scale, mhc_base, grad_out]


def get_init_inputs():
    return []