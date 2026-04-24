import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Implements:
        output = torch.sigmoid(input_mix * mhc_scale + mhc_base) + mhc_pre_eps
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        input_mix: torch.Tensor,
        mhc_scale: torch.Tensor,
        mhc_base: torch.Tensor,
        mhc_pre_eps: float,
    ) -> torch.Tensor:
        mhc_head_layer_mix = input_mix * mhc_scale + mhc_base
        return torch.sigmoid(mhc_head_layer_mix) + mhc_pre_eps


batch_size = 16
n1 = 16384
mhc_mult = 4


def get_inputs():
    input_mix = torch.randn(batch_size, n1, mhc_mult)
    mhc_scale = torch.randn(1)
    mhc_base = torch.randn(mhc_mult)
    mhc_pre_eps = 1e-2
    return [input_mix, mhc_scale, mhc_base, mhc_pre_eps]


def get_init_inputs():
    return []