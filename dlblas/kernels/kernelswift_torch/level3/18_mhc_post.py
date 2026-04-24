
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        term2 = torch.einsum('abmn,abmc->abnc', comb_res_mix, residual.float())
        return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()

n0=2
n1=4096
h=1280
mhc_mult=4

def generate_mhc_post_test_data(
    n0: int,
    n1: int,
    h: int,
    mhc_mult: int
) -> dict[str, torch.Tensor]:
    x = torch.randn((n0, n1, h), dtype=torch.bfloat16)
    residual = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16)
    post_layer_mix = torch.randn((n0, n1, mhc_mult, 1), dtype=torch.float32)
    comb_res_mix = torch.randn((n0, n1, mhc_mult, mhc_mult), dtype=torch.float32)

    o_grad = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16)
    return [x,residual,post_layer_mix,comb_res_mix,o_grad]

def get_inputs():
    x,residual,post_layer_mix,comb_res_mix,o_grad = generate_mhc_post_test_data(n0, n1, h, mhc_mult)
    return [x,residual,post_layer_mix,comb_res_mix]

def get_init_inputs():
    return []

