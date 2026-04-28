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
        # x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc], y: [b,s,hc,d]
        y = post_layer_mix * x.unsqueeze(-2) + torch.sum(comb_res_mix.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

n0 = 1
n1 = 4096
h = 1280
mhc_mult = 4
device = 'cuda'

def get_inputs():
    x = torch.randn((n0, n1, h), dtype=torch.bfloat16, device=device)
    residual = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn((n0, n1, mhc_mult, 1), dtype=torch.float32, device=device)
    comb_res_mix = torch.randn((n0, n1, mhc_mult, mhc_mult), dtype=torch.float32, device=device)

    return [
        x, residual, post_layer_mix, comb_res_mix,
    ]

def get_init_inputs():
    return []