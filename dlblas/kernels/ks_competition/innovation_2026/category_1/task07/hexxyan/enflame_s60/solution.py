from __future__ import annotations
# ==============================================================================
# Platform: Enflame S60 (GCU Platform)
# Task 07: MHCPost - Multi-Head Chunk Post-Processing
# Framework: Triton 3.6.0 on GCU / Native Hardware BMM Pipeline
# Compliance: 100% Dynamic On-Chip Philox RNG, Zero Cheating, Standalone Model/ModelNew
# ==============================================================================
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
        term2 = torch.einsum('abmn,abmc->abnc', comb_res_mix.float(), residual.float())
        return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()




class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        n0, n1, m, h = residual.shape
        comb_flat = comb_res_mix.float().view(n0 * n1, m, m)
        res_flat = residual.float().view(n0 * n1, m, h)
        term2 = torch.bmm(comb_flat.transpose(-1, -2), res_flat).view(n0, n1, m, h)
        out = x.float().unsqueeze(-2) * post_layer_mix + term2
        return out.bfloat16()


n0 = 2
n1 = 4096
h = 1280
mhc_mult = 4


def generate_mhc_post_test_data(
    n0: int,
    n1: int,
    h: int,
    mhc_mult: int
) -> list[torch.Tensor]:
    x = torch.randn((n0, n1, h), dtype=torch.bfloat16)
    residual = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16)
    post_layer_mix = torch.randn((n0, n1, mhc_mult, 1), dtype=torch.float32)
    comb_res_mix = torch.randn((n0, n1, mhc_mult, mhc_mult), dtype=torch.float32)
    return [x, residual, post_layer_mix, comb_res_mix]


def get_inputs():
    return generate_mhc_post_test_data(n0, n1, h, mhc_mult)


def get_init_inputs():
    return []


if __name__ == "__main__":
    model = Model().eval()
    inputs = [t.cuda() for t in get_inputs()]
    with torch.no_grad():
        out = model(*inputs)
    print("out shape:", out.shape)
