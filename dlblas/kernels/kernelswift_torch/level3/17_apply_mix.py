
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        return (x * mix).sum(-2).bfloat16()

n0=2
n1=1024
mhc=4
h=1280

def generate_pre_apply_mix_test_data(
    n0: int, n1: int, mhc: int, h: int
) -> dict[str, torch.Tensor]:
    x = torch.randn(n0, n1, mhc, h, dtype=torch.bfloat16).sigmoid()
    mix = torch.randn(n0, n1, mhc, 1, dtype=torch.float32).softmax(-2)
    o_grad = torch.randn(n0, n1, h, dtype=torch.bfloat16)

    return [x,mix,o_grad]

def get_inputs():
    x,mix,o_grad = generate_pre_apply_mix_test_data(n0=n0, n1=n1, mhc=mhc, h=h)
    return [x,mix]

def get_init_inputs():
    return []

