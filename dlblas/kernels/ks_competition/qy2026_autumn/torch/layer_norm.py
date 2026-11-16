import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return torch.nn.LayerNorm(10).to("npu")(x)


def get_inputs():
    x = torch.rand(10, 10, device="npu")
    return [x]


def get_init_inputs():
    return []


torch.manual_seed(42)
out = Model(*get_init_inputs()).forward(*get_inputs())
print(out)
