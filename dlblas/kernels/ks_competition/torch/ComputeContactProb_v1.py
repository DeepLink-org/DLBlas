# -*- coding: utf-8 -*-
import torch_npu
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, min_bin: float = 2.3125, max_bin: float = 21.6875, no_bins: int = 64, thres: float = 8.0):
        super().__init__()
        self.no_bins = int(no_bins)

        # 一次性预计算bin中心与阈值下标，全局常量
        edges = torch.linspace(min_bin, max_bin, self.no_bins + 1)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        self.thres_idx = int((bin_centers < thres).sum().item())
        self.register_buffer("bin_centers", bin_centers)

    def forward(self, distogram_logits: torch.Tensor) -> torch.Tensor:
        # 极简内联计算，无多余中间函数
        prob = torch.softmax(distogram_logits, dim=-1)
        contact_prob = prob[..., :self.thres_idx].sum(dim=-1)
        return contact_prob

# Hyperparameters
N_TOKEN = 256
NO_BINS = 64
MIN_BIN = 2.3125
MAX_BIN = 21.6875
THRES = 8.0

def get_inputs():
#    evice = 'npu:0'
# logits = torch.randn(N_TOKEN, N_TOKEN, NO_BINS, device=device)
    torch.manual_seed(42)
    logits = torch.randn(N_TOKEN, N_TOKEN, NO_BINS)
    return [logits]

def get_init_inputs():
    return [MIN_BIN, MAX_BIN, NO_BINS, THRES]

if __name__ == "__main__":
    torch_npu.npu.set_device(0)
    device = torch.device("npu:0")

    model = Model(*get_init_inputs()).to(device)
    # JIT编译加速
    model = torch.jit.script(model)

    inputs = get_inputs()
    # fp16混合精度推理
    with torch.npu.amp.autocast(dtype=torch.float16):
        res = model(*inputs)
    print(res.shape)
   