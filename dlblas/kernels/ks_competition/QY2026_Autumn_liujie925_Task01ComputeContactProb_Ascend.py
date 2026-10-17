# -*- coding: utf-7 -*-
import torch_npu
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, min_bin: float = 2.3125, max_bin: float = 21.6875, no_bins: int = 64, thres: float = 8.0):
        super().__init__()
        no_bins = int(no_bins)
        self.no_bins = no_bins
        edges = torch.linspace(min_bin, max_bin, no_bins + 1)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        thres_idx_val = int((bin_centers < thres).sum().item())
        self.register_buffer("bin_centers", bin_centers)
        self.register_buffer("thres_idx", torch.tensor(thres_idx_val, dtype=torch.long))

    def forward(self, distogram_logits: torch.Tensor) -> torch.Tensor:
        N = distogram_logits.shape[0]
        # 取上三角索引（不含对角线，对角线也可根据需求保留）
        triu_idx = torch.triu_indices(N, N, offset=0, device=distogram_logits.device)
        logits_triu = distogram_logits[triu_idx[0], triu_idx[1], :]   # [M, 64]

        prob_triu = torch.softmax(logits_triu, dim=-1)
        contact_triu = prob_triu[:, :self.thres_idx].sum(dim=-1)

        # 构造对称矩阵（对角线直接赋值，非对角线同时赋值对称元素）
        contact_prob = torch.zeros(N, N, device=distogram_logits.device, dtype=contact_triu.dtype)
        contact_prob[triu_idx[0], triu_idx[1]] = contact_triu
        contact_prob = contact_prob + contact_prob.T
        # 如果上三角包含对角线，对角线加了两次，需减半
        if (triu_idx[0] == triu_idx[1]).any():
            diag_mask = triu_idx[0] == triu_idx[1]
            contact_prob[triu_idx[0][diag_mask], triu_idx[1][diag_mask]] = contact_triu[diag_mask]
        # prob = torch.softmax(distogram_logits, dim=-1)
        # contact_prob = prob[..., :self.thres_idx.item()].sum(dim=-1)

        return contact_prob

# Hyperparameters
N_TOKEN = 256
NO_BINS = 64
MIN_BIN = 2.3125
MAX_BIN = 21.6875
THRES = 8.0

def get_inputs():
    device = 'npu:0'
    torch.manual_seed(42)
    logits = torch.randn(N_TOKEN, N_TOKEN, NO_BINS, device=device)
    return [logits]

def get_init_inputs():
    return [MIN_BIN, MAX_BIN, NO_BINS, THRES]

if __name__ == "__main__":
    torch_npu.npu.set_device(0)
    device = torch.device("npu:0")

    raw_model = Model(*get_init_inputs()).to(device)
   
    inputs = get_inputs()

    # 关键替换：用trace而非script，不读取源码
    traced_model = torch.jit.trace(raw_model, inputs)

    # 混合精度推理
    with torch.npu.amp.autocast(dtype=torch.float16):
        res = traced_model(*inputs)
    print("输出shape:", res.shape)
    print(res)
