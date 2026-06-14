"""
Compute Contact Probability (distogram logits -> contact probability)

From: protenix/model/sample_confidence.py:compute_contact_prob

Optimizations (NPU):
  1. reshape 3D→2D for optimal softmax memory layout
  2. torch.mv (matrix-vector) instead of mm avoids [N,1] intermediate
  3. torch.jit.script fuses softmax+mv into fewer kernel launches
  4. mask pre-cached as 1D buffer (no per-forward allocation)
"""

import torch
import torch.nn as nn
import torch_npu


def get_bin_centers(min_bin: float, max_bin: float, no_bins: int) -> torch.Tensor:
    edges = torch.linspace(min_bin, max_bin, no_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers


@torch.jit.script
def _contact_prob_kernel(logits: torch.Tensor, mask: torch.Tensor, n0: int, n1: int, no_bins: int) -> torch.Tensor:
    logits_2d = logits.reshape(-1, no_bins)
    prob = torch.softmax(logits_2d, dim=-1)
    return torch.mv(prob, mask).reshape(n0, n1)


class ModelNew(nn.Module):
    def __init__(self, min_bin: float = 2.3125, max_bin: float = 21.6875, no_bins: int = 64, thres: float = 8.0):
        super().__init__()
        self.min_bin = float(min_bin)
        self.max_bin = float(max_bin)
        self.no_bins = int(no_bins)
        self.thres = float(thres)
        bins = get_bin_centers(min_bin, max_bin, no_bins)
        thres_idx = int((bins < thres).sum().item())
        mask = torch.zeros(no_bins)
        mask[:thres_idx] = 1.0
        self.register_buffer('mask', mask)

    def forward(self, distogram_logits: torch.Tensor) -> torch.Tensor:
        n0, n1 = distogram_logits.shape[0], distogram_logits.shape[1]
        mask = self.mask.to(distogram_logits.device)
        return _contact_prob_kernel(distogram_logits, mask, n0, n1, self.no_bins)


# ==========================================
# Hyperparameters & Data Generation
# ==========================================

N_TOKEN = 256
NO_BINS = 64
MIN_BIN = 2.3125
MAX_BIN = 21.6875
THRES = 8.0


def get_inputs():
    device = 'npu'
    torch.manual_seed(42)
    distogram_logits = torch.randn(N_TOKEN, N_TOKEN, NO_BINS, device=device)
    return [distogram_logits]


def get_init_inputs():
    return [MIN_BIN, MAX_BIN, NO_BINS, THRES]


if __name__ == "__main__":
    torch.set_default_device("npu")
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)
    print(output)
    print(f"Shape: {output.shape}")
