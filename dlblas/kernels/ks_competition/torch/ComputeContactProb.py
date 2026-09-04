"""
Compute Contact Probability (distogram logits -> contact probability)

From: protenix/model/sample_confidence.py:compute_contact_prob
"""

import torch
import torch.nn as nn


def get_bin_centers(min_bin: float, max_bin: float, no_bins: int) -> torch.Tensor:
    """
    distogram bins centers锛堝父瑙佸仛娉曪細绾挎€х瓑闂撮殧锛�
    """
    edges = torch.linspace(min_bin, max_bin, no_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers


def compute_contact_prob(
    distogram_logits: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    thres: float = 8.0,
) -> torch.Tensor:
    """
    Args:
        distogram_logits: [N_token, N_token, no_bins]
    Returns:
        contact_prob: [N_token, N_token]
    """
    distogram_prob = torch.softmax(distogram_logits, dim=-1)
    bins = get_bin_centers(min_bin, max_bin, no_bins).to(distogram_logits.device)
    thres_idx = int((bins < thres).sum().item())
    return distogram_prob[..., :thres_idx].sum(dim=-1)


class Model(nn.Module):
    def __init__(self, min_bin: float = 2.3125, max_bin: float = 21.6875, no_bins: int = 64, thres: float = 8.0):
        super().__init__()
        self.min_bin = float(min_bin)
        self.max_bin = float(max_bin)
        self.no_bins = int(no_bins)
        self.thres = float(thres)

    def forward(self, distogram_logits: torch.Tensor) -> torch.Tensor:
        return compute_contact_prob(
            distogram_logits=distogram_logits,
            min_bin=self.min_bin,
            max_bin=self.max_bin,
            no_bins=self.no_bins,
            thres=self.thres,
        )


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
    print(model(*inputs))