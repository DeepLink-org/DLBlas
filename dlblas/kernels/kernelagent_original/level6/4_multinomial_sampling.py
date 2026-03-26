import torch
import torch.nn as nn
from torch import Tensor


def multinomial_sampling_ref(scores: Tensor, seeds: Tensor, offsets: Tensor, indices: Tensor) -> Tensor:
    """Reference multinomial sampling.

    For the KernelBench test we mimic lmdeploy's behavior used in their test,
    which compares to indices[batch_ids, select_ids] when `scores` has a single
    1 at `select_ids` per batch. We implement that general case by taking argmax
    over `scores` and mapping through `indices`.

    Args:
        scores: (B, N)
        seeds: (B,) unused in reference
        offsets: (B,) unused in reference
        indices: (B, N) permutation mapping
    Returns:
        Tensor: (B,) selected token ids after mapping via `indices`.
    """
    batch_size = scores.size(0)
    sel = torch.argmax(scores.to(torch.float32), dim=1)
    batch_ids = torch.arange(batch_size, device=scores.device)
    out = indices[batch_ids, sel]
    return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scores: Tensor, seeds: Tensor, offsets: Tensor, indices: Tensor) -> Tensor:
        return multinomial_sampling_ref(scores, seeds, offsets, indices)


# Hyperparameters mirroring test setup
num_tokens = 2000
batch_size = 2  # len(select_ids)
dtype = torch.float16


def get_inputs():
    # Build scores with a single 1 at select positions per batch
    scores = torch.zeros(batch_size, num_tokens, dtype=dtype)
    batch_ids = torch.arange(batch_size)
    select_ids = (500, 1500)
    scores[batch_ids, torch.tensor(select_ids)] = 1

    # Seeds and offsets are unused in reference, keep API parity
    seeds = torch.randint(1000, 2000, (batch_size,), dtype=torch.int64)
    offsets = torch.randint(1000, 2000, (batch_size,), dtype=torch.int64)
    indices = torch.stack([torch.randperm(num_tokens) for _ in range(batch_size)], 0)
    return [scores, seeds, offsets, indices]


def get_init_inputs():
    return []

