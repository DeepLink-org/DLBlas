import torch
import torch.nn as nn
from torch import Tensor


def fused_lora_ref(
    input: Tensor,
    fused_lora_a: Tensor,
    fused_lora_b: Tensor,
    scaling: Tensor,
    rank_start: Tensor,
    ranks: Tensor,
    seq_start: Tensor,
    seq_lens: Tensor,
    adapter_ids: Tensor,
) -> Tensor:
    """Reference fused LoRA application across variable-length sequences.

    Args align with lmdeploy test expectations.
    """
    out_list = []
    for loc, s_len, r_id in zip(seq_start, seq_lens, adapter_ids):
        loc_i = int(loc.item())
        s_len_i = int(s_len.item())
        rank_off = int(rank_start[r_id].item())
        rank_len = int(ranks[r_id].item())
        a_sub = fused_lora_a[rank_off:rank_off + rank_len].t().contiguous()  # (H, r)
        b_sub = fused_lora_b[rank_off:rank_off + rank_len]  # (r, O)
        s = scaling[r_id]
        x = input[loc_i:loc_i + s_len_i]
        out_list.append(x @ a_sub @ b_sub * s)
    return torch.cat(out_list, dim=0)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input: Tensor,
        fused_lora_a: Tensor,
        fused_lora_b: Tensor,
        scaling: Tensor,
        rank_start: Tensor,
        ranks: Tensor,
        seq_start: Tensor,
        seq_lens: Tensor,
        adapter_ids: Tensor,
        max_rank: int,
        max_seqlen: int,
    ) -> Tensor:
        return fused_lora_ref(
            input,
            fused_lora_a,
            fused_lora_b,
            scaling=scaling,
            rank_start=rank_start,
            ranks=ranks,
            seq_start=seq_start,
            seq_lens=seq_lens,
            adapter_ids=adapter_ids,
        )


# Hyperparameters mirroring test setup
dtype = torch.float16
head_size = 32
out_head_size = 16


def get_inputs():
    # Sequence lengths and ranks
    seq_lens = torch.tensor((2, 4, 6, 8))
    ranks = torch.tensor([2, 4])
    start_loc = seq_lens.cumsum(0) - seq_lens
    total_len = int(seq_lens.sum().item())

    # Deterministic inputs
    torch.manual_seed(123)
    input = torch.rand(total_len, head_size, dtype=dtype)

    # Build per-rank LoRA and fuse
    lora_a = [torch.rand(head_size, int(r), dtype=dtype) for r in ranks]
    lora_b = [torch.rand(int(r), out_head_size, dtype=dtype) for r in ranks]
    fused_lora_a = torch.cat(lora_a, dim=1).t().contiguous()
    fused_lora_b = torch.cat(lora_b, dim=0).contiguous()

    adapter_ids = (torch.arange(len(seq_lens)) % len(ranks)).contiguous()
    scaling = (torch.arange(len(ranks)) + 1).contiguous()
    rank_offset = ranks.cumsum(0) - ranks
    max_rank = int(ranks.max().item())
    max_seqlen = int(seq_lens.max().item())

    return [
        input,
        fused_lora_a,
        fused_lora_b,
        scaling,
        rank_offset,
        ranks,
        start_loc,
        seq_lens,
        adapter_ids,
        max_rank,
        max_seqlen,
    ]


def get_init_inputs():
    return []

