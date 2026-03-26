import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def fill_kv_cache_ref(
    k_states: Tensor,
    v_states: Tensor,
    k_caches: Tensor,
    v_caches: Tensor,
    q_start_loc: Tensor,
    q_seq_length: Tensor,
    kv_seq_length: Tensor,
    max_q_seq_length: int,
    block_offsets: Tensor,
    kv_layout: str = "bshd",
) -> Tuple[Tensor, Tensor]:
    """Pure PyTorch reference for filling paged KV caches.

    Args:
        k_states: (num_tokens, num_heads, head_dim)
        v_states: (num_tokens, num_heads, head_dim_v)
        k_caches: (num_blocks_total, block_size, num_heads, head_dim) if kv_layout == 'bshd'
                  (num_blocks_total, num_heads, block_size, head_dim) if kv_layout == 'bhsd'
        v_caches: same layout as k_caches, but with head_dim_v
        q_start_loc: (batch,)
        q_seq_length: (batch,)
        kv_seq_length: (batch,)
        max_q_seq_length: int (unused in reference but kept for API parity)
        block_offsets: (batch, max_num_blocks)
        kv_layout: 'bshd' or 'bhsd'

    Returns:
        (k_caches, v_caches) after in-place updates
    """
    assert kv_layout in ("bshd", "bhsd")

    device = k_states.device
    assert v_states.device == device
    assert k_caches.device == device and v_caches.device == device

    block_size = k_caches.size(1) if kv_layout == "bshd" else k_caches.size(2)
    batch_size = q_start_loc.numel()

    for batch_idx in range(batch_size):
        start = int(q_start_loc[batch_idx].item())
        seqlen = int(q_seq_length[batch_idx].item())
        kvlen = int(kv_seq_length[batch_idx].item())
        history_len = kvlen - seqlen

        # Determine starting block and in-block offset
        block_id = _div_up(history_len + 1, block_size) - 1
        fill_start = history_len % block_size

        token_offset = 0
        while token_offset < seqlen:
            current_block_offset = int(block_offsets[batch_idx, block_id].item())
            tokens_to_copy = min(block_size - fill_start, seqlen - token_offset)

            src_slice = slice(start + token_offset, start + token_offset + tokens_to_copy)
            pos_slice = slice(fill_start, fill_start + tokens_to_copy)

            if kv_layout == "bshd":
                k_caches[current_block_offset, pos_slice] = k_states[src_slice]
                v_caches[current_block_offset, pos_slice] = v_states[src_slice]
            else:  # "bhsd": (blocks, heads, seq, dim)
                # transpose (T, H, D) -> (H, T, D) for direct assignment
                k_src = k_states[src_slice].transpose(0, 1).contiguous()
                v_src = v_states[src_slice].transpose(0, 1).contiguous()
                k_caches[current_block_offset, :, pos_slice] = k_src
                v_caches[current_block_offset, :, pos_slice] = v_src

            token_offset += tokens_to_copy
            block_id += 1
            fill_start = 0

    return k_caches, v_caches


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        k_states: Tensor,
        v_states: Tensor,
        k_caches: Tensor,
        v_caches: Tensor,
        q_start_loc: Tensor,
        q_seq_length: Tensor,
        kv_seq_length: Tensor,
        max_q_seq_length: int,
        block_offsets: Tensor,
        kv_layout: str = "bshd",
    ) -> Tuple[Tensor, Tensor]:
        return fill_kv_cache_ref(
            k_states,
            v_states,
            k_caches,
            v_caches,
            q_start_loc,
            q_seq_length,
            kv_seq_length,
            max_q_seq_length,
            block_offsets,
            kv_layout,
        )


# Hyperparameters mirroring test setup
batch_size = 4
num_heads = 4
head_dim = 32
head_dim_v = 32
block_size = 16
dtype = torch.float16


def get_inputs():
    # variable seqlens/history
    seq_lens = torch.tensor([1, 8, 16, 24], dtype=torch.int32)
    history_lens = torch.tensor([1, 16, 31, 24], dtype=torch.int32)
    kv_lens = seq_lens + history_lens
    max_q_seq_length = int(seq_lens.max().item())
    num_tokens = int(seq_lens.sum().item())

    # States: (T, H, D)
    k_states = torch.rand(num_tokens, num_heads, head_dim, dtype=dtype)
    v_states = torch.rand(num_tokens, num_heads, head_dim_v, dtype=dtype)

    # Blocked caches layout bshd: (B*max_blocks, S, H, D)
    num_blocks_per_input = [_div_up(int(kv), block_size) for kv in kv_lens.tolist()]
    max_num_blocks = max(num_blocks_per_input)
    total_blocks = batch_size * max_num_blocks
    k_caches = torch.full((total_blocks, block_size, num_heads, head_dim), 0.0, dtype=dtype)
    v_caches = torch.full((total_blocks, block_size, num_heads, head_dim_v), 0.0, dtype=dtype)

    # Offsets and locs
    q_seq_length = seq_lens
    kv_seq_length = kv_lens
    q_start_loc = q_seq_length.cumsum(0) - q_seq_length
    block_offsets = torch.arange(max_num_blocks, dtype=torch.int32)
    block_offsets = block_offsets.unsqueeze(0).repeat(batch_size, 1)
    block_offsets = block_offsets * batch_size + torch.arange(batch_size, dtype=torch.int32).unsqueeze(1)

    return [
        k_states,
        v_states,
        k_caches,
        v_caches,
        q_start_loc,
        q_seq_length,
        kv_seq_length,
        max_q_seq_length,
        block_offsets,
    ]


def get_init_inputs():
    return []
