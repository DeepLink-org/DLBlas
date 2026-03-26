import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


def flatten_kv_cache_ref(
    k_caches: Tensor,
    v_caches: Tensor,
    seqlens: Tensor,
    block_offsets: Tensor,
    out_size: int = None,
    kv_layout: str = "bshd",
    flatten_kv_layout: str = "hsd",
) -> Tuple[Tensor, Tensor]:
    """Pure PyTorch reference to flatten blocked KV caches into contiguous states.

    Args:
        k_caches: (B*Nb, S, H, D) if bshd else (B*Nb, H, S, D)
        v_caches: same layout as k_caches
        seqlens: (B,)
        block_offsets: (B, Nb)
        out_size: total tokens across batch; if None computed from seqlens
        kv_layout: 'bshd' or 'bhsd'
        flatten_kv_layout: 'hsd' or 'shd'
    Returns:
        k_states, v_states in requested flatten layout and dtype matching inputs
    """
    assert kv_layout in ("bshd", "bhsd")
    assert flatten_kv_layout in ("hsd", "shd")

    device = k_caches.device
    batch_size, num_blocks = block_offsets.shape

    if kv_layout == "bshd":
        s_dim = 1
        h_dim = 2
        d_dim = 3
    else:
        s_dim = 2
        h_dim = 1
        d_dim = 3

    block_size = k_caches.size(s_dim)
    num_heads = k_caches.size(h_dim)
    head_dim_k = k_caches.size(d_dim)
    head_dim_v = v_caches.size(d_dim)

    if out_size is None:
        out_size = int(seqlens.sum().item())

    if flatten_kv_layout == "hsd":
        k_states = torch.empty(num_heads, out_size, head_dim_k, dtype=k_caches.dtype, device=device)
        v_states = torch.empty(num_heads, out_size, head_dim_v, dtype=v_caches.dtype, device=device)
        head_first = True
    else:
        k_states = torch.empty(out_size, num_heads, head_dim_k, dtype=k_caches.dtype, device=device)
        v_states = torch.empty(out_size, num_heads, head_dim_v, dtype=v_caches.dtype, device=device)
        head_first = False

    start_loc = 0
    for b in range(batch_size):
        kv_len = int(seqlens[b].item())
        remain = kv_len
        bi = 0
        while remain > 0:
            blk_id = int(block_offsets[b, bi].item())
            take = min(block_size, remain)
            end_loc = start_loc + take
            if kv_layout == "bshd":
                k_block = k_caches[blk_id, :take]  # (take, H, D)
                v_block = v_caches[blk_id, :take]
            else:
                k_block = k_caches[blk_id, :, :take, :].transpose(0, 1).contiguous()
                v_block = v_caches[blk_id, :, :take, :].transpose(0, 1).contiguous()

            if head_first:
                k_states[:, start_loc:end_loc] = k_block.transpose(0, 1)
                v_states[:, start_loc:end_loc] = v_block.transpose(0, 1)
            else:
                k_states[start_loc:end_loc] = k_block
                v_states[start_loc:end_loc] = v_block

            start_loc = end_loc
            remain -= take
            bi += 1

    return k_states, v_states


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        k_caches: Tensor,
        v_caches: Tensor,
        seqlens: Tensor,
        block_offsets: Tensor,
        out_size: int = None,
        kv_layout: str = "bshd",
        flatten_kv_layout: str = "hsd",
    ) -> Tuple[Tensor, Tensor]:
        return flatten_kv_cache_ref(
            k_caches,
            v_caches,
            seqlens,
            block_offsets,
            out_size=out_size,
            kv_layout=kv_layout,
            flatten_kv_layout=flatten_kv_layout,
        )


dtype = torch.float16


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def get_inputs():
    num_heads = 4
    head_dim = 32
    block_size = 16
    kv_lens_list = [2, 24, 47, 48]
    batch_size = len(kv_lens_list)
    seqlens = torch.tensor(kv_lens_list)

    num_blocks_per_input = [_div_up(kv_len, block_size) for kv_len in kv_lens_list]
    max_num_blocks = max(num_blocks_per_input)
    out_size = sum(kv_lens_list)

    k_caches = torch.rand(batch_size * max_num_blocks, block_size, num_heads, head_dim, dtype=dtype)
    v_caches = torch.rand_like(k_caches)

    block_offsets = torch.arange(max_num_blocks)
    block_offsets = block_offsets.unsqueeze(0).repeat(batch_size, 1)
    block_offsets = block_offsets * batch_size + torch.arange(batch_size).unsqueeze(1)

    return [k_caches, v_caches, seqlens, block_offsets, out_size]


def get_init_inputs():
    return []
