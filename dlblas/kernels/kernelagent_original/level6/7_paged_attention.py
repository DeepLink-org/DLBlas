import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


def _reconstruct_kv(
    blocked_k: Tensor,
    blocked_v: Tensor,
    block_offsets: Tensor,
    kv_len: int,
    block_size: int,
    kv_layout: str,
) -> Tuple[Tensor, Tensor]:
    """Reconstruct contiguous K/V from blocked caches for a single batch.

    Returns K: (kv_len, Hk, Dk), V: (kv_len, Hk, Dv)
    """
    if kv_layout == "bshd":
        num_heads_k = blocked_k.size(2)
        dim_k = blocked_k.size(3)
        dim_v = blocked_v.size(3)
    else:
        num_heads_k = blocked_k.size(1)
        dim_k = blocked_k.size(3)
        dim_v = blocked_v.size(3)

    k_seq = blocked_k.new_zeros((kv_len, num_heads_k, dim_k))
    v_seq = blocked_v.new_zeros((kv_len, num_heads_k, dim_v))

    num_blocks = (kv_len + block_size - 1) // block_size
    filled = 0
    for i in range(num_blocks):
        blk_id = int(block_offsets[i].item())
        take = min(block_size, kv_len - filled)
        if take <= 0:
            break
        if kv_layout == "bshd":
            k_chunk = blocked_k[blk_id, :take]  # (take, Hk, Dk)
            v_chunk = blocked_v[blk_id, :take]  # (take, Hk, Dv)
        else:  # bhsd
            k_chunk = blocked_k[blk_id, :, :take, :].transpose(0, 1).contiguous()
            v_chunk = blocked_v[blk_id, :, :take, :].transpose(0, 1).contiguous()
        k_seq[filled:filled + take] = k_chunk
        v_seq[filled:filled + take] = v_chunk
        filled += take

    return k_seq, v_seq


def paged_attention_ref(
    q: Tensor,
    blocked_k: Tensor,
    blocked_v: Tensor,
    block_offsets: Tensor,
    kv_seqlens: Tensor,
    kv_layout: str = "bshd",
    sm_scale: float = None,
) -> Tensor:
    """Reference PyTorch implementation of paged attention forward (decoding path).

    Args:
        q: (T, Hq, D)
        blocked_k: (B*Nb, S, Hk, D) if bshd, else (B*Nb, Hk, S, D)
        blocked_v: same layout as blocked_k with Dv
        block_offsets: (B, Nb)
        kv_seqlens: (B,)
        kv_layout: 'bshd' or 'bhsd'

    Returns:
        o: (T, Hq, Dv)
    """
    assert kv_layout in ("bshd", "bhsd")
    device = q.device
    dtype = q.dtype

    batch_size = kv_seqlens.numel()
    total_tokens, num_heads_q, head_dim = q.shape
    assert total_tokens % batch_size == 0
    seq_len = total_tokens // batch_size
    assert seq_len >= 1

    num_heads_k = blocked_k.size(2) if kv_layout == "bshd" else blocked_k.size(1)
    head_dim_v = blocked_v.size(-1)

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    out = q.new_empty(total_tokens, num_heads_q, head_dim_v)

    block_size = blocked_k.size(1) if kv_layout == "bshd" else blocked_k.size(2)

    # reshape q to (B, S, Hq, D)
    q_bshd = q.view(batch_size, seq_len, num_heads_q, head_dim)

    for b in range(batch_size):
        kv_len = int(kv_seqlens[b].item())
        k_seq, v_seq = _reconstruct_kv(
            blocked_k,
            blocked_v,
            block_offsets[b],
            kv_len,
            block_size,
            kv_layout,
        )
        # Shapes: q: (S, Hq, D), k: (Kv, Hk, D), v: (Kv, Hk, Dv)
        q_s = q_bshd[b]  # (S, Hq, D)
        # expand k/v to Hq via groups
        group = num_heads_q // num_heads_k
        # k: (Hq, D, Kv)
        k_hd = k_seq.permute(1, 2, 0)  # (Hk, D, Kv)
        k_hd = k_hd.unsqueeze(1).expand(-1, group, -1, -1).reshape(num_heads_q, head_dim, kv_len)
        # v: (Hq, Kv, Dv)
        v_hkv = v_seq.transpose(0, 1)  # (Hk, Kv, Dv)
        v_hkv = v_hkv.unsqueeze(1).expand(-1, group, -1, -1).reshape(num_heads_q, kv_len, head_dim_v)

        # q: (Hq, S, D)
        q_hsd = q_s.transpose(0, 1)

        # attention
        qk = torch.matmul(q_hsd, k_hd) * sm_scale  # (Hq, S, Kv)
        attn = torch.softmax(qk, dim=-1, dtype=torch.float32).to(dtype)
        o_hsd = torch.matmul(attn, v_hkv)  # (Hq, S, Dv)
        o_shd = o_hsd.transpose(0, 1)

        out[b * seq_len:(b + 1) * seq_len] = o_shd

    return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q: Tensor,
        blocked_k: Tensor,
        blocked_v: Tensor,
        block_offsets: Tensor,
        kv_seqlens: Tensor,
        kv_layout: str = "bshd",
    ) -> Tensor:
        return paged_attention_ref(q, blocked_k, blocked_v, block_offsets, kv_seqlens, kv_layout=kv_layout)


# Hyperparameters mirroring test setup
dtype = torch.float16
block_size = 16
kv_layout = "bshd"
num_heads_q = 8
num_heads_k = 2
head_dim = 32
head_dim_v = 32


def get_inputs():
    # history and single-token decode
    history_lens = torch.tensor([50, 40, 30, 20])
    seq_lens = torch.ones_like(history_lens)
    kv_seqlens = seq_lens + history_lens
    batch_size = history_lens.numel()

    # Build q (T=B, Hq, D)
    q_bshd = torch.rand(batch_size, 1, num_heads_q, head_dim, dtype=dtype)
    q = torch.cat([q_bshd[i, : seq_lens[i]] for i in range(batch_size)], dim=0)
    q = q.squeeze(1)  # (B, Hq, D)

    # Build blocked K/V
    num_blocks = (kv_seqlens + block_size - 1) // block_size
    max_num_blocks = int(num_blocks.max().item())
    total_blocks = batch_size * max_num_blocks
    blocked_k = torch.zeros(total_blocks, block_size, num_heads_k, head_dim, dtype=dtype)
    blocked_v = torch.zeros(total_blocks, block_size, num_heads_k, head_dim_v, dtype=dtype)

    # Build continuous K/V then fill into blocks to be realistic
    kv_max = int(kv_seqlens.max().item())
    bk = torch.rand(batch_size, kv_max, num_heads_k, head_dim, dtype=dtype)
    bv = torch.rand(batch_size, kv_max, num_heads_k, head_dim_v, dtype=dtype)

    block_offsets = [torch.arange(int(n)) * batch_size + i for i, n in enumerate(num_blocks.tolist())]
    max_len = max(len(x) for x in block_offsets)
    block_offsets_full = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, bo in enumerate(block_offsets):
        block_offsets_full[i, : len(bo)] = bo

    for b in range(batch_size):
        kv_len = int(kv_seqlens[b].item())
        filled = 0
        for bi in range(int(num_blocks[b].item())):
            blk_id = int(block_offsets_full[b, bi].item())
            take = min(block_size, kv_len - filled)
            if take <= 0:
                break
            blocked_k[blk_id, :take] = bk[b, filled : filled + take]
            blocked_v[blk_id, :take] = bv[b, filled : filled + take]
            filled += take

    return [q, blocked_k, blocked_v, block_offsets_full, kv_seqlens]


def get_init_inputs():
    return []
