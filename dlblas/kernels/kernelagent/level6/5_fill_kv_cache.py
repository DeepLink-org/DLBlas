import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Tuple


# Robust import shim for utils.get_device_props if needed in future
def _noop(*args, **kwargs):
    return None


@triton.jit
def _div_up(val, other):
    return (val + other - 1) // other


@triton.jit
def _fill_kv_cache_kernel(
    KStates,
    VStates,
    KCaches,
    VCaches,
    QStartLoc,
    QSeqLens,
    KVSeqLens,
    BlockOffsets,
    is_decoding: tl.constexpr,
    head_dim: tl.constexpr,
    head_dim_v: tl.constexpr,
    stride_kss,
    stride_ksh,
    stride_ksd,
    stride_vss,
    stride_vsh,
    stride_vsd,
    stride_kcn: tl.constexpr,
    stride_kcb: tl.constexpr,
    stride_kch: tl.constexpr,
    stride_kcd: tl.constexpr,
    stride_vcn: tl.constexpr,
    stride_vcb: tl.constexpr,
    stride_vch: tl.constexpr,
    stride_vcd: tl.constexpr,
    stride_boff,
    BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    batch_id = tl.program_id(2)
    head_id = tl.program_id(0)
    block_id = tl.program_id(1)

    q_startloc = tl.load(QStartLoc + batch_id)
    q_seqlen = tl.load(QSeqLens + batch_id)
    kv_seqlen = tl.load(KVSeqLens + batch_id)
    history_seqlen = kv_seqlen - q_seqlen

    kv_block_id = history_seqlen // BLOCK + block_id
    if kv_seqlen <= 0:
        return
    if kv_block_id * BLOCK >= kv_seqlen:
        return

    if is_decoding:
        page_offs = tl.full((1, ), history_seqlen % BLOCK, dtype=tl.int32)
        kv_mask = tl.full((1, ), 1, dtype=tl.int1)
        q_offs = tl.full((1, ), q_startloc, dtype=tl.int32)
    else:
        page_offs = tl.arange(0, BLOCK)
        kv_offs = kv_block_id * BLOCK + page_offs
        kv_mask = (kv_offs >= history_seqlen) & (kv_offs < kv_seqlen)
        token_off = q_startloc + kv_block_id * BLOCK - history_seqlen
        q_offs = token_off + page_offs

    block_off = tl.load(BlockOffsets + batch_id * stride_boff + kv_block_id)

    d_off = tl.arange(0, BLOCK_D)
    mask_ks = kv_mask[:, None]
    mask_kc = mask_ks & (d_off[None, :] < head_dim)
    d_off = d_off % head_dim

    ks_ptr = KStates + head_id * stride_ksh
    ks_ptrs = ks_ptr + q_offs[:, None] * stride_kss + d_off[None, :] * stride_ksd
    kc_ptr = KCaches + block_off * stride_kcn + head_id * stride_kch
    kc_ptrs = kc_ptr + page_offs[:, None] * stride_kcb + d_off[None, :] * stride_kcd

    if BLOCK_DV > 0:
        dv_off = tl.arange(0, BLOCK_DV)
        mask_vs = kv_mask[:, None]
        mask_vc = mask_vs & (dv_off[None, :] < head_dim_v)
        dv_off = dv_off % head_dim_v
        vs_ptr = VStates + head_id * stride_vsh
        vs_ptrs = vs_ptr + q_offs[:, None] * stride_vss + dv_off[None, :] * stride_vsd
        vc_ptr = VCaches + block_off * stride_vcn + head_id * stride_vch
        vc_ptrs = vc_ptr + page_offs[:, None] * stride_vcb + dv_off[None, :] * stride_vcd

    k = tl.load(ks_ptrs, mask=mask_ks)
    if BLOCK_DV > 0:
        v = tl.load(vs_ptrs, mask=mask_vs)
    tl.store(kc_ptrs, k, mask=mask_kc)
    if BLOCK_DV > 0:
        tl.store(vc_ptrs, v, mask=mask_vc)


def fill_kv_cache(
    k_states: Tensor,
    v_states: Tensor,
    k_caches: Tensor,
    v_caches: Tensor,
    q_start_loc: Tensor,
    q_seq_length: Tensor,
    kv_seq_length: Tensor,
    max_q_seq_length: int,
    block_offsets: Tensor,
    kv_layout: str = 'bshd',
) -> Tuple[Tensor, Tensor]:
    if kv_layout == 'bshd':
        b_dim, s_dim, h_dim, d_dim = (0, 1, 2, 3)
    elif kv_layout == 'bhsd':
        b_dim, s_dim, h_dim, d_dim = (0, 2, 1, 3)
    else:
        raise RuntimeError('Unsupported layout.')

    block_offsets = block_offsets.contiguous()
    batch_size = block_offsets.size(0)
    block_size = k_caches.size(s_dim)
    num_heads = k_caches.size(h_dim)
    head_dim = k_caches.size(d_dim)
    head_dim_v = v_states.size(-1)
    if max_q_seq_length == 1:
        max_num_blocks = 1
    else:
        max_num_blocks = triton.cdiv(max_q_seq_length, block_size) + 1

    BLOCK = block_size
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_DV = triton.next_power_of_2(head_dim_v)
    if k_caches.data_ptr() == v_caches.data_ptr() and head_dim_v <= head_dim:
        BLOCK_DV = 0

    grid = (num_heads, max_num_blocks, batch_size)
    is_decoding = max_num_blocks == 1
    _fill_kv_cache_kernel[grid](
        k_states,
        v_states,
        k_caches,
        v_caches,
        q_start_loc,
        q_seq_length,
        kv_seq_length,
        block_offsets,
        is_decoding=is_decoding,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        stride_kss=k_states.stride(-3),
        stride_ksh=k_states.stride(-2),
        stride_ksd=k_states.stride(-1),
        stride_vss=v_states.stride(-3),
        stride_vsh=v_states.stride(-2),
        stride_vsd=v_states.stride(-1),
        stride_kcn=k_caches.stride(b_dim),
        stride_kcb=k_caches.stride(s_dim),
        stride_kch=k_caches.stride(h_dim),
        stride_kcd=k_caches.stride(d_dim),
        stride_vcn=v_caches.stride(b_dim),
        stride_vcb=v_caches.stride(s_dim),
        stride_vch=v_caches.stride(h_dim),
        stride_vcd=v_caches.stride(d_dim),
        stride_boff=block_offsets.stride(0),
        BLOCK=BLOCK,
        BLOCK_D=BLOCK_D,
        BLOCK_DV=BLOCK_DV,
        num_warps=4,
        num_stages=3,
    )
    return k_caches, v_caches


class ModelNew(torch.nn.Module):
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
        kv_layout: str = 'bshd',
    ) -> Tuple[Tensor, Tensor]:
        return fill_kv_cache(
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

