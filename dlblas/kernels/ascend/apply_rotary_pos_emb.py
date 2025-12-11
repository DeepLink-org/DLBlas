import torch
import triton
import triton.language as tl
from torch import Tensor
from dlblas.utils.device_utils import NUM_CORES


@triton.jit
def rotary_emb_compute(
    X,
    X_EMBED,
    cos_h,
    cos_l,
    sin_h,
    sin_l,
    base_offset,
    feat_offset_l,
    feat_offset_h,
    seq_mask,
):
    x_l = tl.load(X + base_offset + feat_offset_l[None, :], mask=seq_mask)
    x_h = tl.load(X + base_offset + feat_offset_h[None, :], mask=seq_mask)
    o_l = x_l * cos_l - x_h * sin_l
    o_h = x_h * cos_h + x_l * sin_h
    tl.store(X_EMBED + base_offset + feat_offset_l[None, :], o_l, mask=seq_mask)
    tl.store(X_EMBED + base_offset + feat_offset_h[None, :], o_h, mask=seq_mask)


@triton.jit
def apply_rotary_pos_emb_kernel(
    X,
    COS,
    SIN,
    O,
    seq_len,
    stride_xs: tl.constexpr,
    stride_xh: tl.constexpr,
    stride_xd: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_BLOCKS = tl.cdiv(seq_len, BLOCK)
    for seq_block_id in range(pid, NUM_BLOCKS, NUM_CORES):
        pos_offset = seq_block_id * BLOCK + tl.arange(0, BLOCK)
        pos_mask = pos_offset < seq_len
        half_dim: tl.constexpr = DIM // 2
        feat_offset_l = tl.arange(0, half_dim)
        feat_offset_h = half_dim + feat_offset_l
        seq_mask = pos_mask[:, None]
        cs_offset_l = pos_offset[:, None] * DIM + feat_offset_l[None, :]
        cs_offset_h = pos_offset[:, None] * DIM + feat_offset_h[None, :]

        cos_l = tl.load(COS + cs_offset_l, mask=seq_mask)
        cos_h = tl.load(COS + cs_offset_h, mask=seq_mask)
        sin_l = tl.load(SIN + cs_offset_l, mask=seq_mask)
        sin_h = tl.load(SIN + cs_offset_h, mask=seq_mask)

        for head_id in range(NUM_HEADS):
            base_offset = pos_offset[:, None] * stride_xs + head_id * stride_xh
            rotary_emb_compute(
                X,
                O,
                cos_h,
                cos_l,
                sin_h,
                sin_l,
                base_offset,
                feat_offset_l,
                feat_offset_h,
                seq_mask,
            )


def apply_rotary_pos_emb_triton(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
):
    assert x.is_contiguous()
    assert x.size(-1) == cos.size(-1)
    assert x.size(-1) == sin.size(-1)
    x_embed = torch.empty_like(x)
    seq_len = cos.numel() // cos.size(-1)
    assert seq_len == x.numel() // x.size(-1) // x.size(-2)
    BLOCK = 128
    apply_rotary_pos_emb_kernel[(NUM_CORES,)](
        x,
        cos,
        sin,
        x_embed,
        seq_len=seq_len,
        stride_xs=x.stride(-3),
        stride_xh=x.stride(-2),
        stride_xd=x.stride(-1),
        DIM=x.size(-1),
        BLOCK=BLOCK,
        NUM_HEADS=x.size(-2),
        NUM_CORES=NUM_CORES,
    )
    return x_embed


@triton.jit
def partial_rotary_emb_kernel(
    Q,
    K,
    COS,
    SIN,
    Q_EMBED,
    K_EMBED,
    seq_len,
    stride_qs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kh: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_BLOCKS = tl.cdiv(seq_len, BLOCK)
    for seq_block_id in range(pid, NUM_BLOCKS, NUM_CORES):
        pos_offset = seq_block_id * BLOCK + tl.arange(0, BLOCK)
        pos_mask = pos_offset < seq_len
        half_dim: tl.constexpr = DIM // 2
        feat_offset_l = tl.arange(0, half_dim)
        feat_offset_h = half_dim + feat_offset_l
        seq_mask = pos_mask[:, None]
        cs_offset_l = pos_offset[:, None] * DIM + feat_offset_l[None, :]
        cs_offset_h = pos_offset[:, None] * DIM + feat_offset_h[None, :]

        cos_l = tl.load(COS + cs_offset_l, mask=seq_mask)
        cos_h = tl.load(COS + cs_offset_h, mask=seq_mask)
        sin_l = tl.load(SIN + cs_offset_l, mask=seq_mask)
        sin_h = tl.load(SIN + cs_offset_h, mask=seq_mask)
        for head_id in range(NUM_Q_HEADS):
            base_offset = pos_offset[:, None] * stride_qs + head_id * stride_qh
            rotary_emb_compute(
                Q,
                Q_EMBED,
                cos_h,
                cos_l,
                sin_h,
                sin_l,
                base_offset,
                feat_offset_l,
                feat_offset_h,
                seq_mask,
            )
        for head_id in range(NUM_K_HEADS):
            base_offset = pos_offset[:, None] * stride_ks + head_id * stride_kh
            rotary_emb_compute(
                K,
                K_EMBED,
                cos_h,
                cos_l,
                sin_h,
                sin_l,
                base_offset,
                feat_offset_l,
                feat_offset_h,
                seq_mask,
            )


def partial_apply_rotary_pos_emb_triton(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    partial_rotary_factor: float,
    inplace: bool = False,
):
    assert q.is_contiguous() and k.is_contiguous()
    assert q.shape == k.shape
    assert q.size(-1) == cos.size(-1)
    assert q.size(-1) == sin.size(-1)
    if inplace:
        q_embed, k_embed = q, k
    else:
        q_embed = torch.empty_like(q)
        k_embed = torch.empty_like(k)
    seq_len = cos.numel() // cos.size(-1)
    assert seq_len == q.numel() // q.size(-1) // q.size(-2)
    rotary_dim = q.size(-1) * partial_rotary_factor
    BLOCK = 128
    partial_rotary_emb_kernel[(NUM_CORES,)](
        q,
        k,
        cos,
        sin,
        q_embed,
        k_embed,
        seq_len=seq_len,
        stride_xs=q.stride(-3),
        stride_xh=q.stride(-2),
        stride_xd=q.stride(-1),
        DIM=rotary_dim,
        BLOCK=BLOCK,
        NUM_HEADS=q.size(-2),
        NUM_CORES=NUM_CORES,
    )
    return q_embed, k_embed
