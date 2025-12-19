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


@triton.jit
def partial_rope_qk_compute(
    X,
    O,
    cos_l,
    cos_h,
    sin_l,
    sin_h,
    pos_offset,
    cso_dim_offset_l,
    cso_dim_offset_h,
    seq_mask,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    half_dim: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    x_dim_offset_l = NOPE_DIM + tl.arange(0, half_dim)
    x_dim_offset_h = x_dim_offset_l + half_dim
    for head_id in range(NUM_HEADS):
        x_base_offset = (pos_offset[:, None] * NUM_HEADS * HEAD_DIM) + (
            head_id * HEAD_DIM
        )
        x_l = tl.load(X + x_base_offset + x_dim_offset_l[None, :], mask=seq_mask)
        x_h = tl.load(X + x_base_offset + x_dim_offset_h[None, :], mask=seq_mask)
        o_l = x_l * cos_l - x_h * sin_l
        o_h = x_h * cos_h + x_l * sin_h
        o_base_offset = (pos_offset[:, None] * NUM_HEADS * ROPE_DIM) + (
            head_id * ROPE_DIM
        )
        tl.store(O + o_base_offset + cso_dim_offset_l[None, :], o_l, mask=seq_mask)
        tl.store(O + o_base_offset + cso_dim_offset_h[None, :], o_h, mask=seq_mask)


@triton.jit
def partial_rope_qk_kernel(
    Q,
    K,
    COS,
    SIN,
    Q_EMBED,
    K_EMBED,
    total_seq_len,
    NOPE_DIM_Q: tl.constexpr,
    NOPE_DIM_K: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_BLOCKS = tl.cdiv(total_seq_len, BLOCK)
    Q_HEAD_DIM = NOPE_DIM_Q + ROPE_DIM
    K_HEAD_DIM = NOPE_DIM_K + ROPE_DIM
    for seq_block_id in range(pid, NUM_BLOCKS, NUM_CORES):
        pos_offset = seq_block_id * BLOCK + tl.arange(0, BLOCK)
        pos_mask = pos_offset < total_seq_len
        half_dim: tl.constexpr = ROPE_DIM // 2
        cso_dim_offset_l = tl.arange(0, half_dim)
        cso_dim_offset_h = cso_dim_offset_l + half_dim
        seq_mask = pos_mask[:, None]
        cs_offset_l = pos_offset[:, None] * ROPE_DIM + cso_dim_offset_l[None, :]
        cs_offset_h = pos_offset[:, None] * ROPE_DIM + cso_dim_offset_h[None, :]

        cos_l = tl.load(COS + cs_offset_l, mask=seq_mask)
        cos_h = tl.load(COS + cs_offset_h, mask=seq_mask)
        sin_l = tl.load(SIN + cs_offset_l, mask=seq_mask)
        sin_h = tl.load(SIN + cs_offset_h, mask=seq_mask)

        partial_rope_qk_compute(
            X=Q,
            O=Q_EMBED,
            cos_l=cos_l,
            cos_h=cos_h,
            sin_l=sin_l,
            sin_h=sin_h,
            pos_offset=pos_offset,
            cso_dim_offset_l=cso_dim_offset_l,
            cso_dim_offset_h=cso_dim_offset_h,
            seq_mask=seq_mask,
            NOPE_DIM=NOPE_DIM_Q,
            ROPE_DIM=ROPE_DIM,
            half_dim=half_dim,
            NUM_HEADS=NUM_Q_HEADS,
            HEAD_DIM=Q_HEAD_DIM,
        )
        partial_rope_qk_compute(
            X=K,
            O=K_EMBED,
            cos_l=cos_l,
            cos_h=cos_h,
            sin_l=sin_l,
            sin_h=sin_h,
            pos_offset=pos_offset,
            cso_dim_offset_l=cso_dim_offset_l,
            cso_dim_offset_h=cso_dim_offset_h,
            seq_mask=seq_mask,
            NOPE_DIM=NOPE_DIM_K,
            ROPE_DIM=ROPE_DIM,
            half_dim=half_dim,
            NUM_HEADS=NUM_K_HEADS,
            HEAD_DIM=K_HEAD_DIM,
        )


def partial_rope_qk_triton(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
    assert q.is_contiguous() and k.is_contiguous()
    assert cos.shape == sin.shape
    seq_len = cos.numel() // cos.size(-1)
    assert seq_len == q.numel() // q.size(-1) // q.size(-2)
    assert seq_len == k.numel() // k.size(-1) // k.size(-2)
    rope_dim = cos.size(-1)
    num_heads_q = q.size(-2)
    num_heads_k = k.size(-2)
    q_embed = torch.empty(
        (seq_len, num_heads_q, rope_dim), dtype=q.dtype, device=q.device
    )
    k_embed = torch.empty(
        (seq_len, num_heads_k, rope_dim), dtype=k.dtype, device=k.device
    )
    partial_rope_qk_kernel[(NUM_CORES,)](
        Q=q,
        K=k,
        COS=cos,
        SIN=sin,
        Q_EMBED=q_embed,
        K_EMBED=k_embed,
        total_seq_len=seq_len,
        NOPE_DIM_Q=q.size(-1) - rope_dim,
        NOPE_DIM_K=k.size(-1) - rope_dim,
        ROPE_DIM=rope_dim,
        BLOCK=128,
        NUM_Q_HEADS=num_heads_q,
        NUM_K_HEADS=num_heads_k,
        NUM_CORES=NUM_CORES,
    )
    return q_embed, k_embed


# @triton.jit
# def partial_rope_compute(
#     X,
#     X_EMBED,
#     cos_l,
#     cos_h,
#     sin_l,
#     sin_h,
#     seq_len_id,
#     in_stride_s,
#     out_stride_s,
#     in_heads_offset,
#     out_heads_offset,
#     half_range,
#     HALF_ROPE_DIM: tl.constexpr,
# ):
#     in_seq_offset = seq_len_id * in_stride_s
#     in_rope_offset_l = in_seq_offset + in_heads_offset + half_range[None, :]
#     in_rope_offset_h = (
#         in_seq_offset + in_heads_offset + (HALF_ROPE_DIM + half_range)[None, :]
#     )
#     rope_l = tl.load(X + in_rope_offset_l)
#     rope_h = tl.load(X + in_rope_offset_h)

#     rope_out_l = rope_l * cos_l - rope_h * sin_l
#     rope_out_h = rope_h * cos_h + rope_l * sin_h

#     out_seq_offset = seq_len_id * out_stride_s
#     out_rope_offset_l = out_seq_offset + out_heads_offset + half_range[None, :]
#     out_rope_offset_h = (
#         out_seq_offset + out_heads_offset + (HALF_ROPE_DIM + half_range)[None, :]
#     )
#     tl.store(X_EMBED + out_rope_offset_l, rope_out_l)
#     tl.store(X_EMBED + out_rope_offset_h, rope_out_h)


# @triton.jit
# def partial_rope_qk_kernel(
#     Q,
#     K,
#     COS,
#     SIN,
#     Q_EMBED,
#     K_EMBED,
#     seq_len,
#     in_stride_qs: tl.constexpr,
#     in_stride_qh: tl.constexpr,
#     in_stride_ks: tl.constexpr,
#     in_stride_kh: tl.constexpr,
#     out_stride_qs: tl.constexpr,
#     out_stride_qh: tl.constexpr,
#     out_stride_ks: tl.constexpr,
#     out_stride_kh: tl.constexpr,
#     ROPE_HEAD_DIM: tl.constexpr,
#     NUM_HEADS: tl.constexpr,
#     NUM_CORES: tl.constexpr,
# ):
#     pid = tl.program_id(0)
#     HALF_ROPE_DIM: tl.constexpr = ROPE_HEAD_DIM // 2
#     half_range = tl.arange(0, HALF_ROPE_DIM)
#     in_q_heads_offset = tl.arange(0, NUM_HEADS)[:, None] * in_stride_qh
#     out_q_heads_offset = tl.arange(0, NUM_HEADS)[:, None] * out_stride_qh
#     in_k_heads_offset = tl.arange(0, NUM_HEADS)[:, None] * in_stride_kh
#     out_k_heads_offset = tl.arange(0, NUM_HEADS)[:, None] * out_stride_kh

#     for seq_len_id in range(pid, seq_len, NUM_CORES):
#         cs_offset_l = seq_len_id * ROPE_HEAD_DIM + half_range
#         cs_offset_h = seq_len_id * ROPE_HEAD_DIM + (HALF_ROPE_DIM + half_range)
#         cos_l = tl.load(COS + cs_offset_l)
#         cos_h = tl.load(COS + cs_offset_h)
#         sin_l = tl.load(SIN + cs_offset_l)
#         sin_h = tl.load(SIN + cs_offset_h)
#         cos_l = tl.broadcast_to(tl.expand_dims(cos_l, 0), (NUM_HEADS, HALF_ROPE_DIM))
#         cos_h = tl.broadcast_to(tl.expand_dims(cos_h, 0), (NUM_HEADS, HALF_ROPE_DIM))
#         sin_l = tl.broadcast_to(tl.expand_dims(sin_l, 0), (NUM_HEADS, HALF_ROPE_DIM))
#         sin_h = tl.broadcast_to(tl.expand_dims(sin_h, 0), (NUM_HEADS, HALF_ROPE_DIM))

#         partial_rope_compute(
#             X=Q,
#             X_EMBED=Q_EMBED,
#             cos_l=cos_l,
#             cos_h=cos_h,
#             sin_l=sin_l,
#             sin_h=sin_h,
#             seq_len_id=seq_len_id,
#             in_stride_s=in_stride_qs,
#             out_stride_s=out_stride_qs,
#             in_heads_offset=in_q_heads_offset,
#             out_heads_offset=out_q_heads_offset,
#             half_range=half_range,
#             HALF_ROPE_DIM=HALF_ROPE_DIM,
#         )
#         partial_rope_compute(
#             X=K,
#             X_EMBED=K_EMBED,
#             cos_l=cos_l,
#             cos_h=cos_h,
#             sin_l=sin_l,
#             sin_h=sin_h,
#             seq_len_id=seq_len_id,
#             in_stride_s=in_stride_ks,
#             out_stride_s=out_stride_ks,
#             in_heads_offset=in_k_heads_offset,
#             out_heads_offset=out_k_heads_offset,
#             half_range=half_range,
#             HALF_ROPE_DIM=HALF_ROPE_DIM,
#         )


# def partial_rope_qk_triton(
#     q: Tensor,
#     k: Tensor,
#     cos: Tensor,
#     sin: Tensor,
# ):
#     assert q.is_contiguous() and k.is_contiguous()
#     assert cos.shape == sin.shape

#     seq_len = cos.numel() // cos.size(-1)
#     assert seq_len == q.numel() // q.size(-1) // q.size(-2)
#     rotary_dim = cos.size(-1)
#     num_heads = q.size(-2)
#     q_embed = torch.empty(
#         (seq_len, num_heads, rotary_dim), dtype=q.dtype, device=q.device
#     )
#     k_embed = torch.empty(
#         (seq_len, num_heads, rotary_dim), dtype=k.dtype, device=k.device
#     )
#     partial_rope_qk_kernel[(NUM_CORES,)](
#         Q=q,
#         K=k,
#         COS=cos,
#         SIN=sin,
#         Q_EMBED=q_embed,
#         K_EMBED=k_embed,
#         seq_len=seq_len,
#         in_stride_qs=q.stride(-3),
#         in_stride_qh=q.stride(-2),
#         in_stride_ks=k.stride(-3),
#         in_stride_kh=k.stride(-2),
#         out_stride_qs=q_embed.stride(-3),
#         out_stride_qh=q_embed.stride(-2),
#         out_stride_ks=k_embed.stride(-3),
#         out_stride_kh=k_embed.stride(-2),
#         ROPE_HEAD_DIM=rotary_dim,
#         NUM_HEADS=num_heads,
#         NUM_CORES=NUM_CORES,
#     )
#     return q_embed, k_embed
