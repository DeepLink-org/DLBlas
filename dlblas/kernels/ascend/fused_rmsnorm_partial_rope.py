import torch
import triton
import triton.language as tl
from torch import Tensor
from dlblas.utils.device_utils import NUM_CORES


@triton.jit
def fused_single_norm_and_partial_rope_kernel(
    # rmsnorm
    INPUT,
    NORM_WEIGHT,
    seq_len,
    eps: tl.constexpr,
    input_stride_s: tl.constexpr,
    input_stride_h: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    TOTAL_HEAD_DIM: tl.constexpr,
    # rotary_pos_emb
    COS,
    SIN,
    OUTPUT,
    ROPE_HEAD_DIM: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    out_stride_s: tl.constexpr = NUM_HEADS * TOTAL_HEAD_DIM
    HALF_ROPE_DIM: tl.constexpr = ROPE_HEAD_DIM // 2
    half_range = tl.arange(0, HALF_ROPE_DIM)
    NOPE_DIM: tl.constexpr = TOTAL_HEAD_DIM - ROPE_HEAD_DIM
    nope_range = tl.arange(0, NOPE_DIM)
    out_stride_h: tl.constexpr = TOTAL_HEAD_DIM
    out_heads_offset = tl.arange(0, NUM_HEADS)[:, None] * out_stride_h
    input_heads_offset = tl.arange(0, NUM_HEADS)[:, None] * input_stride_h

    w_rope_l = tl.load(NORM_WEIGHT + half_range)
    w_rope_h = tl.load(NORM_WEIGHT + HALF_ROPE_DIM + half_range)
    w_nope = tl.load(NORM_WEIGHT + ROPE_HEAD_DIM + nope_range)
    w_rope_l = tl.broadcast_to(tl.expand_dims(w_rope_l, 0), (NUM_HEADS, HALF_ROPE_DIM))
    w_rope_h = tl.broadcast_to(tl.expand_dims(w_rope_h, 0), (NUM_HEADS, HALF_ROPE_DIM))
    w_nope = tl.broadcast_to(tl.expand_dims(w_nope, 0), (NUM_HEADS, NOPE_DIM))

    for seq_len_id in range(pid, seq_len, NUM_CORES):
        input_seq_offset = seq_len_id * input_stride_s
        input_rope_offset_l = (
            input_seq_offset + input_heads_offset + half_range[None, :]
        )
        input_rope_offset_h = (
            input_seq_offset
            + input_heads_offset
            + (HALF_ROPE_DIM + half_range)[None, :]
        )
        input_nope_offset = (
            input_seq_offset
            + input_heads_offset
            + (ROPE_HEAD_DIM + nope_range)[None, :]
        )

        x_rope_l = tl.load(INPUT + input_rope_offset_l).to(tl.float32)
        x_rope_h = tl.load(INPUT + input_rope_offset_h).to(tl.float32)
        x_nope = tl.load(INPUT + input_nope_offset).to(tl.float32)
        sum = (
            tl.sum(x_rope_l * x_rope_l, -1)
            + tl.sum(x_rope_h * x_rope_h, -1)
            + tl.sum(x_nope * x_nope, -1)
        )

        var = sum / TOTAL_HEAD_DIM
        qrt = tl.expand_dims(tl.math.rsqrt(var + eps), -1)
        qrt_broadcast_to_rope = tl.broadcast_to(qrt, (NUM_HEADS, HALF_ROPE_DIM))
        norm_rope_l = x_rope_l * qrt_broadcast_to_rope * w_rope_l
        norm_rope_h = x_rope_h * qrt_broadcast_to_rope * w_rope_h
        norm_nope = x_nope * tl.broadcast_to(qrt, (NUM_HEADS, NOPE_DIM)) * w_nope
        norm_rope_l = norm_rope_l.to(w_nope.dtype)
        norm_rope_h = norm_rope_h.to(w_nope.dtype)
        norm_nope = norm_nope.to(w_nope.dtype)

        out_seq_offset = seq_len_id * out_stride_s
        out_rope_offset_l = out_seq_offset + out_heads_offset + half_range[None, :]
        out_rope_offset_h = (
            out_seq_offset + out_heads_offset + (HALF_ROPE_DIM + half_range)[None, :]
        )
        out_nope_offset = (
            out_seq_offset + out_heads_offset + (ROPE_HEAD_DIM + nope_range)[None, :]
        )
        tl.store(OUTPUT + out_nope_offset, norm_nope)

        cs_offset_l = seq_len_id * ROPE_HEAD_DIM + half_range
        cs_offset_h = seq_len_id * ROPE_HEAD_DIM + (HALF_ROPE_DIM + half_range)

        cos_l = tl.load(COS + cs_offset_l)
        cos_h = tl.load(COS + cs_offset_h)
        sin_l = tl.load(SIN + cs_offset_l)
        sin_h = tl.load(SIN + cs_offset_h)
        cos_l = tl.broadcast_to(tl.expand_dims(cos_l, 0), (NUM_HEADS, HALF_ROPE_DIM))
        cos_h = tl.broadcast_to(tl.expand_dims(cos_h, 0), (NUM_HEADS, HALF_ROPE_DIM))
        sin_l = tl.broadcast_to(tl.expand_dims(sin_l, 0), (NUM_HEADS, HALF_ROPE_DIM))
        sin_h = tl.broadcast_to(tl.expand_dims(sin_h, 0), (NUM_HEADS, HALF_ROPE_DIM))

        rope_out_l = norm_rope_l * cos_l - norm_rope_h * sin_l
        rope_out_h = norm_rope_h * cos_h + norm_rope_l * sin_h

        tl.store(OUTPUT + out_rope_offset_l, rope_out_l)
        tl.store(OUTPUT + out_rope_offset_h, rope_out_h)


def fused_single_norm_and_partial_rope_triton(
    x: Tensor,
    norm_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
    partial_rotary_factor: float,
    eps: float = 1e-6,
    inplace: bool = True,
):
    assert x.is_contiguous()
    head_dim = norm_weight.shape[0]
    seq_len = cos.numel() // cos.size(-1)
    assert seq_len == x.numel() // x.size(-1) // x.size(-2)
    rotary_dim = int(head_dim * partial_rotary_factor)
    assert rotary_dim == cos.size(-1) and rotary_dim == sin.size(-1)
    num_heads = x.size(-2)
    if inplace:
        assert head_dim == x.size(-1)
        x_embed = x
    else:
        x_embed = torch.empty(
            (seq_len, num_heads, head_dim), dtype=x.dtype, device=x.device
        )

    fused_single_norm_and_partial_rope_kernel[(NUM_CORES,)](
        # rmsnorm
        INPUT=x,
        NORM_WEIGHT=norm_weight,
        seq_len=seq_len,
        eps=eps,
        input_stride_s=x.stride(-3),
        input_stride_h=x.stride(-2),
        NUM_HEADS=num_heads,
        TOTAL_HEAD_DIM=head_dim,
        # rotary_pos_emb
        COS=cos,
        SIN=sin,
        OUTPUT=x_embed,
        ROPE_HEAD_DIM=rotary_dim,
        NUM_CORES=NUM_CORES,
    )
    return x_embed
