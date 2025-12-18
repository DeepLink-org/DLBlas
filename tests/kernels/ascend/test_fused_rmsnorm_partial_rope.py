import torch
from dlblas.kernels.ascend.fused_rmsnorm_partial_rope import (
    fused_single_norm_and_partial_rope_triton,
)
from dlblas.kernels.ascend.rms_norm import rms_norm_block_triton
from tests.kernels.ascend.common import benchmark_test


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_ref(q, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # b, h, s, d = q.shape
    # q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    # b, h, s, d = k.shape
    # k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


def test_fused_single_norm_and_partial_rope_k(do_bench=False):
    device_ = "npu"
    dtype_ = torch.float16
    seq_len = 4096
    num_heads = 16
    head_dim = 256
    partial_rotary_factor = 0.25
    rope_head_dim = int(partial_rotary_factor * head_dim)
    q = torch.randn((seq_len, num_heads, head_dim), dtype=dtype_, device=device_)
    q_test = q.clone()

    norm_weight = torch.randn((head_dim,), dtype=dtype_, device=device_)
    cos = torch.randn((seq_len, rope_head_dim), dtype=dtype_, device=device_)
    sin = torch.randn((seq_len, rope_head_dim), dtype=dtype_, device=device_)
    q_out_triton = fused_single_norm_and_partial_rope_triton(
        q, norm_weight, cos, sin, partial_rotary_factor, inplace=True
    )
    q_calc_rope_emb = q_out_triton[..., :rope_head_dim]
    q_calc_nope = q_out_triton[..., rope_head_dim:]
    q_norm = rms_norm_block_triton(q_test, norm_weight, eps=1e-06)
    q_norm_rope = q_norm[..., :rope_head_dim]
    q_ref_nope = q_norm[..., rope_head_dim:]
    q_ref_rope_emb = apply_rotary_pos_emb_ref(q_norm_rope, cos, sin)

    torch.testing.assert_close(q_ref_rope_emb, q_calc_rope_emb, rtol=0.02, atol=0.02)
    torch.testing.assert_close(q_ref_nope, q_calc_nope, rtol=0.02, atol=0.02)
    if do_bench:
        benchmark_test(
            fused_single_norm_and_partial_rope_triton,
            fused_single_norm_and_partial_rope_triton,
            (q, norm_weight, cos, sin, partial_rotary_factor),
            "fused_single_norm_and_partial_rope_triton",
        )


def test_fused_single_norm_and_partial_rope_qq(do_bench=False):
    device_ = "npu"
    dtype_ = torch.float16
    seq_len = 4096
    num_heads = 16
    head_dim = 256
    partial_rotary_factor = 0.25
    rope_head_dim = int(partial_rotary_factor * head_dim)
    qq = torch.randn((seq_len, num_heads, 2 * head_dim), dtype=dtype_, device=device_)
    qa, qb = qq.chunk(2, dim=-1)
    q_test = qa.contiguous().clone()

    norm_weight = torch.randn((head_dim,), dtype=dtype_, device=device_)
    cos = torch.randn((seq_len, rope_head_dim), dtype=dtype_, device=device_)
    sin = torch.randn((seq_len, rope_head_dim), dtype=dtype_, device=device_)
    q_out_triton = fused_single_norm_and_partial_rope_triton(
        qq, norm_weight, cos, sin, partial_rotary_factor, inplace=False
    )
    q_calc_rope_emb = q_out_triton[..., :rope_head_dim]
    q_calc_nope = q_out_triton[..., rope_head_dim:]
    q_norm = rms_norm_block_triton(q_test, norm_weight, eps=1e-06)
    q_norm_rope = q_norm[..., :rope_head_dim]
    q_ref_nope = q_norm[..., rope_head_dim:]
    q_ref_rope_emb = apply_rotary_pos_emb_ref(q_norm_rope, cos, sin)

    torch.testing.assert_close(q_ref_rope_emb, q_calc_rope_emb, rtol=0.02, atol=0.02)
    torch.testing.assert_close(q_ref_nope, q_calc_nope, rtol=0.02, atol=0.02)
    if do_bench:
        benchmark_test(
            fused_single_norm_and_partial_rope_triton,
            fused_single_norm_and_partial_rope_triton,
            (qq, norm_weight, cos, sin, partial_rotary_factor, 1e-6, False),
            "fused_single_norm_and_partial_rope_triton",
        )


if __name__ == "__main__":
    test_fused_single_norm_and_partial_rope_qq(do_bench=True)
