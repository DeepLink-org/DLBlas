import torch

from dlblas.kernels.ascend.apply_rotary_pos_emb import apply_rotary_pos_emb_triton


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_ref(q, cos, sin, unsqueeze_dim=2):
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


def test_apply_rotary_pos_emb():
    device_ = "npu"
    dtype_ = torch.float16
    b = 3
    seq_len = 4096
    num_heads = 128  # N: head number
    qk_rope_head_dim = 64  # Dr: qk 位置编码维度
    q = torch.randn(
        (b, seq_len, num_heads, qk_rope_head_dim),
        dtype=dtype_,
        device=device_,
    )
    k = torch.randn(
        (b, seq_len, 1, qk_rope_head_dim),
        dtype=dtype_,
        device=device_,
    )
    cos = torch.randn((b, seq_len, qk_rope_head_dim), dtype=dtype_, device=device_)
    sin = torch.randn((b, seq_len, qk_rope_head_dim), dtype=dtype_, device=device_)
    q_pe_triton = apply_rotary_pos_emb_triton(q, cos, sin)
    k_pe_triton = apply_rotary_pos_emb_triton(k, cos, sin)
    q_pe_ref = apply_rotary_pos_emb_ref(q, cos, sin)
    kr_cache_ref = apply_rotary_pos_emb_ref(k, cos, sin)
    torch.testing.assert_close(q_pe_ref, q_pe_triton, rtol=1e-02, atol=1e-02)
    torch.testing.assert_close(kr_cache_ref, k_pe_triton, rtol=1e-02, atol=1e-02)
