import torch
import triton
import triton.language as tl
import triton.language.extra.deeplink as dl


def _compute_inv_freq(base: float, head_size) -> torch.Tensor:
    """Compute the inverse frequency."""
    # NOTE(woosuk): To exactly match the HF implementation, we need to
    # use CPU to compute the cache and then move it to GPU. However, we
    # create the cache on GPU for faster initialization. This may cause
    # a slight numerical difference between the HF implementation and ours.
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, head_size, 2, dtype=torch.float, device="npu") / head_size)
    )
    return inv_freq


def _compute_cos_sin_cache(head_size) -> torch.Tensor:
    """Compute the cos and sin cache."""
    inv_freq = _compute_inv_freq(10000.0, head_size)
    t = torch.arange(4096, dtype=torch.float32, device="npu")

    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)
    return cache


@triton.jit
def _compute_rotary_emb(
    x1,
    x2,
    cos,
    sin,
):
    cos = tl.expand_dims(cos, -2)
    sin = tl.expand_dims(sin, -2)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return o1, o2


@triton.jit
def rms_norm_rope_kernel(
    q1,
    q2,
    k1,
    k2,
    v,
    weight,
    cos,
    sin,
    q1_out,
    q2_out,
    k1_out,
    k2_out,
    v_out,
    q_size: tl.constexpr,
    kv_size: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SUB_BLK: tl.constexpr,
):
    id = tl.program_id(0)
    num_q_offsets = tl.arange(0, q_size // head_dim)
    num_kv_offsets = tl.arange(0, kv_size // head_dim)
    head_offsets = tl.arange(0, head_dim)
    half_head_offsets = tl.arange(0, head_dim // 2)
    num_token_offsets = tl.arange(0, BLOCK_SIZE) + id * BLOCK_SIZE
    half_q_size = q_size // 2
    half_kv_size = kv_size // 2
    half_dim = head_dim // 2

    q1_data = tl.load(
        q1 + num_token_offsets[:, None, None] * half_q_size
           + num_q_offsets[None, :, None] * half_dim
           + half_head_offsets[None, None, :],
    )
    q2_data = tl.load(
        q2 + num_token_offsets[:, None, None] * half_q_size
           + num_q_offsets[None, :, None] * half_dim
           + half_head_offsets[None, None, :],
    )
    k1_data = tl.load(
        k1 + num_token_offsets[:, None, None] * half_kv_size
           + num_kv_offsets[None, :, None] * half_dim
           + half_head_offsets[None, None, :],
    )
    k2_data = tl.load(
        k2 + num_token_offsets[:, None, None] * half_kv_size
           + num_kv_offsets[None, :, None] * half_dim
           + half_head_offsets[None, None, :],
    )
    v_data = tl.load(
        v
        + num_token_offsets[:, None, None] * kv_size
        + num_kv_offsets[None, :, None] * head_dim
        + head_offsets[None, None, :]
    )
    cos_data = tl.load(
        cos
        + num_token_offsets[:, None] * head_dim // 2
        + half_head_offsets[None, :]
    )
    sin_data = tl.load(
        sin
        + num_token_offsets[:, None] * head_dim // 2
        + half_head_offsets[None, :]
    )
    weight_data = tl.load(weight + half_head_offsets)

    for s in dl.parallel(0, 2, bind_sub_block=False):
        q1_sub_data = dl.extract_slice(
            q1_data, (s * SUB_BLK, 0, 0), (SUB_BLK, q_size // head_dim, head_dim // 2), (1, 1, 1)
        )
        q2_sub_data = dl.extract_slice(
            q2_data, (s * SUB_BLK, 0, 0), (SUB_BLK, q_size // head_dim, head_dim // 2), (1, 1, 1)
        )
        k1_sub_data = dl.extract_slice(
            k1_data, (s * SUB_BLK, 0, 0), (SUB_BLK, kv_size // head_dim, head_dim // 2), (1, 1, 1)
        )
        k2_sub_data = dl.extract_slice(
            k2_data, (s * SUB_BLK, 0, 0), (SUB_BLK, kv_size // head_dim, head_dim // 2), (1, 1, 1)
        )
        cos_sub_data = dl.extract_slice(
            cos_data, (s * SUB_BLK, 0), (SUB_BLK, head_dim // 2), (1, 1)
        )
        sin_sub_data = dl.extract_slice(
            sin_data, (s * SUB_BLK, 0), (SUB_BLK, head_dim // 2), (1, 1)
        )

        # rms norm
        var_q1 = tl.sum(q1_sub_data * q1_sub_data, -1) / head_dim
        var_q2 = tl.sum(q2_sub_data * q2_sub_data, -1) / head_dim
        var_q = var_q1 + var_q2
        var_q = tl.expand_dims(var_q, -1)
        q1_sub_data = q1_sub_data * tl.math.rsqrt(var_q + 1e-5)
        q2_sub_data = q2_sub_data * tl.math.rsqrt(var_q + 1e-5)
        k1_sub_data = tl.view(k1_sub_data, SUB_BLK, kv_size // head_dim, head_dim // 2)
        k2_sub_data = tl.view(k2_sub_data, SUB_BLK, kv_size // head_dim, head_dim // 2)
        q1_sub_data = weight_data * q1_sub_data
        q2_sub_data = weight_data * q2_sub_data
        var_k1 = tl.sum(k1_sub_data * k1_sub_data, axis=-1) / head_dim
        var_k2 = tl.sum(k2_sub_data * k2_sub_data, axis=-1) / head_dim
        var_k = var_k1 + var_k2
        var_k = tl.expand_dims(var_k, -1)
        k1_sub_data = k1_sub_data * tl.math.rsqrt(var_k + 1e-5)
        k2_sub_data = k2_sub_data * tl.math.rsqrt(var_k + 1e-5)
        k1_sub_data = weight_data * k1_sub_data
        k2_sub_data = weight_data * k2_sub_data

        # rotary embedding
        q1_rope, q2_rope = _compute_rotary_emb(q1_sub_data, q2_sub_data, cos_sub_data, sin_sub_data)
        k1_rope, k2_rope = _compute_rotary_emb(k1_sub_data, k2_sub_data, cos_sub_data, sin_sub_data)
        num_token_sub_offsets = tl.arange(0, SUB_BLK) + id * BLOCK_SIZE + s * SUB_BLK

        tl.store(
            q1_out
            + num_token_sub_offsets[:, None, None] * half_q_size
            + num_q_offsets[None, :, None] * half_dim
            + half_head_offsets[None, None, :],
            q1_rope,
        )
        tl.store(
            q2_out
            + num_token_sub_offsets[:, None, None] * half_q_size
            + num_q_offsets[None, :, None] * half_dim
            + half_head_offsets[None, None, :],
            q2_rope,
        )
        tl.store(
            k1_out
            + num_token_sub_offsets[:, None, None] * half_kv_size
            + num_kv_offsets[None, :, None] * half_dim
            + half_head_offsets[None, None, :],
            k1_rope,
        )
        tl.store(
            k2_out
            + num_token_sub_offsets[:, None, None] * half_kv_size
            + num_kv_offsets[None, :, None] * half_dim
            + half_head_offsets[None, None, :],
            k2_rope,
        )

    tl.store(
        v_out
        + num_token_offsets[:, None, None] * kv_size
        + num_kv_offsets[None, :, None] * head_dim
        + head_offsets[None, None, :],
        v_data,
    )


def rms_norm_rope(
    qkv,
    positions,
    num_heads_q,
    num_heads_kv,
    head_dim,
    num_tokens,
    BLOCK_SIZE,
):
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim

    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q1, q2 = q.view(num_tokens, num_heads_q, head_dim).chunk(2, dim=-1)
    k1, k2 = k.view(num_tokens, num_heads_kv, head_dim).chunk(2, dim=-1)
    weight = torch.ones(head_dim // 2, dtype=torch.float32, device="npu")
    cache = _compute_cos_sin_cache(head_dim)

    positions = positions.flatten()
    cos_sin = cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    q1_out = torch.empty(
        (num_tokens, num_heads_q, head_dim // 2), dtype=torch.float32, device="npu"
    )
    q2_out = torch.empty(
        (num_tokens, num_heads_q, head_dim // 2), dtype=torch.float32, device="npu"
    )
    k1_out = torch.empty(
        (num_tokens, num_heads_kv, head_dim // 2), dtype=torch.float32, device="npu"
    )
    k2_out = torch.empty(
        (num_tokens, num_heads_kv, head_dim // 2), dtype=torch.float32, device="npu"
    )
    v_out = torch.empty(
        (num_tokens, num_heads_kv, head_dim), dtype=torch.float32, device="npu"
    )
    grid = (num_tokens // BLOCK_SIZE,)

    rms_norm_rope_kernel[grid](
        q1.contiguous(),
        q2.contiguous(),
        k1.contiguous(),
        k2.contiguous(),
        v.contiguous(),
        weight.contiguous(),
        cos.contiguous(),
        sin.contiguous(),
        q1_out,
        q2_out,
        k1_out,
        k2_out,
        v_out,
        q_size,
        kv_size,
        head_dim,
        BLOCK_SIZE,
        BLOCK_SIZE // 2,
    )
    q_out = torch.cat([q1_out, q2_out], dim=-1)
    k_out = torch.cat([k1_out, k2_out], dim=-1)
    q_out = q_out.view(num_tokens, q_size)
    k_out = k_out.view(num_tokens, kv_size)
    v_out = v_out.view(num_tokens, kv_size)
    return torch.cat([q_out, k_out, v_out], dim=-1)
