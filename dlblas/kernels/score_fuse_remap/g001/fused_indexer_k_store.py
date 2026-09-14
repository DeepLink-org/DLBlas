# SPDX-License-Identifier: Apache-2.0
# Focused extract of G001 fused operator from
#   vllm/models/deepseek_v32/common/kernels.py
# Kept:
#   helpers (_dummy, _fp8_ue8m0_quantize, _fp8_quant_and_cache_write)
#   _fused_norm_rope_kernel  (G001 uses pid==0 only: LN + RoPE + UE8M0 + cache)
#   fused_indexer_k_store    (launch grid = (1, num_tokens), pid 0 only)
# Full file also lives next to this extract as kernels.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton

# Cache of tiny 1-element dummy tensors (per device, dtype) reused by the
# has_indexer=False path so the indexer args don't allocate every call.
_DUMMY_CACHE: dict[tuple, torch.Tensor] = {}


def _dummy(shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (shape, dtype, device)
    t = _DUMMY_CACHE.get(key)
    if t is None:
        t = torch.empty(shape, dtype=dtype, device=device)
        _DUMMY_CACHE[key] = t
    return t


@triton.jit
def _rms_norm(x, w, eps, HIDDEN_SIZE: tl.constexpr):
    x = x.to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / HIDDEN_SIZE
    rrms = tl.rsqrt(mean_sq + eps)
    w = w.to(tl.float32)
    return (x * rrms) * w


@triton.jit
def _get_cos_sin(
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    pos,
    HALF_ROT_DIM: tl.constexpr,
):
    block = tl.arange(0, HALF_ROT_DIM)
    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block)
    cos = cos.to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block + HALF_ROT_DIM)
    sin = sin.to(tl.float32)
    return cos, sin


@triton.jit
def _fp8_ue8m0_quantize(vals):
    """Quantize float32 values to FP8 E4M3 with a ue8m0 (power-of-2) scale.

    Returns (fp8_vals, scale) so the caller can store them or reuse the scale.
    """
    vals = vals.to(tl.float32)
    amax = tl.max(tl.abs(vals))
    scale = tl.div_rn(tl.maximum(amax, 1e-4), 448.0)
    scale = tl.math.exp2(tl.math.ceil(tl.math.log2(scale)))
    fp8_vals = tl.div_rn(vals, scale).to(tl.float8e4nv)
    return fp8_vals, scale


@triton.jit
def _fp8_quant_and_cache_write(
    vals,
    mask,
    slot_idx,
    kv_cache_ptr,
    kv_cache_scale_ptr,
    cache_block_size,
    cache_stride,
    offsets,
    HEAD_DIM: tl.constexpr,
):
    k_fp8, scale = _fp8_ue8m0_quantize(vals)

    block_idx = slot_idx // cache_block_size
    block_offset = slot_idx % cache_block_size
    block_start = block_idx * cache_block_size * cache_stride

    tl.store(
        kv_cache_ptr + block_start + block_offset * HEAD_DIM + offsets,
        k_fp8,
        mask=mask,
    )
    scale_byte_off = block_start + cache_block_size * HEAD_DIM + block_offset * 4
    tl.store(kv_cache_scale_ptr + scale_byte_off // 4, scale)


@triton.jit
def _fused_norm_rope_kernel(
    pos_ptr,
    # Q RMS norm
    q_c_ptr,
    q_c_stride,
    q_rms_norm_w_ptr,
    q_rms_eps,
    q_c_out_ptr,
    q_c_out_stride,
    Q_DIM: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
    # KV RMS norm
    kv_ptr,
    kv_stride,
    kv_rms_norm_w_ptr,
    kv_rms_eps,
    KV_DIM: tl.constexpr,
    # KV RoPE
    kpe_ptr,
    kpe_stride,
    kpe_rope_cos_sin_cache_ptr,
    kpe_rope_cos_sin_cache_stride,
    KPE_HALF_ROT_DIM: tl.constexpr,
    # Index K layer norm
    index_k_ptr,
    index_k_stride,
    index_k_layer_norm_w_ptr,
    index_k_layer_norm_bias_ptr,
    index_k_layer_norm_eps,
    INDEX_K_DIM: tl.constexpr,
    INDEX_K_BLOCK_SIZE: tl.constexpr,
    # Index K RoPE
    index_k_rope_cos_sin_cache_ptr,
    index_k_rope_cos_sin_cache_stride,
    INDEX_K_HALF_ROT_DIM: tl.constexpr,
    # Cache params (shared by indexer K and MLA)
    slot_mapping_ptr,
    # Index K FP8 cache
    indexer_cache_ptr,
    indexer_cache_scale_ptr,
    indexer_cache_block_size,
    indexer_cache_stride,
    # MLA KV cache (concat kv_c_normed + k_pe_roped, uses slot_mapping_ptr)
    mla_cache_ptr,
    mla_cache_block_stride,
    mla_cache_entry_stride,
    MLA_CACHE_FP8: tl.constexpr,
    mla_cache_scale_ptr,
    # fp8_ds_mla cache views (block-scaled fp8 NoPE + unquantized bf16 RoPE).
    # mla_cache_ptr is the fp8 (1-byte) view, so the block/entry strides above
    # are byte offsets; these two share the same buffer as fp32 / bf16 views.
    mla_cache_ds_scale_ptr,
    mla_cache_ds_rope_ptr,
    MLA_CACHE_DS_MLA: tl.constexpr,
    MLA_NUM_TILES: tl.constexpr,
    MLA_TILE_DIM: tl.constexpr,
    # Top k indices
    topk_indices_ptr,
    topk_indices_stride,
    TOPK: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
    HAS_INDEXER: tl.constexpr,
    INDEX_ROPE_INTERLEAVE: tl.constexpr,
):
    pid = tl.program_id(0)
    tok_idx = tl.program_id(1)
    if pid == 3:
        if not HAS_INDEXER:
            # Shared layer: reuse the previous indexer layer's top-k; do not
            # clear the buffer.
            return
        # Fill top k indices buffer with -1
        for i in range(0, TOPK, TOPK_BLOCK_SIZE):
            offset = i + tl.arange(0, TOPK_BLOCK_SIZE)
            mask = offset < TOPK
            tl.store(
                topk_indices_ptr + tok_idx * topk_indices_stride + offset,
                -1,
                mask=mask,
            )
        return

    if slot_mapping_ptr is None:
        # Memory profiling run.
        return
    slot_idx = tl.load(slot_mapping_ptr + tok_idx)
    if slot_idx < 0:
        # Padding
        return

    if pid == 2:
        # Q RMS norm
        q_block = tl.arange(0, Q_BLOCK_SIZE)
        q_mask = q_block < Q_DIM
        q_c = tl.load(q_c_ptr + tok_idx * q_c_stride + q_block, mask=q_mask, other=0.0)
        q_c_rms_w = tl.load(q_rms_norm_w_ptr + q_block, mask=q_mask)
        q_c = _rms_norm(q_c, q_c_rms_w, q_rms_eps, Q_DIM)
        tl.store(q_c_out_ptr + tok_idx * q_c_out_stride + q_block, q_c, mask=q_mask)
    elif pid == 1:
        # KV RMS Norm + KV RoPE + MLA concat_and_cache.
        # Merged so the normed kv_c and RoPE'd k_pe can be written
        # to the MLA KV cache directly without a separate kernel.

        # KV RMS Norm (result stays in registers for MLA cache write)
        kv_block = tl.arange(0, KV_DIM)
        kv_c = tl.load(kv_ptr + tok_idx * kv_stride + kv_block)
        kv_c_rms_w = tl.load(kv_rms_norm_w_ptr + kv_block)
        kv_c = _rms_norm(kv_c, kv_c_rms_w, kv_rms_eps, KV_DIM)

        # KV RoPE (interleaved) on k_pe — in registers only.
        # k_pe is not needed after the cache write (MLA decode reads
        # from kv_cache), so we skip writing back to kpe_ptr.
        pos = tl.load(pos_ptr + tok_idx)
        cos, sin = _get_cos_sin(
            kpe_rope_cos_sin_cache_ptr,
            kpe_rope_cos_sin_cache_stride,
            pos,
            KPE_HALF_ROT_DIM,
        )
        dim_off = tl.arange(0, KPE_HALF_ROT_DIM)
        kpe_base = kpe_ptr + tok_idx * kpe_stride
        x1 = tl.load(kpe_base + dim_off * 2).to(tl.float32)
        x2 = tl.load(kpe_base + dim_off * 2 + 1).to(tl.float32)
        r1 = x1 * cos - x2 * sin
        r2 = x2 * cos + x1 * sin

        # MLA concat_and_cache: write [kv_c_normed, k_pe_roped] to cache.
        if mla_cache_entry_stride == 0:
            return

        mla_block_size = mla_cache_block_stride // mla_cache_entry_stride
        mla_block_idx = slot_idx // mla_block_size
        mla_block_off = slot_idx % mla_block_size

        if MLA_CACHE_DS_MLA:
            # fp8_ds_mla layout (DeepSeek-V3.2, KV_DIM == 512): per-128-element
            # tile of the NoPE is dynamically quantized to fp8 with its own
            # float32 scale; the RoPE tail is stored unquantized in bf16.
            #   bytes [0, KV_DIM)            : KV_DIM fp8 NoPE values
            #   bytes [KV_DIM, KV_DIM + 16)  : MLA_NUM_TILES float32 scales
            #   bytes [KV_DIM + 16, ...)     : 2 * KPE_HALF_ROT_DIM bf16 RoPE
            # mla_cache_block_stride / mla_cache_entry_stride are byte strides
            # (mla_cache_ptr is the 1-byte fp8 view of the uint8 cache).
            byte_base = (
                mla_block_idx * mla_cache_block_stride
                + mla_block_off * mla_cache_entry_stride
            )
            kv_2d = tl.reshape(kv_c, (MLA_NUM_TILES, MLA_TILE_DIM))
            tile_amax = tl.max(tl.abs(kv_2d), axis=1, keep_dims=True)
            # scale = amax / 448 (fp8 e4m3 max), matching the reference
            # concat_and_cache_ds_mla kernel; floored to FLT_MIN.
            tile_scale = tl.maximum(tile_amax * (1.0 / 448.0), 1.1754944e-38)
            kv_c_fp8 = tl.reshape((kv_2d / tile_scale).to(tl.float8e4nv), (KV_DIM,))
            tl.store(mla_cache_ptr + byte_base + kv_block, kv_c_fp8)
            tile_off = tl.arange(0, MLA_NUM_TILES)
            tl.store(
                mla_cache_ds_scale_ptr + byte_base // 4 + KV_DIM // 4 + tile_off,
                tl.reshape(tile_scale, (MLA_NUM_TILES,)),
            )
            rope_dst = mla_cache_ds_rope_ptr + byte_base // 2 + (KV_DIM // 2 + 8)
            tl.store(rope_dst + dim_off * 2, r1.to(tl.bfloat16))
            tl.store(rope_dst + dim_off * 2 + 1, r2.to(tl.bfloat16))
            return

        dst = (
            mla_cache_ptr
            + mla_block_idx * mla_cache_block_stride
            + mla_block_off * mla_cache_entry_stride
        )
        # kv_c_normed (KV_DIM elements)
        if MLA_CACHE_FP8:
            scale = tl.load(mla_cache_scale_ptr)
            kv_c_fp8 = (kv_c.to(tl.float32) / scale).to(tl.float8e4nv)
            tl.store(dst + kv_block, kv_c_fp8)
        else:
            tl.store(dst + kv_block, kv_c)
        # k_pe_roped (from registers, interleaved layout)
        if MLA_CACHE_FP8:
            tl.store(dst + KV_DIM + dim_off * 2, (r1 / scale).to(tl.float8e4nv))
            tl.store(dst + KV_DIM + dim_off * 2 + 1, (r2 / scale).to(tl.float8e4nv))
        else:
            tl.store(dst + KV_DIM + dim_off * 2, r1)
            tl.store(dst + KV_DIM + dim_off * 2 + 1, r2)
    elif pid == 0:
        if not HAS_INDEXER:
            # Shared layer: no indexer K to process.
            return
        # Fused: Index K LayerNorm + RoPE + FP8 quant + cache write.
        # Eliminates the separate indexer_k_quant_and_cache kernel launch.

        index_k_block = tl.arange(0, INDEX_K_BLOCK_SIZE)
        index_k_mask = index_k_block < INDEX_K_DIM
        index_k = tl.load(
            index_k_ptr + tok_idx * index_k_stride + index_k_block,
            mask=index_k_mask,
            other=0.0,
        ).to(tl.float32)
        index_k_w = tl.load(
            index_k_layer_norm_w_ptr + index_k_block, mask=index_k_mask
        ).to(tl.float32)
        index_k_b = tl.load(
            index_k_layer_norm_bias_ptr + index_k_block, mask=index_k_mask
        ).to(tl.float32)

        # 1. LayerNorm. Keep (mean, rstd) so the RoPE rotation partner can be
        #    re-normalized in registers below, avoiding a global scratch buffer.
        mean = tl.sum(index_k, axis=0) / INDEX_K_DIM
        diff = tl.where(index_k_mask, index_k - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / INDEX_K_DIM
        rstd = tl.rsqrt(var + index_k_layer_norm_eps)
        normed = (index_k - mean) * rstd * index_k_w + index_k_b

        # 2. RoPE on the rotation region. Supports both interleaved (adjacent
        #    pairs, e.g. GLM-5.2) and NeoX (split-half, e.g. DeepSeek-V3.2). The
        #    rotation partner is gathered from the read-only inputs and
        #    re-normalized with the same (mean, rstd) — no scratch, no atomics.
        pos = tl.load(pos_ptr + tok_idx)
        in_rope = index_k_block < 2 * INDEX_K_HALF_ROT_DIM
        if INDEX_ROPE_INTERLEAVE:
            # pair i = block // 2; partner = block ^ 1; even -> -sin, odd -> +sin.
            cos_idx = index_k_block // 2
            partner_offs = tl.where(in_rope, index_k_block ^ 1, index_k_block)
            sign = tl.where(index_k_block % 2 == 0, -1.0, 1.0)
        else:
            # NeoX: pair across halves; partner = block ^ HALF.
            cos_idx = index_k_block % INDEX_K_HALF_ROT_DIM
            partner_offs = tl.where(
                in_rope, index_k_block ^ INDEX_K_HALF_ROT_DIM, index_k_block
            )
            sign = tl.where(index_k_block < INDEX_K_HALF_ROT_DIM, -1.0, 1.0)
        cos_full = tl.load(
            index_k_rope_cos_sin_cache_ptr
            + pos * index_k_rope_cos_sin_cache_stride
            + cos_idx,
            mask=in_rope,
            other=1.0,
        ).to(tl.float32)
        sin_full = tl.load(
            index_k_rope_cos_sin_cache_ptr
            + pos * index_k_rope_cos_sin_cache_stride
            + INDEX_K_HALF_ROT_DIM
            + cos_idx,
            mask=in_rope,
            other=0.0,
        ).to(tl.float32)
        # normed[partner_offs] == (raw_partner - mean) * rstd * w_partner +
        # b_partner: gather the raw partner and its norm affine (read-only
        # loads), then apply the same per-token mean/rstd.
        raw_partner = tl.load(
            index_k_ptr + tok_idx * index_k_stride + partner_offs,
            mask=index_k_mask,
            other=0.0,
        ).to(tl.float32)
        w_partner = tl.load(
            index_k_layer_norm_w_ptr + partner_offs, mask=index_k_mask
        ).to(tl.float32)
        b_partner = tl.load(
            index_k_layer_norm_bias_ptr + partner_offs, mask=index_k_mask
        ).to(tl.float32)
        normed_partner = (raw_partner - mean) * rstd * w_partner + b_partner
        roped = normed * cos_full + sign * normed_partner * sin_full
        result = tl.where(in_rope, roped, normed)

        # 3. FP8 quantize + cache write from registers.
        #    No need to write back to index_k_ptr — the only consumer
        #    (sparse_attn_indexer) reads from the cache, not index_k.
        _fp8_quant_and_cache_write(
            result,
            index_k_mask,
            slot_idx,
            indexer_cache_ptr,
            indexer_cache_scale_ptr,
            indexer_cache_block_size,
            indexer_cache_stride,
            index_k_block,
            INDEX_K_DIM,
        )


def fused_indexer_k_store(
    positions: torch.Tensor,
    index_k: torch.Tensor,
    index_k_layer_norm_w: torch.Tensor,
    index_k_layer_norm_bias: torch.Tensor,
    index_k_layer_norm_eps: float,
    index_k_rope_cos_sin_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    indexer_k_cache: torch.Tensor,
    index_rope_interleave: bool = True,
) -> None:
xin    """Indexer-K store fuse used by ``sparse_attn_indexer`` (GLM-5.2 DSA G001).

    Runs only pid 0 of ``_fused_norm_rope_kernel``: LayerNorm + RoPE (GLM
    interleaved or DeepSeek NeoX) + UE8M0 FP8 quant + indexer cache write.
    Q-RMS / MLA / top-k fill pids are not launched.

    ``index_k`` is the raw GEMM output (before LN). The caller must skip the
    unfused ``indexer_k_quant_and_cache`` insert for the same tokens.
    """
    assert positions.ndim == 1
    assert index_k.ndim == 2
    assert slot_mapping.ndim == 1
    assert indexer_k_cache.ndim == 3
    num_tokens = positions.shape[0]
    assert index_k.shape[0] == num_tokens
    assert slot_mapping.shape[0] == num_tokens
    device = positions.device
    dtype = index_k.dtype
    index_k_dim = index_k.shape[-1]

    idx_cache_scale_view = indexer_k_cache.view(torch.uint8).view(torch.float32)
    idx_cache_block_size = indexer_k_cache.shape[1]
    idx_cache_stride = indexer_k_cache.shape[2]
    cache = indexer_k_cache
    if cache.dtype == torch.uint8:
        cache = cache.view(torch.float8_e4m3fn)

    # Unused pids (Q / MLA / topk fill) still need typed pointers.
    q_dummy = _dummy((1, 128), dtype, device)
    kv_dummy = _dummy((1, 128), dtype, device)
    kpe_dummy = _dummy((1, 64), dtype, device)
    w_dummy = _dummy((128,), torch.float32, device)
    topk_dummy = _dummy((1, 8), torch.int32, device)
    mla_dummy = _dummy((1,), torch.bfloat16, device)
    mla_k_scale = _dummy((1,), torch.float32, device)
    empty_f32 = _dummy((1,), torch.float32, device)
    empty_bf16 = _dummy((1,), torch.bfloat16, device)

    _fused_norm_rope_kernel[(1, num_tokens)](
        positions,
        q_dummy,
        q_dummy.stride(0),
        w_dummy,
        1e-6,
        q_dummy,
        q_dummy.stride(0),
        128,
        128,
        kv_dummy,
        kv_dummy.stride(0),
        w_dummy,
        1e-6,
        128,
        kpe_dummy,
        kpe_dummy.stride(0),
        index_k_rope_cos_sin_cache,
        index_k_rope_cos_sin_cache.stride(0),
        index_k_rope_cos_sin_cache.shape[-1] // 2,
        index_k,
        index_k.stride(0),
        index_k_layer_norm_w,
        index_k_layer_norm_bias,
        index_k_layer_norm_eps,
        index_k_dim,
        triton.next_power_of_2(index_k_dim),
        index_k_rope_cos_sin_cache,
        index_k_rope_cos_sin_cache.stride(0),
        index_k_rope_cos_sin_cache.shape[-1] // 2,
        slot_mapping,
        cache,
        idx_cache_scale_view,
        idx_cache_block_size,
        idx_cache_stride,
        mla_dummy,
        0,
        0,
        False,
        mla_k_scale,
        empty_f32,
        empty_bf16,
        False,
        1,
        1,
        topk_dummy,
        topk_dummy.stride(0),
        8,
        TOPK_BLOCK_SIZE=1024,
        HAS_INDEXER=True,
        INDEX_ROPE_INTERLEAVE=index_rope_interleave,
    )

