"""Triton implementation of Task05: MusicFlamingoRotaryEmbedding."""

import math
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl
from cuda.bindings import runtime as _cuda_runtime


_CHIP_CONFIGS = {
    "auto": (128, 4),
    "ascend_a2": (128, 4),
    # Warp size is 64 on C500; a single native wave owns one complete tile.
    "metax_c500": (64, 1),
    "iluvatar_bi150": (128, 4),
    "enflame_s60": (128, 4),
    "thead_810e": (64, 2),
    "cambricon_mlu590": (64, 2),
    "hygon_bw1000": (256, 4),
    "mthreads_ph100": (256, 4),
    "kunlun_p800": (64, 2),
    "nvidia_h200": (128, 4),
}

# (tokens per program, warps) for the A2-only two-dimensional kernel.
_A2_TOKEN_TILE_CONFIG = (4, 4)
_A2_FLAT_CONFIGS = {
    "64x1": (64, 1),
    "64x2": (64, 2),
    "64x4": (64, 4),
    "128x2": (128, 2),
    "128x4": (128, 4),
    "256x2": (256, 2),
    "256x4": (256, 4),
    "512x4": (512, 4),
    "1024x4": (1024, 4),
}
_A2_PERSISTENT_PROGRAMS = {
    "24": 24,
    "32": 32,
    "48": 48,
    "64": 64,
    "96": 96,
}


def _get_launch_config():
    chip = os.environ.get("KERNELSWIFT_CHIP", "nvidia_h200").strip().lower()
    if chip == "ascend_a2":
        key = os.environ.get("KERNELSWIFT_TASK05_FLAT_CONFIG", "64x2")
        return _A2_FLAT_CONFIGS.get(key, _A2_FLAT_CONFIGS["64x2"])
    if chip == "metax_c500":
        # Keep the C500 sweep on the portable flat Triton kernel.  The named
        # choices change only flat chunk size and launch width.
        key = os.environ.get("KERNELSWIFT_TASK05_FLAT_CONFIG", "64x1")
        variants = {
            "64x1": _A2_FLAT_CONFIGS["64x1"],
            "64x2": _A2_FLAT_CONFIGS["64x2"],
            "128x2": _A2_FLAT_CONFIGS["128x2"],
            "256x2": _A2_FLAT_CONFIGS["256x2"],
            "256x4": _A2_FLAT_CONFIGS["256x4"],
            "512x4": _A2_FLAT_CONFIGS["512x4"],
        }
        return variants.get(key, _CHIP_CONFIGS["metax_c500"])
    if chip == "nvidia_h200":
        # Every named choice drives the same portable flat kernel.  Keeping the
        # accepted set explicit makes the remote H200 sweep reproducible.  The
        # measured no-selector token owner uses one 32-lane warp.  Explicit
        # MODE=flat with no geometry override remains a one-variable rollback
        # to the original fused 128x4 launch; the third-round split-trig path
        # is recovered by explicitly selecting MODE=split_trig/64x2.
        key = os.environ.get("KERNELSWIFT_TASK05_FLAT_CONFIG", "").strip().lower()
        if key in ("", "default"):
            mode = os.environ.get(
                "KERNELSWIFT_TASK05_MODE", "split_token"
            ).strip().lower()
            if mode == "flat":
                return _CHIP_CONFIGS["nvidia_h200"]
            if mode == "split_trig":
                return _A2_FLAT_CONFIGS["64x2"]
            return _A2_FLAT_CONFIGS["64x1"]
        if key not in _A2_FLAT_CONFIGS:
            raise ValueError(f"unsupported Task05 H200 flat config: {key}")
        return _A2_FLAT_CONFIGS[key]
    return _CHIP_CONFIGS.get(chip, _CHIP_CONFIGS["auto"])


@triton.jit
def _music_flamingo_rotary_kernel(
    timestamps_ptr,
    inv_freq_ptr,
    position_angles_ptr,
    cos_ptr,
    sin_ptr,
    timestamp_stride_b: tl.constexpr,
    timestamp_stride_s: tl.constexpr,
    position_stride_s: tl.constexpr,
    output_stride_b: tl.constexpr,
    output_stride_s: tl.constexpr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    rotary_dim: tl.constexpr,
    max_seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_POLY_TRIG: tl.constexpr,
    FAST_POLY_TRIG: tl.constexpr,
    DERIVE_SIN_FROM_COS: tl.constexpr,
):
    """Compute each repeated frequency once and write its two equal lanes."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_dim = 2 * rotary_dim
    # Each half contains rotary_dim / 2 unique frequencies because the
    # reference uses repeat_interleave(2).  Across batch+time halves there are
    # rotary_dim unique phases per token, half as many as output elements.
    total = batch_size * seq_len * rotary_dim
    mask = offsets < total

    unique_feature = offsets % rotary_dim
    token = (offsets // rotary_dim) % seq_len
    batch = offsets // (seq_len * rotary_dim)

    pair_count = rotary_dim // 2
    is_time = unique_feature >= pair_count
    pair = unique_feature % pair_count
    inv_freq = tl.load(inv_freq_ptr + pair, mask=mask, other=0.0).to(tl.float32)
    batch_freq = (batch.to(tl.float32) / max_seq_len) * inv_freq
    time_freq = tl.load(
        position_angles_ptr + token * position_stride_s + pair * 2,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    freq = tl.where(is_time, time_freq, batch_freq)

    timestamp = tl.load(
        timestamps_ptr + batch * timestamp_stride_b + token * timestamp_stride_s,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    phase = freq * (-6.283185307179586 * timestamp)

    output_feature = pair * 2 + tl.where(is_time, rotary_dim, 0)
    output_offsets = batch * output_stride_b + token * output_stride_s + output_feature
    if USE_POLY_TRIG:
        # The official timestamps keep phase in [-4.8, 0].  Reduce to the
        # nearest pi/2 quadrant and evaluate both functions with FP32 FMAs.
        # On Hopper this trades two SFU operations for the much wider FP32
        # pipelines while retaining <1e-5 approximation error on this range.
        quadrant_f = tl.floor(phase * 0.6366197723675814 + 0.5)
        reduced = phase - quadrant_f * 1.5707963267948966
        z = reduced * reduced
        if FAST_POLY_TRIG:
            # On |reduced| <= pi/4 these degree-5/4 Taylor polynomials have
            # worst-case truncation below 3.3e-4, still far inside the task's
            # 1e-2 tolerance while saving two dependent FMAs.
            sin_reduced = reduced * (
                1.0 + z * (-0.16666666666666666 + z * 0.008333333333333333)
            )
            cos_reduced = 1.0 + z * (-0.5 + z * 0.041666666666666664)
        else:
            sin_reduced = reduced * (
                1.0
                + z
                * (
                    -0.16666666666666666
                    + z
                    * (
                        0.008333333333333333
                        + z * -0.0001984126984126984
                    )
                )
            )
            cos_reduced = 1.0 + z * (
                -0.5
                + z * (0.041666666666666664 + z * -0.001388888888888889)
            )
        quadrant = quadrant_f.to(tl.int32) & 3
        sin_value = tl.where(
            quadrant == 0,
            sin_reduced,
            tl.where(
                quadrant == 1,
                cos_reduced,
                tl.where(quadrant == 2, -sin_reduced, -cos_reduced),
            ),
        )
        cos_value = tl.where(
            quadrant == 0,
            cos_reduced,
            tl.where(
                quadrant == 1,
                -sin_reduced,
                tl.where(quadrant == 2, -cos_reduced, sin_reduced),
            ),
        )
    elif DERIVE_SIN_FROM_COS:
        # Hopper candidate: issue only cosine and recover the sine magnitude
        # from sin^2 + cos^2 = 1.  The sign is exact for every finite phase:
        # floor(phase / pi) is odd precisely on negative-sine half turns.
        # Clamping handles the tiny overshoot permitted by cos.approx.f32.
        cos_value = tl.cos(phase)
        sin_sq = tl.maximum(0.0, 1.0 - cos_value * cos_value)
        sin_magnitude = tl.sqrt(sin_sq)
        half_turn = tl.floor(phase * 0.3183098861837907).to(tl.int32)
        sin_value = tl.where((half_turn & 1) != 0, -sin_magnitude, sin_magnitude)
    else:
        cos_value = tl.cos(phase)
        sin_value = tl.sin(phase)
    tl.store(cos_ptr + output_offsets, cos_value, mask=mask)
    tl.store(cos_ptr + output_offsets + 1, cos_value, mask=mask)
    tl.store(sin_ptr + output_offsets, sin_value, mask=mask)
    tl.store(sin_ptr + output_offsets + 1, sin_value, mask=mask)


@triton.jit
def _music_flamingo_rotary_split_trig_kernel(
    timestamps_ptr,
    inv_freq_ptr,
    position_angles_ptr,
    cos_ptr,
    sin_ptr,
    timestamp_stride_b: tl.constexpr,
    timestamp_stride_s: tl.constexpr,
    position_stride_s: tl.constexpr,
    output_stride_b: tl.constexpr,
    output_stride_s: tl.constexpr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    rotary_dim: tl.constexpr,
    max_seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One launch, separate cosine/sine CTAs to remove register spilling.

    The normal fused program retains both transcendental results and both
    output address streams at once; H200 metadata reports eight spill slots.
    Grid axis one assigns each CTA exactly one intrinsic and one destination.
    Frequency/timestamp loads are repeated, but they are tiny relative to the
    spill traffic and the final four output stores of the fused program.
    """
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_kind = tl.program_id(1)
    total = batch_size * seq_len * rotary_dim
    mask = offsets < total
    unique_feature = offsets % rotary_dim
    token = (offsets // rotary_dim) % seq_len
    batch = offsets // (seq_len * rotary_dim)
    pair_count = rotary_dim // 2
    is_time = unique_feature >= pair_count
    pair = unique_feature % pair_count
    inv_freq = tl.load(inv_freq_ptr + pair, mask=mask, other=0.0).to(tl.float32)
    batch_freq = (batch.to(tl.float32) / max_seq_len) * inv_freq
    time_freq = tl.load(
        position_angles_ptr + token * position_stride_s + pair * 2,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    freq = tl.where(is_time, time_freq, batch_freq)
    timestamp = tl.load(
        timestamps_ptr + batch * timestamp_stride_b + token * timestamp_stride_s,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    phase = freq * (-6.283185307179586 * timestamp)
    output_feature = pair * 2 + tl.where(is_time, rotary_dim, 0)
    output_offsets = batch * output_stride_b + token * output_stride_s + output_feature
    if output_kind == 0:
        value = tl.cos(phase)
        tl.store(cos_ptr + output_offsets, value, mask=mask)
        tl.store(cos_ptr + output_offsets + 1, value, mask=mask)
    else:
        value = tl.sin(phase)
        tl.store(sin_ptr + output_offsets, value, mask=mask)
        tl.store(sin_ptr + output_offsets + 1, value, mask=mask)


@triton.jit
def _h200_exact_sincos(value):
    """Emit the same PTX approximate sin/cos instructions as tl.sin/tl.cos."""
    return tl.inline_asm_elementwise(
        "sin.approx.f32 $0, $2; cos.approx.f32 $1, $2;",
        "=f,=f,f",
        [value],
        dtype=(tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _music_flamingo_rotary_split_token_kernel(
    timestamps_ptr,
    inv_freq_ptr,
    position_angles_ptr,
    cos_ptr,
    sin_ptr,
):
    """Official-shape token owner with one trig kind per CTA.

    Grid axis zero is exactly the 128 ``(batch, token)`` rows and axis one is
    sin/cos.  Thirty-two lanes own frequency pairs and materialize both the
    batch and time halves.  This removes every per-lane divide/modulo from the
    flat mapping and loads each timestamp/inverse/time frequency only once.
    """
    global_token = tl.program_id(0)
    output_kind = tl.program_id(1)
    pair = tl.arange(0, 32)
    batch = global_token // 32
    token = global_token - batch * 32
    timestamp = tl.load(timestamps_ptr + global_token).to(tl.float32)
    inv_freq = tl.load(inv_freq_ptr + pair).to(tl.float32)
    time_freq = tl.load(
        position_angles_ptr + token * 64 + pair * 2
    ).to(tl.float32)
    common_angle = -6.283185307179586 * timestamp
    batch_phase = (batch.to(tl.float32) / 256.0 * inv_freq) * common_angle
    time_phase = time_freq * common_angle
    output_base = global_token * 128 + pair * 2
    if output_kind == 0:
        batch_value = tl.cos(batch_phase)
        time_value = tl.cos(time_phase)
        tl.store(cos_ptr + output_base, batch_value)
        tl.store(cos_ptr + output_base + 1, batch_value)
        tl.store(cos_ptr + output_base + 64, time_value)
        tl.store(cos_ptr + output_base + 65, time_value)
    else:
        batch_value = tl.sin(batch_phase)
        time_value = tl.sin(time_phase)
        tl.store(sin_ptr + output_base, batch_value)
        tl.store(sin_ptr + output_base + 1, batch_value)
        tl.store(sin_ptr + output_base + 64, time_value)
        tl.store(sin_ptr + output_base + 65, time_value)


@triton.jit
def _music_flamingo_rotary_token_sincos_kernel(
    timestamps_ptr,
    inv_freq_ptr,
    position_angles_ptr,
    cos_ptr,
    sin_ptr,
    INLINE_ASM: tl.constexpr,
):
    """One official-shape CTA shares phase construction across sin and cos."""
    global_token = tl.program_id(0)
    pair = tl.arange(0, 32)
    batch = global_token // 32
    token = global_token - batch * 32
    timestamp = tl.load(timestamps_ptr + global_token).to(tl.float32)
    inv_freq = tl.load(inv_freq_ptr + pair).to(tl.float32)
    time_freq = tl.load(
        position_angles_ptr + token * 64 + pair * 2
    ).to(tl.float32)
    common_angle = -6.283185307179586 * timestamp
    batch_phase = (batch.to(tl.float32) / 256.0 * inv_freq) * common_angle
    time_phase = time_freq * common_angle
    if INLINE_ASM:
        batch_sin, batch_cos = _h200_exact_sincos(batch_phase)
        time_sin, time_cos = _h200_exact_sincos(time_phase)
    else:
        batch_sin = tl.sin(batch_phase)
        batch_cos = tl.cos(batch_phase)
        time_sin = tl.sin(time_phase)
        time_cos = tl.cos(time_phase)
    output_base = global_token * 128 + pair * 2
    tl.store(cos_ptr + output_base, batch_cos)
    tl.store(cos_ptr + output_base + 1, batch_cos)
    tl.store(cos_ptr + output_base + 64, time_cos)
    tl.store(cos_ptr + output_base + 65, time_cos)
    tl.store(sin_ptr + output_base, batch_sin)
    tl.store(sin_ptr + output_base + 1, batch_sin)
    tl.store(sin_ptr + output_base + 64, time_sin)
    tl.store(sin_ptr + output_base + 65, time_sin)


@triton.jit
def _a2_music_flamingo_persistent_kernel(
    timestamps_ptr,
    inv_freq_ptr,
    position_angles_ptr,
    cos_ptr,
    sin_ptr,
    timestamp_stride_b: tl.constexpr,
    timestamp_stride_s: tl.constexpr,
    position_stride_s: tl.constexpr,
    output_stride_b: tl.constexpr,
    output_stride_s: tl.constexpr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    rotary_dim: tl.constexpr,
    max_seq_len: tl.constexpr,
    TOTAL: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """A2 persistent form of the proven one-dimensional vector kernel."""
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    lane = tl.arange(0, BLOCK_SIZE)
    output_dim = 2 * rotary_dim
    pair_count = rotary_dim // 2

    for chunk in range(pid, NUM_CHUNKS, programs):
        offsets = chunk * BLOCK_SIZE + lane
        mask = offsets < TOTAL
        unique_feature = offsets % rotary_dim
        token = (offsets // rotary_dim) % seq_len
        batch = offsets // (seq_len * rotary_dim)

        is_time = unique_feature >= pair_count
        pair = unique_feature % pair_count
        inv_freq = tl.load(
            inv_freq_ptr + pair, mask=mask, other=0.0
        ).to(tl.float32)
        batch_freq = (batch.to(tl.float32) / max_seq_len) * inv_freq
        time_freq = tl.load(
            position_angles_ptr + token * position_stride_s + pair * 2,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        freq = tl.where(is_time, time_freq, batch_freq)
        timestamp = tl.load(
            timestamps_ptr
            + batch * timestamp_stride_b
            + token * timestamp_stride_s,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        phase = freq * (-6.283185307179586 * timestamp)

        output_feature = pair * 2 + tl.where(is_time, rotary_dim, 0)
        output_offsets = (
            batch * output_stride_b
            + token * output_stride_s
            + output_feature
        )
        cos_value = tl.cos(phase)
        sin_value = tl.sin(phase)
        tl.store(cos_ptr + output_offsets, cos_value, mask=mask)
        tl.store(cos_ptr + output_offsets + 1, cos_value, mask=mask)
        tl.store(sin_ptr + output_offsets, sin_value, mask=mask)
        tl.store(sin_ptr + output_offsets + 1, sin_value, mask=mask)


@triton.jit
def _a2_music_flamingo_token_row_kernel(
    timestamps_ptr,
    inv_freq_ptr,
    position_angles_ptr,
    cos_ptr,
    sin_ptr,
    timestamp_stride_b: tl.constexpr,
    timestamp_stride_s: tl.constexpr,
    position_stride_s: tl.constexpr,
    output_stride_b: tl.constexpr,
    output_stride_s: tl.constexpr,
    seq_len: tl.constexpr,
    rotary_dim: tl.constexpr,
    max_seq_len: tl.constexpr,
    TOTAL_TOKENS: tl.constexpr,
    PAIR_BLOCK: tl.constexpr,
):
    """A2 row owner with contiguous stores for repeat_interleave(2).

    Each logical tile is exactly one token.  The phase calculation stays
    one-dimensional (unlike the pathological multi-token 2-D lowering), while
    adjacent equal lanes are materialised in UB and written as contiguous
    64-element halves rather than four stride-2 GM streams.
    """
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    pair = tl.arange(0, PAIR_BLOCK)
    out_lane = tl.arange(0, 2 * PAIR_BLOCK)

    for global_token in range(pid, TOTAL_TOKENS, programs):
        batch = global_token // seq_len
        token = global_token - batch * seq_len
        timestamp = tl.load(
            timestamps_ptr
            + batch * timestamp_stride_b
            + token * timestamp_stride_s
        ).to(tl.float32)
        inv_freq = tl.load(inv_freq_ptr + pair).to(tl.float32)
        time_freq = tl.load(
            position_angles_ptr + token * position_stride_s + 2 * pair
        ).to(tl.float32)
        angle = -6.283185307179586 * timestamp
        batch_phase = (
            batch.to(tl.float32) / max_seq_len * inv_freq * angle
        )
        time_phase = time_freq * angle

        batch_cos = tl.cos(batch_phase)
        batch_sin = tl.sin(batch_phase)
        time_cos = tl.cos(time_phase)
        time_sin = tl.sin(time_phase)
        # ``repeat_interleave(2)`` is formed in UB, allowing contiguous GM
        # stores and eliminating the stride-2 MTE3 pattern of the flat kernel.
        batch_cos_pairs = tl.reshape(
            tl.broadcast_to(batch_cos[:, None], (PAIR_BLOCK, 2)),
            (2 * PAIR_BLOCK,),
        )
        batch_sin_pairs = tl.reshape(
            tl.broadcast_to(batch_sin[:, None], (PAIR_BLOCK, 2)),
            (2 * PAIR_BLOCK,),
        )
        time_cos_pairs = tl.reshape(
            tl.broadcast_to(time_cos[:, None], (PAIR_BLOCK, 2)),
            (2 * PAIR_BLOCK,),
        )
        time_sin_pairs = tl.reshape(
            tl.broadcast_to(time_sin[:, None], (PAIR_BLOCK, 2)),
            (2 * PAIR_BLOCK,),
        )

        output_base = (
            batch * output_stride_b + token * output_stride_s
        )
        tl.store(cos_ptr + output_base + out_lane, batch_cos_pairs)
        tl.store(sin_ptr + output_base + out_lane, batch_sin_pairs)
        time_base = output_base + rotary_dim
        tl.store(cos_ptr + time_base + out_lane, time_cos_pairs)
        tl.store(sin_ptr + time_base + out_lane, time_sin_pairs)


@triton.jit
def _a2_music_flamingo_token_tile_kernel(
    timestamps_ptr,
    inv_freq_ptr,
    position_angles_ptr,
    cos_ptr,
    sin_ptr,
    timestamp_stride_b: tl.constexpr,
    timestamp_stride_s: tl.constexpr,
    position_stride_s: tl.constexpr,
    output_stride_b: tl.constexpr,
    output_stride_s: tl.constexpr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    rotary_dim: tl.constexpr,
    max_seq_len: tl.constexpr,
    PAIR_COUNT: tl.constexpr,
    PAIR_BLOCK: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    """A2 token-frequency tile with broadcasted timestamp/frequency loads.

    The flat portable kernel loads the same timestamp once per feature and
    launches 64 tiny programs for the official shape.  This two-dimensional
    tile loads four timestamps and 32 inverse frequencies, broadcasts them in
    registers, and emits both the batch and time halves in 32 programs.  That
    is enough parallel work to occupy the A2's compute cores without the 64 tiny
    programs and redundant scalar loads of the flat layout.
    """
    global_token = (
        tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    )
    pair = tl.arange(0, PAIR_BLOCK)
    total_tokens = batch_size * seq_len
    token_mask = global_token < total_tokens
    pair_mask = pair < PAIR_COUNT

    batch = global_token // seq_len
    token = global_token % seq_len
    timestamp = tl.load(
        timestamps_ptr
        + batch * timestamp_stride_b
        + token * timestamp_stride_s,
        mask=token_mask,
        other=0.0,
    ).to(tl.float32)
    inv_freq = tl.load(
        inv_freq_ptr + pair,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    time_freq = tl.load(
        position_angles_ptr
        + token[:, None] * position_stride_s
        + (2 * pair[None, :]),
        mask=token_mask[:, None] & pair_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    common_angle = -6.283185307179586 * timestamp[:, None]
    batch_freq = (
        batch[:, None].to(tl.float32) / max_seq_len
    ) * inv_freq[None, :]
    batch_phase = batch_freq * common_angle
    time_phase = time_freq * common_angle

    output_base = (
        batch[:, None] * output_stride_b
        + token[:, None] * output_stride_s
        + 2 * pair[None, :]
    )
    output_mask = token_mask[:, None] & pair_mask[None, :]
    batch_cos = tl.cos(batch_phase)
    batch_sin = tl.sin(batch_phase)
    time_cos = tl.cos(time_phase)
    time_sin = tl.sin(time_phase)

    # repeat_interleave(2) makes every adjacent pair exactly equal.
    tl.store(cos_ptr + output_base, batch_cos, mask=output_mask)
    tl.store(cos_ptr + output_base + 1, batch_cos, mask=output_mask)
    tl.store(sin_ptr + output_base, batch_sin, mask=output_mask)
    tl.store(sin_ptr + output_base + 1, batch_sin, mask=output_mask)
    time_output = output_base + rotary_dim
    tl.store(cos_ptr + time_output, time_cos, mask=output_mask)
    tl.store(cos_ptr + time_output + 1, time_cos, mask=output_mask)
    tl.store(sin_ptr + time_output, time_sin, mask=output_mask)
    tl.store(sin_ptr + time_output + 1, time_sin, mask=output_mask)


class ModelNew(nn.Module):
    """Drop-in module whose forward pass is a single custom Triton kernel."""

    def __init__(
        self,
        dim: int = 64,
        max_seq_len: int = 256,
        base: float = 10000.0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (
            torch.arange(0, dim, 2, dtype=torch.float) / dim
        ))
        self.register_buffer("inv_freq", inv_freq)

        positions = torch.arange(max_seq_len, dtype=torch.float)
        positions_norm = positions / max_seq_len * (2 * math.pi)
        position_angles = positions_norm.unsqueeze(-1) * inv_freq
        position_angles = position_angles.repeat_interleave(2, dim=-1)
        self.register_buffer("position_angles", position_angles)
        self._block_size, self._num_warps = _get_launch_config()
        # The multi-token 2-D broadcast form lowers pathologically on
        # Triton-Ascend 3.2.1.  The one-token row owner is the measured A2
        # default; other backends retain the portable flat kernel.
        self._use_a2_token_tiles = False
        chip = os.environ.get("KERNELSWIFT_CHIP", "nvidia_h200").strip().lower()
        if chip == "ascend_a2":
            a2_mode = os.environ.get(
                "KERNELSWIFT_TASK05_MODE", "token_row"
            ).strip().lower()
        elif chip == "metax_c500":
            # All four modes are portable Triton.  C500 can therefore compare
            # token ownership against the flat launch instead of inheriting an
            # A2-only restriction.
            a2_mode = os.environ.get(
                "KERNELSWIFT_TASK05_MODE", "flat"
            ).strip().lower()
        elif chip == "nvidia_h200":
            # These three implementations use standard Triton operations and
            # are safe bounded A/B candidates on CUDA.  The Ascend persistent
            # scheduler remains excluded because its program-count assumption
            # is device-specific.  The measured token-owner winner is the
            # no-selector default; explicit MODE=flat restores the historical
            # fused path and MODE=split_trig restores the third-round path.
            a2_mode = os.environ.get(
                "KERNELSWIFT_TASK05_MODE", "split_token"
            ).strip().lower()
            if a2_mode in ("", "default"):
                a2_mode = "split_token"
            if a2_mode not in (
                "flat", "token_row", "token_tile", "split_trig",
                "split_token", "token_sincos", "token_sincos_asm",
            ):
                raise ValueError(f"unsupported Task05 H200 mode: {a2_mode}")
        else:
            # Never let a stale A2 experiment variable route another backend
            # into an Ascend-specific persistent/token-row kernel.
            a2_mode = "flat"
        self._use_a2_persistent = a2_mode == "persistent"
        self._use_a2_token_row = a2_mode == "token_row"
        self._use_a2_token_tiles = a2_mode == "token_tile"
        self._use_split_trig = (
            chip == "nvidia_h200"
            and a2_mode in (
                "split_trig", "split_token", "token_sincos",
                "token_sincos_asm",
            )
        )
        self._use_split_token = (
            chip == "nvidia_h200" and a2_mode == "split_token"
        )
        self._use_token_sincos = (
            chip == "nvidia_h200"
            and a2_mode in ("token_sincos", "token_sincos_asm")
        )
        self._use_token_sincos_asm = (
            chip == "nvidia_h200" and a2_mode == "token_sincos_asm"
        )
        trig_mode = os.environ.get(
            "KERNELSWIFT_TASK05_TRIG", "intrinsic"
        ).strip().lower()
        if chip == "nvidia_h200" and trig_mode not in (
            "intrinsic",
            "poly",
            "poly_fast",
            "cos_sqrt",
        ):
            raise ValueError(f"unsupported Task05 H200 trig mode: {trig_mode}")
        self._use_poly_trig = chip == "nvidia_h200" and trig_mode in (
            "poly",
            "poly_fast",
        )
        self._use_fast_poly_trig = (
            chip == "nvidia_h200" and trig_mode == "poly_fast"
        )
        self._derive_sin_from_cos = (
            chip == "nvidia_h200" and trig_mode == "cos_sqrt"
        )
        # Optional allocation-overhead candidate for C500.  Slicing the
        # leading dimension gives two contiguous tensors with the exact
        # public shapes while retaining one flat backing allocation.
        self._use_packed_output = (
            chip in ("metax_c500", "nvidia_h200")
            and os.environ.get(
                "KERNELSWIFT_TASK05_PACKED_OUTPUT", "0"
            ).strip().lower() in ("1", "true", "yes")
        )
        program_key = os.environ.get(
            "KERNELSWIFT_TASK05_PROGRAMS", "48"
        ).strip()
        self._a2_persistent_programs = _A2_PERSISTENT_PROGRAMS.get(
            program_key, 48
        )
        token_tile_variants = {
            "2x1": (2, 1),
            "2x2": (2, 2),
            "4x2": (4, 2),
            "4x4": (4, 4),
            "8x4": (8, 4),
        }
        token_tile_key = os.environ.get(
            "KERNELSWIFT_TASK05_TOKEN_TILE", "4x4"
        ).strip().lower()
        self._token_tile_config = token_tile_variants.get(
            token_tile_key, _A2_TOKEN_TILE_CONFIG
        )
        triton36 = (
            str(getattr(triton, "__version__", "")).split("+", 1)[0]
            == "3.6.0"
        )
        compiled_default = "1" if chip == "nvidia_h200" and triton36 else "0"
        compiled_key = os.environ.get(
            "KERNELSWIFT_TASK05_COMPILED_LAUNCH", compiled_default
        ).strip().lower()
        self._use_compiled_launch = (
            chip == "nvidia_h200"
            and triton36
            and {
                "0": False,
                "false": False,
                "no": False,
                "1": True,
                "true": True,
                "yes": True,
            }.get(compiled_key, False)
        )
        self._compiled_launcher = None
        self._compiled_device = None
        self._compiled_config_is_official = (
            dim == 64
            and max_seq_len == 256
            and base == 10000.0
            and self._use_split_trig
            and self._block_size == 64
            and self._num_warps == 2
            and not self._use_poly_trig
            and not self._derive_sin_from_cos
            and not self._use_packed_output
        )
        self._fixed_token_config_is_official = (
            dim == 64 and max_seq_len == 256 and base == 10000.0
        )

        graph_key = os.environ.get("KERNELSWIFT_TASK05_GRAPH_REPLAY")
        if graph_key is None:
            graph_key = os.environ.get("KERNELSWIFT_H200_GRAPH_REPLAY", "1")
        graph_key = graph_key.strip().lower()
        if graph_key not in ("0", "false", "no", "1", "true", "yes"):
            raise ValueError("Task05 GRAPH_REPLAY must be 0 or 1")
        self._use_graph_replay = (
            chip == "nvidia_h200"
            and graph_key in ("1", "true", "yes")
            and torch.cuda.is_available()
        )
        raw_graph_key = os.environ.get(
            "KERNELSWIFT_TASK05_RAW_GRAPH_LAUNCH", "0"
        ).strip().lower()
        if raw_graph_key not in ("0", "false", "no", "1", "true", "yes"):
            raise ValueError("Task05 RAW_GRAPH_LAUNCH must be 0 or 1")
        self._use_raw_graph_launch = (
            self._use_graph_replay
            and raw_graph_key in ("1", "true", "yes")
            and getattr(triton, "__version__", "") == "3.6.0"
        )
        self._graph = None
        self._graph_inputs = None
        self._graph_outputs = None
        self._raw_graph_exec = None
        self._raw_graph_device_index = None

    def forward(self, timestamps: torch.Tensor, seq_len: int):
        graph = self._graph
        graph_inputs = self._graph_inputs
        if (
            graph is not None
            and graph_inputs is not None
            and graph_inputs[0] is timestamps
            and graph_inputs[1] == seq_len
            and graph_inputs[2] is self.inv_freq
            and graph_inputs[3] is self.position_angles
        ):
            raw_graph_exec = self._raw_graph_exec
            if raw_graph_exec is None:
                graph.replay()
            else:
                stream = _cuda_runtime.cudaStream_t(
                    torch._C._cuda_getCurrentRawStream(
                        self._raw_graph_device_index
                    )
                )
                status = _cuda_runtime.cudaGraphLaunch(
                    raw_graph_exec, stream
                )[0]
                if status != _cuda_runtime.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"cudaGraphLaunch failed: {status}")
            return self._graph_outputs

        official = (
            self._use_graph_replay
            and self._use_compiled_launch
            and self._fixed_token_config_is_official
            and self._use_split_token
            and timestamps.shape == (4, 32)
            and timestamps.dtype == torch.float32
            and timestamps.is_contiguous()
            and timestamps.device.type == "cuda"
            and self.inv_freq.device == timestamps.device
            and self.position_angles.device == timestamps.device
            and seq_len == 32
        )
        if not official:
            return self._forward_impl(timestamps, seq_len)

        self._forward_impl(timestamps, seq_len)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = self._forward_impl(timestamps, seq_len)
        self._graph = graph
        self._graph_inputs = (
            timestamps,
            seq_len,
            self.inv_freq,
            self.position_angles,
        )
        self._graph_outputs = outputs
        if self._use_raw_graph_launch:
            self._raw_graph_exec = _cuda_runtime.cudaGraphExec_t(
                graph.raw_cuda_graph_exec()
            )
            self._raw_graph_device_index = timestamps.device.index
        else:
            self._raw_graph_exec = None
            self._raw_graph_device_index = None
        graph.replay()
        return outputs

    def _forward_impl(self, timestamps: torch.Tensor, seq_len: int):
        if timestamps.device.type == "cpu":
            raise RuntimeError("ModelNew requires accelerator tensors")
        if timestamps.ndim != 2:
            raise ValueError("timestamps must have shape [batch_size, sequence_length]")
        if seq_len < 0 or seq_len > timestamps.shape[1]:
            raise ValueError("seq_len must be within the timestamps sequence dimension")
        if seq_len > self.position_angles.shape[0]:
            raise ValueError("seq_len exceeds max_seq_len")

        batch_size = timestamps.shape[0]
        rotary_dim = self.position_angles.shape[1]
        output_dim = 2 * rotary_dim
        output_shape = (batch_size, seq_len, output_dim)
        if self._use_packed_output:
            output_backing = torch.empty(
                (2, *output_shape),
                device=timestamps.device,
                dtype=self.inv_freq.dtype,
            )
            cos = output_backing[0]
            sin = output_backing[1]
        else:
            cos = torch.empty(
                output_shape,
                device=timestamps.device,
                dtype=self.inv_freq.dtype,
            )
            sin = torch.empty_like(cos)

        if self._use_a2_token_row:
            token_total = batch_size * seq_len
            programs = min(token_total, self._a2_persistent_programs)
            pair_block = triton.next_power_of_2(rotary_dim // 2)
            _a2_music_flamingo_token_row_kernel[(programs,)](
                timestamps,
                self.inv_freq,
                self.position_angles,
                cos,
                sin,
                timestamps.stride(0),
                timestamps.stride(1),
                self.position_angles.stride(0),
                cos.stride(0),
                cos.stride(1),
                seq_len,
                rotary_dim,
                self.max_seq_len,
                TOTAL_TOKENS=token_total,
                PAIR_BLOCK=pair_block,
                num_warps=self._num_warps,
            )
        elif self._use_a2_persistent:
            total = batch_size * seq_len * rotary_dim
            chunks = triton.cdiv(total, self._block_size)
            programs = min(chunks, self._a2_persistent_programs)
            _a2_music_flamingo_persistent_kernel[(programs,)](
                timestamps,
                self.inv_freq,
                self.position_angles,
                cos,
                sin,
                timestamps.stride(0),
                timestamps.stride(1),
                self.position_angles.stride(0),
                cos.stride(0),
                cos.stride(1),
                batch_size,
                seq_len,
                rotary_dim,
                self.max_seq_len,
                TOTAL=total,
                NUM_CHUNKS=chunks,
                BLOCK_SIZE=self._block_size,
                num_warps=self._num_warps,
            )
        elif self._use_a2_token_tiles:
            # Four tokens per program yields 32 independent tasks for the
            # official shape: enough to fill A2 while still halving the flat
            # kernel's launch-task count and broadcasting repeated loads.
            block_tokens, token_tile_warps = self._token_tile_config
            pair_count = rotary_dim // 2
            pair_block = triton.next_power_of_2(pair_count)
            token_total = batch_size * seq_len
            grid = (triton.cdiv(token_total, block_tokens),)
            _a2_music_flamingo_token_tile_kernel[grid](
                timestamps,
                self.inv_freq,
                self.position_angles,
                cos,
                sin,
                timestamps.stride(0),
                timestamps.stride(1),
                self.position_angles.stride(0),
                cos.stride(0),
                cos.stride(1),
                batch_size,
                seq_len,
                rotary_dim,
                self.max_seq_len,
                PAIR_COUNT=pair_count,
                PAIR_BLOCK=pair_block,
                BLOCK_TOKENS=block_tokens,
                num_warps=token_tile_warps,
            )
        elif (
            (self._use_split_token or self._use_token_sincos)
            and self._fixed_token_config_is_official
            and timestamps.shape == (4, 32)
            and timestamps.dtype == torch.float32
            and timestamps.is_contiguous()
            and timestamps.device.type == "cuda"
            and self.inv_freq.device == timestamps.device
            and self.position_angles.device == timestamps.device
            and seq_len == 32
        ):
            use_compiled = (
                self._use_compiled_launch
                and not self._use_packed_output
                and not self._use_poly_trig
                and not self._derive_sin_from_cos
            )
            if (
                use_compiled
                and self._compiled_launcher is not None
                and self._compiled_device == timestamps.device
            ):
                if self._use_split_token:
                    self._compiled_launcher(
                        timestamps,
                        self.inv_freq,
                        self.position_angles,
                        cos,
                        sin,
                    )
                else:
                    self._compiled_launcher(
                        timestamps,
                        self.inv_freq,
                        self.position_angles,
                        cos,
                        sin,
                        self._use_token_sincos_asm,
                    )
            else:
                if self._use_split_token:
                    compiled = _music_flamingo_rotary_split_token_kernel[
                        (128, 2)
                    ](
                        timestamps,
                        self.inv_freq,
                        self.position_angles,
                        cos,
                        sin,
                        num_warps=self._num_warps,
                    )
                    compiled_grid = (128, 2, 1)
                else:
                    compiled = _music_flamingo_rotary_token_sincos_kernel[
                        (128,)
                    ](
                        timestamps,
                        self.inv_freq,
                        self.position_angles,
                        cos,
                        sin,
                        INLINE_ASM=self._use_token_sincos_asm,
                        num_warps=self._num_warps,
                    )
                    compiled_grid = (128, 1, 1)
                if (
                    use_compiled
                    and self._compiled_launcher is None
                    and compiled is not None
                ):
                    self._compiled_launcher = compiled[compiled_grid]
                    self._compiled_device = timestamps.device
        elif self._use_split_trig:
            total = batch_size * seq_len * rotary_dim
            grid = (triton.cdiv(total, self._block_size), 2)
            use_compiled = (
                self._use_compiled_launch
                and self._compiled_config_is_official
                and timestamps.shape == (4, 32)
                and timestamps.dtype == torch.float32
                and timestamps.is_contiguous()
                and timestamps.device.type == "cuda"
                and self.inv_freq.device == timestamps.device
                and self.position_angles.device == timestamps.device
                and seq_len == 32
            )
            if (
                use_compiled
                and self._compiled_launcher is not None
                and self._compiled_device == timestamps.device
            ):
                self._compiled_launcher(
                    timestamps,
                    self.inv_freq,
                    self.position_angles,
                    cos,
                    sin,
                    timestamps.stride(0),
                    timestamps.stride(1),
                    self.position_angles.stride(0),
                    cos.stride(0),
                    cos.stride(1),
                    batch_size,
                    seq_len,
                    rotary_dim,
                    self.max_seq_len,
                    self._block_size,
                )
            else:
                compiled = _music_flamingo_rotary_split_trig_kernel[grid](
                    timestamps,
                    self.inv_freq,
                    self.position_angles,
                    cos,
                    sin,
                    timestamps.stride(0),
                    timestamps.stride(1),
                    self.position_angles.stride(0),
                    cos.stride(0),
                    cos.stride(1),
                    batch_size,
                    seq_len,
                    rotary_dim,
                    self.max_seq_len,
                    BLOCK_SIZE=self._block_size,
                    num_warps=self._num_warps,
                )
                if (
                    use_compiled
                    and self._compiled_launcher is None
                    and compiled is not None
                ):
                    self._compiled_launcher = compiled[(grid[0], 2, 1)]
                    self._compiled_device = timestamps.device
        else:
            total = batch_size * seq_len * rotary_dim
            grid = (triton.cdiv(total, self._block_size),)
            _music_flamingo_rotary_kernel[grid](
                timestamps,
                self.inv_freq,
                self.position_angles,
                cos,
                sin,
                timestamps.stride(0),
                timestamps.stride(1),
                self.position_angles.stride(0),
                cos.stride(0),
                cos.stride(1),
                batch_size,
                seq_len,
                rotary_dim,
                self.max_seq_len,
                BLOCK_SIZE=self._block_size,
                USE_POLY_TRIG=self._use_poly_trig,
                FAST_POLY_TRIG=self._use_fast_poly_trig,
                DERIVE_SIN_FROM_COS=self._derive_sin_from_cos,
                num_warps=self._num_warps,
            )
        return cos, sin


class Model(ModelNew):
    """Compatibility name matching the reference module."""

    pass


def get_inputs():
    """Official Task05 benchmark inputs (also required by auto_bench.py)."""
    batch_size, seq_len = 4, 32
    timestamps = torch.rand(batch_size, seq_len, device="cuda")
    return [timestamps, seq_len]


def get_init_inputs():
    return [64, 256, 10000.0]
