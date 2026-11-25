"""Triton implementation of Task05: MusicFlamingoRotaryEmbedding."""

import math
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl


def _target_device():
    """Prefer the MLU device for this profile, with a CUDA smoke-test fallback."""
    mlu = getattr(torch, "mlu", None)
    if mlu is not None and mlu.is_available():
        return torch.device("mlu")
    return torch.device("cuda")


_CHIP_CONFIGS = {
    "auto": (128, 4),
    "ascend_a2": (128, 4),
    "metax_c500": (256, 4),
    "iluvatar_bi150": (128, 4),
    # True S60 screening selected the narrow 128-value/W1 launch.  The default
    # execution mode below is token_row, which uses the same launch width.
    "enflame_s60": (128, 1),
    "thead_810e": (64, 2),
    # Position-owner is the default below; its 32-frequency working set is
    # tiny.  W1 follows public Cambricon elementwise/reduction starting points.
    "cambricon_mlu590": (64, 1),
    "hygon_bw1000": (256, 4),
    "mthreads_ph100": (256, 4),
    "kunlun_p800": (64, 2),
    "biren_br106m": (256, 4),
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
    chip = os.environ.get("KERNELSWIFT_CHIP", "cambricon_mlu590").strip().lower()
    if chip == "ascend_a2":
        key = os.environ.get("KERNELSWIFT_TASK05_FLAT_CONFIG", "64x2")
        return _A2_FLAT_CONFIGS.get(key, _A2_FLAT_CONFIGS["64x2"])
    if chip == "metax_c500":
        # Keep the C500 sweep on the portable flat Triton kernel.  The named
        # choices change only flat chunk size and launch width.
        key = os.environ.get("KERNELSWIFT_TASK05_FLAT_CONFIG", "256x4")
        variants = {
            "64x1": _A2_FLAT_CONFIGS["64x1"],
            "64x2": _A2_FLAT_CONFIGS["64x2"],
            "128x2": _A2_FLAT_CONFIGS["128x2"],
            "256x2": _A2_FLAT_CONFIGS["256x2"],
            "256x4": _A2_FLAT_CONFIGS["256x4"],
            "512x4": _A2_FLAT_CONFIGS["512x4"],
        }
        return variants.get(key, _CHIP_CONFIGS["metax_c500"])
    if chip == "enflame_s60":
        # GCU300 supports only power-of-two num_warps values up to eight.  The
        # named surface keeps block sizes aligned to the 64-value repeated-pair
        # structure and prevents arbitrary launch metadata from being injected.
        key = os.environ.get("KERNELSWIFT_TASK05_FLAT_CONFIG", "128x1")
        variants = {
            "64x1": (64, 1),
            "64x2": (64, 2),
            "64x4": (64, 4),
            "64x8": (64, 8),
            "128x1": (128, 1),
            "128x2": (128, 2),
            "128x4": (128, 4),
            "128x8": (128, 8),
            "256x1": (256, 1),
            "256x2": (256, 2),
            "256x4": (256, 4),
            "256x8": (256, 8),
            "512x1": (512, 1),
            "512x2": (512, 2),
            "512x4": (512, 4),
            "512x8": (512, 8),
        }
        return variants.get(key, _CHIP_CONFIGS["enflame_s60"])
    if chip == "cambricon_mlu590":
        key = os.environ.get("KERNELSWIFT_TASK05_FLAT_CONFIG", "64x1")
        variants = {
            "64x1": (64, 1),
            "64x4": (64, 4),
            "128x1": (128, 1),
            "128x4": (128, 4),
            "256x1": (256, 1),
            "256x4": (256, 4),
        }
        return variants.get(key, _CHIP_CONFIGS["cambricon_mlu590"])
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
):
    """Compute each repeated frequency once and write its two equal lanes."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
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
    cos_value = tl.cos(phase)
    sin_value = tl.sin(phase)
    tl.store(cos_ptr + output_offsets, cos_value, mask=mask)
    tl.store(cos_ptr + output_offsets + 1, cos_value, mask=mask)
    tl.store(sin_ptr + output_offsets, sin_value, mask=mask)
    tl.store(sin_ptr + output_offsets + 1, sin_value, mask=mask)


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
    pair_count = rotary_dim // 2

    for chunk in range(pid, NUM_CHUNKS, programs):
        offsets = chunk * BLOCK_SIZE + lane
        mask = offsets < TOTAL
        unique_feature = offsets % rotary_dim
        token = (offsets // rotary_dim) % seq_len
        batch = offsets // (seq_len * rotary_dim)

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
        output_offsets = (
            batch * output_stride_b + token * output_stride_s + output_feature
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
            timestamps_ptr + batch * timestamp_stride_b + token * timestamp_stride_s
        ).to(tl.float32)
        inv_freq = tl.load(inv_freq_ptr + pair).to(tl.float32)
        time_freq = tl.load(
            position_angles_ptr + token * position_stride_s + 2 * pair
        ).to(tl.float32)
        angle = -6.283185307179586 * timestamp
        batch_phase = batch.to(tl.float32) / max_seq_len * inv_freq * angle
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

        output_base = batch * output_stride_b + token * output_stride_s
        tl.store(cos_ptr + output_base + out_lane, batch_cos_pairs)
        tl.store(sin_ptr + output_base + out_lane, batch_sin_pairs)
        time_base = output_base + rotary_dim
        tl.store(cos_ptr + time_base + out_lane, time_cos_pairs)
        tl.store(sin_ptr + time_base + out_lane, time_sin_pairs)


@triton.jit
def _mlu_music_flamingo_position_owner_kernel(
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
    rotary_dim: tl.constexpr,
    max_seq_len: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    PAIR_BLOCK: tl.constexpr,
):
    """Own one position, reuse its frequencies, and remove batch-zero trig.

    The official batch is four.  For batch zero the batch-frequency half is
    identically zero, hence cos=1 and sin=0 without evaluating either
    transcendental.  The remaining three batch halves and all four time
    halves retain the reference operation order.
    """
    position = tl.program_id(0)
    pair = tl.arange(0, PAIR_BLOCK)
    out_lane = tl.arange(0, 2 * PAIR_BLOCK)
    inv_freq = tl.load(inv_freq_ptr + pair).to(tl.float32)
    time_freq = tl.load(
        position_angles_ptr + position * position_stride_s + 2 * pair
    ).to(tl.float32)

    for batch in tl.static_range(0, BATCH_SIZE):
        timestamp = tl.load(
            timestamps_ptr + batch * timestamp_stride_b + position * timestamp_stride_s
        ).to(tl.float32)
        angle = -6.283185307179586 * timestamp
        time_phase = time_freq * angle
        time_cos = tl.cos(time_phase)
        time_sin = tl.sin(time_phase)
        if batch == 0:
            batch_cos = tl.full((PAIR_BLOCK,), 1.0, tl.float32)
            batch_sin = tl.zeros((PAIR_BLOCK,), tl.float32)
        else:
            batch_freq = (batch * 1.0 / max_seq_len) * inv_freq
            batch_phase = batch_freq * angle
            batch_cos = tl.cos(batch_phase)
            batch_sin = tl.sin(batch_phase)

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
        output_base = batch * output_stride_b + position * output_stride_s
        tl.store(cos_ptr + output_base + out_lane, batch_cos_pairs)
        tl.store(sin_ptr + output_base + out_lane, batch_sin_pairs)
        time_base = output_base + rotary_dim
        tl.store(cos_ptr + time_base + out_lane, time_cos_pairs)
        tl.store(sin_ptr + time_base + out_lane, time_sin_pairs)


@triton.jit
def _music_flamingo_online_angle_kernel(
    timestamps_ptr,
    inv_freq_ptr,
    cos_ptr,
    sin_ptr,
    timestamp_stride_b: tl.constexpr,
    timestamp_stride_s: tl.constexpr,
    output_stride_b: tl.constexpr,
    output_stride_s: tl.constexpr,
    seq_len: tl.constexpr,
    rotary_dim: tl.constexpr,
    max_seq_len: tl.constexpr,
    TOTAL_TOKENS: tl.constexpr,
    PAIR_BLOCK: tl.constexpr,
):
    """Portable row owner that rebuilds time angles from position and inv_freq.

    The operation order mirrors buffer construction in the reference:
    ``float(position) / max_seq_len * 2pi``, then multiply by ``inv_freq``.
    Adjacent lanes are still formed by the same repeat-interleave expansion.
    """
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    pair = tl.arange(0, PAIR_BLOCK)
    out_lane = tl.arange(0, 2 * PAIR_BLOCK)
    two_pi: tl.constexpr = 6.283185307179586

    for global_token in range(pid, TOTAL_TOKENS, programs):
        batch = global_token // seq_len
        position = global_token - batch * seq_len
        timestamp = tl.load(
            timestamps_ptr + batch * timestamp_stride_b + position * timestamp_stride_s
        ).to(tl.float32)
        inv_freq = tl.load(inv_freq_ptr + pair).to(tl.float32)
        common_angle = -two_pi * timestamp
        batch_freq = batch.to(tl.float32) / max_seq_len * inv_freq
        position_norm = position.to(tl.float32) / max_seq_len * two_pi
        time_freq = position_norm * inv_freq
        batch_phase = batch_freq * common_angle
        time_phase = time_freq * common_angle

        batch_cos = tl.cos(batch_phase)
        batch_sin = tl.sin(batch_phase)
        time_cos = tl.cos(time_phase)
        time_sin = tl.sin(time_phase)
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
        output_base = batch * output_stride_b + position * output_stride_s
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
    global_token = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    pair = tl.arange(0, PAIR_BLOCK)
    total_tokens = batch_size * seq_len
    token_mask = global_token < total_tokens
    pair_mask = pair < PAIR_COUNT

    batch = global_token // seq_len
    token = global_token % seq_len
    timestamp = tl.load(
        timestamps_ptr + batch * timestamp_stride_b + token * timestamp_stride_s,
        mask=token_mask,
        other=0.0,
    ).to(tl.float32)
    inv_freq = tl.load(
        inv_freq_ptr + pair,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    time_freq = tl.load(
        position_angles_ptr + token[:, None] * position_stride_s + (2 * pair[None, :]),
        mask=token_mask[:, None] & pair_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    common_angle = -6.283185307179586 * timestamp[:, None]
    batch_freq = (batch[:, None].to(tl.float32) / max_seq_len) * inv_freq[None, :]
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
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
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
        chip = os.environ.get("KERNELSWIFT_CHIP", "cambricon_mlu590").strip().lower()
        if chip == "ascend_a2":
            a2_mode = (
                os.environ.get("KERNELSWIFT_TASK05_MODE", "token_row").strip().lower()
            )
        elif chip == "metax_c500":
            # All four modes are portable Triton.  C500 can therefore compare
            # token ownership against the flat launch instead of inheriting an
            # A2-only restriction.
            a2_mode = os.environ.get("KERNELSWIFT_TASK05_MODE", "flat").strip().lower()
        elif chip in ("enflame_s60", "cambricon_mlu590"):
            # These kernels use portable Triton operations despite their
            # historical A2 names.  Expose them for target-side comparison;
            # an invalid override returns to the chip's static default rather
            # than silently selecting an exploratory kernel.
            candidate_mode = (
                os.environ.get(
                    "KERNELSWIFT_TASK05_MODE",
                    "position_owner" if chip == "cambricon_mlu590" else "token_row",
                )
                .strip()
                .lower()
            )
            if candidate_mode in (
                "flat",
                "persistent",
                "token_row",
                "token_tile",
                "online_angle",
                "position_owner",
            ):
                a2_mode = candidate_mode
            else:
                a2_mode = (
                    "position_owner" if chip == "cambricon_mlu590" else "token_row"
                )
        else:
            # Never let a stale A2 experiment variable route another backend
            # into an Ascend-specific persistent/token-row kernel.
            a2_mode = "flat"
        self._use_a2_persistent = a2_mode == "persistent"
        self._use_a2_token_row = a2_mode == "token_row"
        self._use_a2_token_tiles = a2_mode == "token_tile"
        self._use_online_angle = (
            chip in ("enflame_s60", "cambricon_mlu590") and a2_mode == "online_angle"
        )
        self._use_position_owner = (
            chip == "cambricon_mlu590" and a2_mode == "position_owner"
        )
        if chip == "enflame_s60":
            default_programs = "24"
            fallback_programs = 24
        elif chip == "cambricon_mlu590":
            default_programs = "32"
            fallback_programs = 32
        else:
            default_programs = "48"
            fallback_programs = 48
        program_key = os.environ.get(
            "KERNELSWIFT_TASK05_PROGRAMS", default_programs
        ).strip()
        self._a2_persistent_programs = _A2_PERSISTENT_PROGRAMS.get(
            program_key, fallback_programs
        )
        if chip == "cambricon_mlu590":
            # Cambricon ``num_warps`` selects BLOCK/U1/Ux task geometry rather
            # than CUDA warp-sized thread groups.  Keep the offline MLU sweep
            # to the commonly published W1/W4 choices.
            token_tile_variants = {
                "2x1": (2, 1),
                "4x1": (4, 1),
                "4x4": (4, 4),
                "8x1": (8, 1),
                "8x4": (8, 4),
            }
            token_tile_default = "4x1"
            token_tile_fallback = (4, 1)
        else:
            token_tile_variants = {
                "2x1": (2, 1),
                "2x2": (2, 2),
                "4x2": (4, 2),
                "4x4": (4, 4),
                "8x4": (8, 4),
            }
            token_tile_default = "4x4"
            token_tile_fallback = _A2_TOKEN_TILE_CONFIG
        token_tile_key = (
            os.environ.get("KERNELSWIFT_TASK05_TOKEN_TILE", token_tile_default)
            .strip()
            .lower()
        )
        self._token_tile_config = token_tile_variants.get(
            token_tile_key, token_tile_fallback
        )

    def forward(self, timestamps: torch.Tensor, seq_len: int):
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
        # One allocation, two non-overlapping contiguous views.  The backing
        # is fresh on every call, so this trims allocator overhead without
        # introducing cross-call output reuse.
        output_storage = torch.empty(
            (2,) + output_shape,
            device=timestamps.device,
            dtype=self.inv_freq.dtype,
        )
        cos = output_storage[0]
        sin = output_storage[1]

        if self._use_position_owner:
            pair_block = triton.next_power_of_2(rotary_dim // 2)
            _mlu_music_flamingo_position_owner_kernel[(seq_len,)](
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
                rotary_dim,
                self.max_seq_len,
                BATCH_SIZE=batch_size,
                PAIR_BLOCK=pair_block,
                num_warps=1,
                num_stages=1,
            )
        elif self._use_online_angle:
            token_total = batch_size * seq_len
            programs = min(token_total, self._a2_persistent_programs)
            pair_block = triton.next_power_of_2(rotary_dim // 2)
            _music_flamingo_online_angle_kernel[(programs,)](
                timestamps,
                self.inv_freq,
                cos,
                sin,
                timestamps.stride(0),
                timestamps.stride(1),
                cos.stride(0),
                cos.stride(1),
                seq_len,
                rotary_dim,
                self.max_seq_len,
                TOTAL_TOKENS=token_total,
                PAIR_BLOCK=pair_block,
                num_warps=1,
            )
        elif self._use_a2_token_row:
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
                num_warps=self._num_warps,
            )
        return cos, sin


class Model(ModelNew):
    """Compatibility name matching the reference module."""

    pass


def get_inputs():
    """Official Task05 benchmark inputs (also required by auto_bench.py)."""
    batch_size, seq_len = 4, 32
    timestamps = torch.rand(batch_size, seq_len, device=_target_device())
    return [timestamps, seq_len]


def get_init_inputs():
    return [64, 256, 10000.0]
