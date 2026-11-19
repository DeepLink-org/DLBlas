"""Triton implementation of Task05: MusicFlamingoRotaryEmbedding.

BI150 defaults to the validated FP32 CUDA/CoreX libdevice intrinsics.
``KERNELSWIFT_TASK05_MATH=native`` restores the former ``tl.sin``/``tl.cos``
path exactly; non-BI150 profiles always retain native math.
"""

import math
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


_CHIP_CONFIGS = {
    "auto": (128, 4),
    "ascend_a2": (128, 4),
    "metax_c500": (256, 4),
    "iluvatar_bi150": (128, 4),
    "enflame_s60": (128, 4),
    "thead_810e": (64, 2),
    "cambricon_mlu590": (64, 2),
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
    chip = os.environ.get("KERNELSWIFT_CHIP", "iluvatar_bi150").strip().lower()
    if chip == "ascend_a2":
        key = os.environ.get("KERNELSWIFT_TASK05_FLAT_CONFIG", "64x2")
        return _A2_FLAT_CONFIGS.get(key, _A2_FLAT_CONFIGS["64x2"])
    if chip in ("metax_c500", "iluvatar_bi150"):
        # Keep both 64-thread-warp backends on the portable flat Triton kernel.  The named
        # choices change only flat chunk size and launch width.
        default_key = "256x4" if chip == "metax_c500" else "128x2"
        key = os.environ.get("KERNELSWIFT_TASK05_FLAT_CONFIG", default_key)
        variants = {
            "64x1": _A2_FLAT_CONFIGS["64x1"],
            "64x2": _A2_FLAT_CONFIGS["64x2"],
            "128x2": _A2_FLAT_CONFIGS["128x2"],
            "256x2": _A2_FLAT_CONFIGS["256x2"],
            "256x4": _A2_FLAT_CONFIGS["256x4"],
            "512x4": _A2_FLAT_CONFIGS["512x4"],
        }
        return variants.get(key, _CHIP_CONFIGS[chip])
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
    USE_FAST_LIBDEVICE: tl.constexpr,
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
    if USE_FAST_LIBDEVICE:
        cos_value = libdevice.fast_cosf(phase)
        sin_value = libdevice.fast_sinf(phase)
    else:
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
    USE_FAST_LIBDEVICE: tl.constexpr,
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
        if USE_FAST_LIBDEVICE:
            cos_value = libdevice.fast_cosf(phase)
            sin_value = libdevice.fast_sinf(phase)
        else:
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
    USE_FAST_LIBDEVICE: tl.constexpr,
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

        if USE_FAST_LIBDEVICE:
            batch_cos = libdevice.fast_cosf(batch_phase)
            batch_sin = libdevice.fast_sinf(batch_phase)
            time_cos = libdevice.fast_cosf(time_phase)
            time_sin = libdevice.fast_sinf(time_phase)
        else:
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
    USE_FAST_LIBDEVICE: tl.constexpr,
):
    """A2 token-frequency tile with broadcasted timestamp/frequency loads.

    The flat portable kernel loads the same timestamp once per feature and
    launches 64 tiny programs for the official shape.  This two-dimensional
    tile loads four timestamps and 32 inverse frequencies, broadcasts them in
    registers, and emits both the batch and time halves in 32 programs.  That
    is enough parallel work to occupy the A2's AI cores without the 64 tiny
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
    if USE_FAST_LIBDEVICE:
        batch_cos = libdevice.fast_cosf(batch_phase)
        batch_sin = libdevice.fast_sinf(batch_phase)
        time_cos = libdevice.fast_cosf(time_phase)
        time_sin = libdevice.fast_sinf(time_phase)
    else:
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
        chip = os.environ.get("KERNELSWIFT_CHIP", "iluvatar_bi150").strip().lower()
        if chip == "ascend_a2":
            a2_mode = os.environ.get(
                "KERNELSWIFT_TASK05_MODE", "token_row"
            ).strip().lower()
        elif chip in ("metax_c500", "iluvatar_bi150"):
            # All four modes use portable Triton operations.  BI150's 64-lane
            # warp makes the 2x32 token tile especially useful: it broadcasts
            # timestamps/inverse frequencies within one full warp instead of
            # reloading them once per flat feature.
            a2_mode = os.environ.get(
                "KERNELSWIFT_TASK05_MODE", "flat"
            ).strip().lower()
        else:
            # Never let a stale A2 experiment variable route another backend
            # into an Ascend-specific persistent/token-row kernel.
            a2_mode = "flat"
        self._use_compiled_runner = chip == "iluvatar_bi150"
        # Eight strict seeds and repeated 200/500 trials promote fast libdevice
        # on BI150.  ``native`` is the exact one-variable rollback; a stale
        # variable can never route another backend into CUDA/CoreX externs.
        math_default = "fast_libdevice" if chip == "iluvatar_bi150" else "native"
        math_mode = os.environ.get(
            "KERNELSWIFT_TASK05_MATH", math_default
        ).strip().lower()
        self._use_fast_libdevice = (
            chip == "iluvatar_bi150" and math_mode == "fast_libdevice"
        )
        # Cache launchers only.  The two result tensors remain fresh allocations
        # and are fully written by Triton on every invocation.
        self._compiled_runners = {}
        launch_mode = os.environ.get(
            "KERNELSWIFT_TASK05_LAUNCH", "fixed"
        ).strip().lower()
        self._launch_mode = (
            launch_mode
            if self._use_compiled_runner and launch_mode in ("keyed", "fixed")
            else "keyed"
        )
        self._fixed_runner = None
        self._joint_outputs = (
            self._use_compiled_runner
            and os.environ.get(
                "KERNELSWIFT_TASK05_OUTPUT", "separate"
            ).strip().lower()
            == "joint"
        )
        self._use_a2_persistent = a2_mode == "persistent"
        self._use_a2_token_row = a2_mode == "token_row"
        self._use_a2_token_tiles = a2_mode == "token_tile"
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
        graph_value = os.environ.get("KERNELSWIFT_TASK05_GRAPH_REPLAY")
        if graph_value is None:
                graph_value = os.environ.get("KERNELSWIFT_BI150_GRAPH_REPLAY", "1")
        self._use_graph_replay = (
            chip == "iluvatar_bi150"
            and graph_value.strip().lower() in ("1", "true", "yes")
        )
        self._graph = None
        self._graph_inputs = None
        self._graph_outputs = None

    def forward(self, timestamps: torch.Tensor, seq_len: int):
        if not self._use_graph_replay or timestamps.device.type == "cpu":
            return self._forward_impl(timestamps, seq_len)

        graph_inputs = self._graph_inputs
        if (
            self._graph is not None
            and graph_inputs is not None
            and graph_inputs[0] is timestamps
            and graph_inputs[1] == seq_len
            and graph_inputs[2] is self.inv_freq
            and graph_inputs[3] is self.position_angles
        ):
            self._graph.replay()
            return self._graph_outputs

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
        if self._joint_outputs:
            # One allocator request plus two views; the returned tensors remain
            # fresh and every element is overwritten by the Triton kernel.
            cos_sin = torch.empty(
                (2, *output_shape),
                device=timestamps.device,
                dtype=self.inv_freq.dtype,
            )
            cos = cos_sin[0]
            sin = cos_sin[1]
        else:
            cos = torch.empty(
                output_shape, device=timestamps.device, dtype=self.inv_freq.dtype
            )
            sin = torch.empty_like(cos)

        fixed_layout = self._launch_mode == "fixed" and (
            batch_size == 4
            and seq_len == 32
            and rotary_dim == 64
            and self.max_seq_len == 256
            and timestamps.dtype == torch.float32
            and timestamps.device.type == "cuda"
            and timestamps.is_contiguous()
            and self.inv_freq.dtype == torch.float32
            and self.inv_freq.device == timestamps.device
            and self.position_angles.device == timestamps.device
        )

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
                USE_FAST_LIBDEVICE=self._use_fast_libdevice,
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
                USE_FAST_LIBDEVICE=self._use_fast_libdevice,
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
            compiled_grid = (grid[0], 1, 1)
            runner = False
            if (
                self._use_compiled_runner
                and fixed_layout
            ):
                runner = self._fixed_runner
                if runner is None and hasattr(
                    _a2_music_flamingo_token_tile_kernel, "warmup"
                ):
                    compiled = _a2_music_flamingo_token_tile_kernel.warmup(
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
                        USE_FAST_LIBDEVICE=self._use_fast_libdevice,
                        grid=compiled_grid,
                        num_warps=token_tile_warps,
                    )
                    runner = compiled[compiled_grid]
                    self._fixed_runner = runner
                elif runner is None:
                    self._fixed_runner = False
                    runner = False
                if runner is not False:
                    runner(
                        timestamps,
                        self.inv_freq,
                        self.position_angles,
                        cos,
                        sin,
                    )
            if runner is False:
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
                    USE_FAST_LIBDEVICE=self._use_fast_libdevice,
                    num_warps=token_tile_warps,
                )
        else:
            total = batch_size * seq_len * rotary_dim
            grid = (triton.cdiv(total, self._block_size),)
            compiled_grid = (grid[0], 1, 1)
            runner = False
            if self._use_compiled_runner:
                if fixed_layout:
                    runner = self._fixed_runner
                    if runner is None and hasattr(
                        _music_flamingo_rotary_kernel, "warmup"
                    ):
                        compiled = _music_flamingo_rotary_kernel.warmup(
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
                            self._use_fast_libdevice,
                            grid=compiled_grid,
                            num_warps=self._num_warps,
                        )
                        runner = compiled[compiled_grid]
                        self._fixed_runner = runner
                    elif runner is None:
                        self._fixed_runner = False
                        runner = False
                    if runner is not False:
                        runner(
                            timestamps,
                            self.inv_freq,
                            self.position_angles,
                            cos,
                            sin,
                        )
                else:
                    runner_key = (
                    compiled_grid,
                    self._block_size,
                    self._num_warps,
                    self._use_fast_libdevice,
                    self.max_seq_len,
                    timestamps.device,
                    timestamps.dtype,
                    tuple(timestamps.shape),
                    timestamps.stride(),
                    timestamps.data_ptr() & 15,
                    self.inv_freq.device,
                    self.inv_freq.dtype,
                    tuple(self.inv_freq.shape),
                    self.inv_freq.stride(),
                    self.inv_freq.data_ptr() & 15,
                    self.position_angles.device,
                    self.position_angles.dtype,
                    tuple(self.position_angles.shape),
                    self.position_angles.stride(),
                    self.position_angles.data_ptr() & 15,
                    cos.device,
                    cos.dtype,
                    tuple(cos.shape),
                    cos.stride(),
                    cos.data_ptr() & 15,
                    sin.device,
                    sin.dtype,
                    tuple(sin.shape),
                    sin.stride(),
                    sin.data_ptr() & 15,
                    )
                    runner = self._compiled_runners.get(runner_key)
                    if runner is not False:
                        if runner is None and hasattr(
                            _music_flamingo_rotary_kernel, "warmup"
                        ):
                            compiled = _music_flamingo_rotary_kernel.warmup(
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
                                self._use_fast_libdevice,
                                grid=compiled_grid,
                                num_warps=self._num_warps,
                            )
                            runner = compiled[compiled_grid]
                            self._compiled_runners[runner_key] = runner
                        elif runner is None:
                            self._compiled_runners[runner_key] = False
                            runner = False
                        if runner is not False:
                            # CoreX keeps only the five pointer arguments in the
                            # compiled signature; scalar layout data is constexpr.
                            runner(
                                timestamps,
                                self.inv_freq,
                                self.position_angles,
                                cos,
                                sin,
                            )
            if runner is False:
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
                    USE_FAST_LIBDEVICE=self._use_fast_libdevice,
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
