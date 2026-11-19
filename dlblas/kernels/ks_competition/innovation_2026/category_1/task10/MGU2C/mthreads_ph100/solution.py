"""Fused elementwise sigmoid backward and reductions for Task10.

The PH100 default is a portable non-atomic two-stage ``[rows, 4]`` reduction.
``KERNELSWIFT_TASK10_MODE``, ``KERNELSWIFT_TASK10_BLOCK`` and
``KERNELSWIFT_TASK10_WARPS`` expose the existing bounded alternatives for
reproducible physical-device sweeps.
"""

import os
import torch
import torch.nn as nn
import triton
import triton.language as tl


class _TrustedHotPath:
    """Cache only layout metadata; reductions and fresh outputs rerun."""

    __slots__ = ("_signature",)

    def __init__(self):
        self._signature = None

    def changed(self, tensors) -> bool:
        signature = tuple(
            (
                tuple(int(v) for v in tensor.shape),
                tuple(int(v) for v in tensor.stride()),
                int(tensor.data_ptr()),
                tensor.dtype,
                tensor.device.type,
                tensor.device.index,
            )
            for tensor in tensors
        )
        changed = signature != self._signature
        self._signature = signature
        return changed


class _ReductionWorkspace:
    """Reuse only the private partial-reduction buffer on the hot path.

    The three returned gradients are allocated afresh on every call.  This
    object owns no user-visible result and never stores input-dependent values
    between launches; it only keeps a correctly sized FP32 scratch allocation
    so the graph-like repeated call avoids allocator churn.  A shape/device
    change replaces the scratch buffer before the next kernel launch.
    """

    __slots__ = ("_tensor", "_key")

    def __init__(self):
        self._tensor = None
        self._key = None

    def get(self, shape, device):
        key = (tuple(int(v) for v in shape), device.type, device.index)
        if self._tensor is None or self._key != key:
            self._tensor = torch.empty(shape, device=device, dtype=torch.float32)
            self._key = key
        return self._tensor


def _partial_buffer(workspace, shape, device):
    if workspace is None:
        return torch.empty(shape, device=device, dtype=torch.float32)
    return workspace.get(shape, device)


def _select_kernel_config():
    """Return ``(profile, num_warps)`` for the selected reduction path."""
    configs = {
        "auto": 4,
        "ascend_a2": 4,
        "metax_c500": 4,
        "iluvatar_bi150": 8,
        "enflame_s60": 4,
        "thead_810e": 4,
        "cambricon_mlu590": 4,
        "hygon_bw1000": 8,
        "mthreads_ph100": 4,
        "kunlun_p800": 4,
        "nvidia_h200": 8,
    }
    # This is the chip-specific PH100 submission copy, so it must remain on the
    # PH100 path even when the evaluator does not export KERNELSWIFT_CHIP.
    profile = os.environ.get(
        "KERNELSWIFT_CHIP", "mthreads_ph100"
    ).strip().lower()
    if profile in ("auto", "default") or profile not in configs:
        profile = "mthreads_ph100"

    default_warps = configs[profile]
    warp_options = {"1": 1, "2": 2, "4": 4, "8": 8}
    warp_key = os.environ.get(
        "KERNELSWIFT_TASK10_WARPS", str(default_warps)
    ).strip()
    return profile, warp_options.get(warp_key, default_warps)


# Explicit candidates keep A/B sweeps reproducible for the fixed 8192-element
# competition input.  The H200 winner iterates over two deterministic
# 4096-item tiles in one program; all former blocks remain available for
# diagnostics and for the other chip profiles.
_REDUCTION_BLOCKS = {
    "64": 64,
    "128": 128,
    "256": 256,
    "512": 512,
    "1024": 1024,
    "2048": 2048,
    "4096": 4096,
    "8192": 8192,
}


@triton.jit
def _zero_reductions_kernel(
    grad_scale_ptr, grad_base_ptr, hc_mult: tl.constexpr,
):
    tl.store(grad_scale_ptr, 0.0)
    channel = tl.arange(0, hc_mult)
    tl.store(grad_base_ptr + channel, 0.0)


@triton.jit
def _mix_bwd_atomic_kernel(
    input_ptr, scale_ptr, base_ptr, grad_out_ptr,
    grad_input_ptr, grad_scale_ptr, grad_base_ptr,
    n_elements, hc_mult: tl.constexpr, BLOCK: tl.constexpr,
):
    block_id = tl.program_id(0)
    off = block_id * BLOCK + tl.arange(0, BLOCK)
    valid = off < n_elements
    channel = off % hc_mult
    x = tl.load(input_ptr + off, mask=valid, other=0.0).to(tl.float32)
    grad = tl.load(grad_out_ptr + off, mask=valid, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    base = tl.load(base_ptr + channel, mask=valid, other=0.0).to(tl.float32)
    z = x * scale + base
    sig = 1.0 / (1.0 + tl.exp(-z))
    grad_z = grad * sig * (1.0 - sig)
    tl.store(grad_input_ptr + off, grad_z * scale, mask=valid)
    tl.atomic_add(
        grad_scale_ptr,
        tl.sum(tl.where(valid, grad_z * x, 0.0), axis=0),
    )
    for c in tl.static_range(0, hc_mult):
        tl.atomic_add(
            grad_base_ptr + c,
            tl.sum(tl.where(valid & (channel == c), grad_z, 0.0), axis=0),
        )


@triton.jit
def _mix_bwd_tiled_kernel(
    input_ptr, scale_ptr, base_ptr, grad_out_ptr,
    grad_input_ptr, grad_scale_ptr, grad_base_ptr,
    n_elements, hc_mult: tl.constexpr, NUM_TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One-launch A2 path that reuses a UB-sized tile across the reduction."""
    lane = tl.arange(0, BLOCK)
    scale = tl.load(scale_ptr).to(tl.float32)
    scale_total = 0.0
    base0_total = 0.0
    base1_total = 0.0
    base2_total = 0.0
    base3_total = 0.0

    for tile in tl.range(0, NUM_TILES):
        off = tile * BLOCK + lane
        valid = off < n_elements
        channel = off % hc_mult
        x = tl.load(input_ptr + off, mask=valid, other=0.0).to(tl.float32)
        grad = tl.load(grad_out_ptr + off, mask=valid, other=0.0).to(tl.float32)
        base = tl.load(base_ptr + channel, mask=valid, other=0.0).to(tl.float32)
        z = x * scale + base
        sig = 1.0 / (1.0 + tl.exp(-z))
        grad_z = grad * sig * (1.0 - sig)
        tl.store(grad_input_ptr + off, grad_z * scale, mask=valid)

        scale_total += tl.sum(tl.where(valid, grad_z * x, 0.0), axis=0)
        base0_total += tl.sum(
            tl.where(valid & (channel == 0), grad_z, 0.0), axis=0
        )
        base1_total += tl.sum(
            tl.where(valid & (channel == 1), grad_z, 0.0), axis=0
        )
        base2_total += tl.sum(
            tl.where(valid & (channel == 2), grad_z, 0.0), axis=0
        )
        base3_total += tl.sum(
            tl.where(valid & (channel == 3), grad_z, 0.0), axis=0
        )

    tl.store(grad_scale_ptr, scale_total)
    tl.store(grad_base_ptr, base0_total)
    tl.store(grad_base_ptr + 1, base1_total)
    tl.store(grad_base_ptr + 2, base2_total)
    tl.store(grad_base_ptr + 3, base3_total)


@triton.jit
def _mix_bwd_matrix_tiled_kernel(
    input_ptr, scale_ptr, base_ptr, grad_out_ptr,
    grad_input_ptr, grad_scale_ptr, grad_base_ptr,
    n_rows, hc_mult: tl.constexpr, NUM_TILES: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """One-launch C500 persistent reduction over contiguous ``[rows, 4]``.

    Compared with the flat tiled diagnostic, the two-dimensional mapping makes
    every 16-lane group access consecutive values and reduces all four base
    channels in one vector reduction.  A single program owns the five scalar
    reductions, so no atomic initialization or second finalizer launch is
    required.
    """
    row_lane = tl.arange(0, BLOCK_ROWS)[:, None]
    channels = tl.arange(0, hc_mult)[None, :]
    scale = tl.load(scale_ptr).to(tl.float32)
    scale_total = 0.0
    base_total = tl.zeros((1, hc_mult), tl.float32)

    for tile in tl.static_range(0, NUM_TILES):
        rows = tile * BLOCK_ROWS + row_lane
        valid = rows < n_rows
        offsets = rows * hc_mult + channels
        x = tl.load(input_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        grad = tl.load(
            grad_out_ptr + offsets, mask=valid, other=0.0
        ).to(tl.float32)
        base = tl.load(base_ptr + channels).to(tl.float32)
        z = x * scale + base
        sig = 1.0 / (1.0 + tl.exp(-z))
        grad_z = grad * sig * (1.0 - sig)
        tl.store(grad_input_ptr + offsets, grad_z * scale, mask=valid)

        scale_by_row = tl.sum(tl.where(valid, grad_z * x, 0.0), axis=1)
        scale_total += tl.sum(scale_by_row, axis=0)
        base_total += tl.sum(tl.where(valid, grad_z, 0.0), axis=0)

    tl.store(grad_scale_ptr, scale_total)
    tl.store(grad_base_ptr + channels, base_total)


@triton.jit
def _mix_bwd_single_kernel(
    input_ptr, scale_ptr, base_ptr, grad_out_ptr,
    grad_input_ptr, grad_scale_ptr, grad_base_ptr,
    n_elements, hc_mult: tl.constexpr, BLOCK: tl.constexpr,
):
    """Expected-shape fast path: one program owns all reductions.

    This removes both the initialization launch and atomic operations.  The
    competition case has only 8192 elements, which comfortably fits this
    persistent reduction on the tested backends.
    """
    off = tl.arange(0, BLOCK)
    valid = off < n_elements
    channel = off % hc_mult
    x = tl.load(input_ptr + off, mask=valid, other=0.0)
    grad = tl.load(grad_out_ptr + off, mask=valid, other=0.0)
    scale = tl.load(scale_ptr)
    base = tl.load(base_ptr + channel, mask=valid, other=0.0)
    z = x * scale + base
    sig = 1.0 / (1.0 + tl.exp(-z))
    grad_z = grad * sig * (1.0 - sig)
    tl.store(grad_input_ptr + off, grad_z * scale, mask=valid)
    tl.store(grad_scale_ptr, tl.sum(tl.where(valid, grad_z * x, 0.0), axis=0))
    for c in tl.static_range(0, hc_mult):
        total = tl.sum(tl.where(valid & (channel == c), grad_z, 0.0), axis=0)
        tl.store(grad_base_ptr + c, total)


@triton.jit
def _mix_bwd_partial_kernel(
    input_ptr, scale_ptr, base_ptr, grad_out_ptr,
    grad_input_ptr, partial_ptr,
    n_elements, hc_mult: tl.constexpr, PARTIAL_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write ``grad_input`` and one scale/base partial vector per block.

    ``partial_ptr`` has logical shape ``[num_blocks, 1 + hc_mult]``.  Column
    zero contains the scale partial and the remaining columns contain the
    per-channel base partials.  A single packed workspace avoids a second
    device allocation in the timed forward path.
    """
    block_id = tl.program_id(0)
    off = block_id * BLOCK + tl.arange(0, BLOCK)
    valid = off < n_elements
    channel = off % hc_mult

    x = tl.load(input_ptr + off, mask=valid, other=0.0).to(tl.float32)
    grad = tl.load(grad_out_ptr + off, mask=valid, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    base = tl.load(base_ptr + channel, mask=valid, other=0.0).to(tl.float32)

    z = x * scale + base
    sig = 1.0 / (1.0 + tl.exp(-z))
    grad_z = grad * sig * (1.0 - sig)
    tl.store(grad_input_ptr + off, grad_z * scale, mask=valid)

    partial_row = partial_ptr + block_id * PARTIAL_WIDTH
    scale_partial = tl.sum(tl.where(valid, grad_z * x, 0.0), axis=0)
    tl.store(partial_row, scale_partial)
    for c in tl.static_range(0, hc_mult):
        base_partial = tl.sum(
            tl.where(valid & (channel == c), grad_z, 0.0), axis=0
        )
        tl.store(partial_row + 1 + c, base_partial)


@triton.jit
def _mix_bwd_matrix_partial_kernel(
    input_ptr,
    scale_ptr,
    base_ptr,
    grad_out_ptr,
    grad_input_ptr,
    partial_ptr,
    n_rows,
    hc_mult: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """C500 stage one: contiguous ``[rows,4]`` work and SoA partials."""
    block_id = tl.program_id(0)
    rows = block_id * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    channels = tl.arange(0, hc_mult)[None, :]
    valid = rows < n_rows
    offsets = rows * hc_mult + channels

    x = tl.load(input_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    grad = tl.load(grad_out_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    base = tl.load(base_ptr + channels).to(tl.float32)
    z = x * scale + base
    sig = 1.0 / (1.0 + tl.exp(-z))
    grad_z = grad * sig * (1.0 - sig)
    tl.store(grad_input_ptr + offsets, grad_z * scale, mask=valid)

    scale_by_row = tl.sum(tl.where(valid, grad_z * x, 0.0), axis=1)
    tl.store(partial_ptr + block_id, tl.sum(scale_by_row, axis=0))
    base_partial = tl.sum(tl.where(valid, grad_z, 0.0), axis=0)
    tl.store(
        partial_ptr + (1 + tl.arange(0, hc_mult)) * NUM_BLOCKS + block_id,
        base_partial,
    )


@triton.jit
def _mix_bwd_matrix_56_partial_kernel(
    input_ptr,
    scale_ptr,
    base_ptr,
    grad_out_ptr,
    grad_input_ptr,
    partial_ptr,
    n_rows,
    hc_mult: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """Opt-in 56-way PH100 partial reduction over physical ``[rows, 4]``.

    ``ROWS_PER_PROGRAM`` partitions every row exactly once, while the power-of-
    two ``BLOCK_ROWS`` only controls the vector width.  This keeps the partial
    count fixed at 56 for real-device occupancy experiments without atomics.
    """
    block_id = tl.program_id(0)
    lane = tl.arange(0, BLOCK_ROWS)[:, None]
    rows = block_id * ROWS_PER_PROGRAM + lane
    channels = tl.arange(0, hc_mult)[None, :]
    valid = (lane < ROWS_PER_PROGRAM) & (rows < n_rows)
    offsets = rows * hc_mult + channels

    x = tl.load(input_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    grad = tl.load(grad_out_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    base = tl.load(base_ptr + channels).to(tl.float32)
    z = x * scale + base
    sig = 1.0 / (1.0 + tl.exp(-z))
    grad_z = grad * sig * (1.0 - sig)
    tl.store(grad_input_ptr + offsets, grad_z * scale, mask=valid)

    scale_by_row = tl.sum(tl.where(valid, grad_z * x, 0.0), axis=1)
    tl.store(partial_ptr + block_id, tl.sum(scale_by_row, axis=0))
    base_partial = tl.sum(tl.where(valid, grad_z, 0.0), axis=0)
    tl.store(
        partial_ptr + (1 + tl.arange(0, hc_mult)) * NUM_BLOCKS + block_id,
        base_partial,
    )


@triton.jit
def _mix_bwd_soa_finalize_kernel(
    partial_ptr,
    grad_scale_ptr,
    grad_base_ptr,
    num_blocks: tl.constexpr,
    hc_mult: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
):
    """Reduce five contiguous SoA rows with a single native C500 warp."""
    block = tl.arange(0, REDUCE_BLOCK)
    valid = block < num_blocks
    scale_partial = tl.load(partial_ptr + block, mask=valid, other=0.0)
    tl.store(grad_scale_ptr, tl.sum(scale_partial, axis=0))
    for channel in tl.static_range(0, hc_mult):
        base_partial = tl.load(
            partial_ptr + (1 + channel) * num_blocks + block,
            mask=valid,
            other=0.0,
        )
        tl.store(grad_base_ptr + channel, tl.sum(base_partial, axis=0))


@triton.jit
def _mix_bwd_channel_partial_kernel(
    input_ptr, scale_ptr, base_ptr, grad_out_ptr,
    grad_input_ptr, partial_ptr,
    elements_per_channel: tl.constexpr,
    hc_mult: tl.constexpr, BLOCK: tl.constexpr,
):
    """A2 stage one: each program owns one channel-local contiguous index tile.

    The physical tensor is interleaved by channel.  Mapping a program to one
    channel removes four masked base reductions from every flat tile while
    preserving enough independent programs to occupy the vector cores.
    """
    block_id = tl.program_id(0)
    channel = block_id % hc_mult
    channel_block = block_id // hc_mult
    channel_index = channel_block * BLOCK + tl.arange(0, BLOCK)
    valid = channel_index < elements_per_channel
    off = channel_index * hc_mult + channel

    x = tl.load(input_ptr + off, mask=valid, other=0.0).to(tl.float32)
    grad = tl.load(grad_out_ptr + off, mask=valid, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    base = tl.load(base_ptr + channel).to(tl.float32)
    z = x * scale + base
    sig = 1.0 / (1.0 + tl.exp(-z))
    grad_z = grad * sig * (1.0 - sig)
    tl.store(grad_input_ptr + off, grad_z * scale, mask=valid)

    partial_row = partial_ptr + block_id * 2
    tl.store(
        partial_row,
        tl.sum(tl.where(valid, grad_z * x, 0.0), axis=0),
    )
    tl.store(
        partial_row + 1,
        tl.sum(tl.where(valid, grad_z, 0.0), axis=0),
    )


@triton.jit
def _mix_bwd_channel_finalize_kernel(
    partial_ptr, grad_scale_ptr, grad_base_ptr,
    num_blocks, hc_mult: tl.constexpr, REDUCE_BLOCK: tl.constexpr,
):
    block_id = tl.arange(0, REDUCE_BLOCK)
    valid = block_id < num_blocks
    scale_partial = tl.load(
        partial_ptr + block_id * 2, mask=valid, other=0.0
    )
    base_partial = tl.load(
        partial_ptr + block_id * 2 + 1, mask=valid, other=0.0
    )
    tl.store(grad_scale_ptr, tl.sum(scale_partial, axis=0))
    channel = block_id % hc_mult
    for c in tl.static_range(0, hc_mult):
        tl.store(
            grad_base_ptr + c,
            tl.sum(tl.where(valid & (channel == c), base_partial, 0.0), axis=0),
        )


@triton.jit
def _mix_bwd_finalize_kernel(
    partial_ptr, grad_scale_ptr, grad_base_ptr,
    num_blocks, hc_mult: tl.constexpr, PARTIAL_WIDTH: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
):
    """Reduce the small packed partial matrix without atomics."""
    block_off = tl.arange(0, REDUCE_BLOCK)
    valid = block_off < num_blocks
    row_off = block_off * PARTIAL_WIDTH

    scale_partial = tl.load(partial_ptr + row_off, mask=valid, other=0.0)
    tl.store(grad_scale_ptr, tl.sum(scale_partial, axis=0))
    for c in tl.static_range(0, hc_mult):
        base_partial = tl.load(
            partial_ptr + row_off + 1 + c, mask=valid, other=0.0
        )
        tl.store(grad_base_ptr + c, tl.sum(base_partial, axis=0))


def head_compute_mix_bwd(
    input_mix, mhc_scale, mhc_base, grad_out, profile, num_warps,
    reduction_mode, reduction_block, trusted=False, workspace=None,
):
    tensors = (input_mix, mhc_scale, mhc_base, grad_out)
    if not trusted:
        if any(t.device.type == "cpu" for t in tensors):
            raise ValueError("head_compute_mix_bwd requires accelerator tensors")
        if not all(t.device == input_mix.device for t in tensors):
            raise ValueError("all inputs must be on the same accelerator")
    hc_mult = input_mix.shape[-1]
    n_elements = input_mix.numel()
    grad_input = torch.empty_like(input_mix)
    grad_scale = torch.empty_like(mhc_scale)
    grad_base = torch.empty_like(mhc_base)

    # MetaX exposes the C500 through a CUDA-compatible device type, so the
    # profile is the authoritative selector for its bounded-block reduction.
    # The NPU device check keeps the same source usable on Ascend runners that
    # do not propagate the optional tuning variable.
    if (
        profile in ("ascend_a2", "metax_c500")
        or (profile == "mthreads_ph100" and reduction_mode != "single")
        or (profile == "nvidia_h200" and reduction_mode != "single")
        or input_mix.device.type == "npu"
    ):
        block = reduction_block
        if reduction_mode == "atomic":
            _zero_reductions_kernel[(1,)](
                grad_scale, grad_base,
                hc_mult=hc_mult,
                num_warps=1,
            )
            num_blocks = triton.cdiv(n_elements, block)
            _mix_bwd_atomic_kernel[(num_blocks,)](
                input_mix, mhc_scale, mhc_base, grad_out,
                grad_input, grad_scale, grad_base,
                n_elements,
                hc_mult=hc_mult,
                BLOCK=block,
                num_warps=num_warps,
            )
            return grad_input, grad_scale, grad_base

        if reduction_mode == "channel":
            elements_per_channel = triton.cdiv(n_elements, hc_mult)
            blocks_per_channel = triton.cdiv(elements_per_channel, block)
            num_blocks = blocks_per_channel * hc_mult
            partial = _partial_buffer(
                workspace, (num_blocks, 2), input_mix.device
            )
            _mix_bwd_channel_partial_kernel[(num_blocks,)](
                input_mix, mhc_scale, mhc_base, grad_out,
                grad_input, partial,
                elements_per_channel=elements_per_channel,
                hc_mult=hc_mult,
                BLOCK=block,
                num_warps=num_warps,
            )
            reduce_block = triton.next_power_of_2(num_blocks)
            _mix_bwd_channel_finalize_kernel[(1,)](
                partial, grad_scale, grad_base,
                num_blocks,
                hc_mult=hc_mult,
                REDUCE_BLOCK=reduce_block,
                num_warps=num_warps,
            )
            return grad_input, grad_scale, grad_base

        if reduction_mode == "tiled":
            num_tiles = triton.cdiv(n_elements, block)
            _mix_bwd_tiled_kernel[(1,)](
                input_mix, mhc_scale, mhc_base, grad_out,
                grad_input, grad_scale, grad_base,
                n_elements,
                hc_mult=hc_mult,
                NUM_TILES=num_tiles,
                BLOCK=block,
                num_warps=num_warps,
            )
            return grad_input, grad_scale, grad_base

        if reduction_mode == "matrix_tiled":
            n_rows = triton.cdiv(n_elements, hc_mult)
            block_rows = triton.cdiv(block, hc_mult)
            num_tiles = triton.cdiv(n_rows, block_rows)
            _mix_bwd_matrix_tiled_kernel[(1,)](
                input_mix, mhc_scale, mhc_base, grad_out,
                grad_input, grad_scale, grad_base,
                n_rows,
                hc_mult=hc_mult,
                NUM_TILES=num_tiles,
                BLOCK_ROWS=block_rows,
                num_warps=num_warps,
            )
            return grad_input, grad_scale, grad_base

        if reduction_mode == "matrix":
            n_rows = triton.cdiv(n_elements, hc_mult)
            block_rows = triton.cdiv(block, hc_mult)
            num_blocks = triton.cdiv(n_rows, block_rows)
            partial = _partial_buffer(
                workspace, (1 + hc_mult, num_blocks), input_mix.device
            )
            _mix_bwd_matrix_partial_kernel[(num_blocks,)](
                input_mix,
                mhc_scale,
                mhc_base,
                grad_out,
                grad_input,
                partial,
                n_rows,
                hc_mult=hc_mult,
                NUM_BLOCKS=num_blocks,
                BLOCK_ROWS=block_rows,
                num_warps=num_warps,
            )
            reduce_block = triton.next_power_of_2(num_blocks)
            _mix_bwd_soa_finalize_kernel[(1,)](
                partial,
                grad_scale,
                grad_base,
                num_blocks=num_blocks,
                hc_mult=hc_mult,
                REDUCE_BLOCK=reduce_block,
                num_warps=1,
            )
            return grad_input, grad_scale, grad_base

        if reduction_mode == "partial56":
            if hc_mult != 4:
                raise ValueError("Task10 partial56 mode requires hc_mult=4")
            n_rows = triton.cdiv(n_elements, hc_mult)
            num_blocks = min(56, n_rows)
            rows_per_program = triton.cdiv(n_rows, num_blocks)
            block_rows = triton.next_power_of_2(rows_per_program)
            partial = _partial_buffer(
                workspace, (1 + hc_mult, num_blocks), input_mix.device
            )
            _mix_bwd_matrix_56_partial_kernel[(num_blocks,)](
                input_mix,
                mhc_scale,
                mhc_base,
                grad_out,
                grad_input,
                partial,
                n_rows,
                hc_mult=hc_mult,
                NUM_BLOCKS=num_blocks,
                ROWS_PER_PROGRAM=rows_per_program,
                BLOCK_ROWS=block_rows,
                num_warps=num_warps,
            )
            reduce_block = triton.next_power_of_2(num_blocks)
            _mix_bwd_soa_finalize_kernel[(1,)](
                partial,
                grad_scale,
                grad_base,
                num_blocks=num_blocks,
                hc_mult=hc_mult,
                REDUCE_BLOCK=reduce_block,
                num_warps=1,
            )
            return grad_input, grad_scale, grad_base

        num_blocks = triton.cdiv(n_elements, block)
        partial_width = 1 + hc_mult
        partial = _partial_buffer(
            workspace, (num_blocks, partial_width), input_mix.device
        )
        _mix_bwd_partial_kernel[(num_blocks,)](
            input_mix, mhc_scale, mhc_base, grad_out,
            grad_input, partial,
            n_elements,
            hc_mult=hc_mult,
            PARTIAL_WIDTH=partial_width,
            BLOCK=block,
            num_warps=num_warps,
        )
        reduce_block = triton.next_power_of_2(num_blocks)
        _mix_bwd_finalize_kernel[(1,)](
            partial, grad_scale, grad_base,
            num_blocks,
            hc_mult=hc_mult,
            PARTIAL_WIDTH=partial_width,
            REDUCE_BLOCK=reduce_block,
            num_warps=num_warps,
        )
        return grad_input, grad_scale, grad_base

    block = triton.next_power_of_2(n_elements)
    _mix_bwd_single_kernel[(1,)](
        input_mix, mhc_scale, mhc_base, grad_out,
        grad_input, grad_scale, grad_base,
        n_elements, hc_mult=hc_mult, BLOCK=block, num_warps=num_warps,
    )
    return grad_input, grad_scale, grad_base


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_profile, self._num_warps = _select_kernel_config()
        if self.kernel_profile == "mthreads_ph100":
            # Portable PH1 default: two non-atomic stages over contiguous
            # [rows, 4] tiles.  The 56-way and atomic paths remain opt-in.
            default_mode = "matrix"
        elif self.kernel_profile == "nvidia_h200":
            default_mode = "tiled"
        else:
            default_mode = "matrix_tiled"
        self._reduction_mode = os.environ.get(
            "KERNELSWIFT_TASK10_MODE", default_mode
        ).strip().lower()
        if self._reduction_mode not in (
            "single", "atomic", "channel", "matrix", "matrix_tiled",
            "partial56", "tiled", "two_stage"
        ):
            self._reduction_mode = default_mode
        self._trusted_hot_path = _TrustedHotPath()
        self._trusted_validated = False
        self._reduction_workspace = _ReductionWorkspace()

        # PH100 starts with 256 flat elements = 64 complete rows per partial.
        # H200 retains its measured value; other profiles keep their seed.
        if self.kernel_profile == "mthreads_ph100":
            default_block = 256
        elif self.kernel_profile == "nvidia_h200":
            default_block = 4096
        else:
            default_block = 8192
        block_key = os.environ.get(
            "KERNELSWIFT_TASK10_BLOCK", str(default_block)
        ).strip()
        self._reduction_block = _REDUCTION_BLOCKS.get(block_key, default_block)

    def forward(self, input_mix, mhc_scale, mhc_base, grad_out):
        changed = self._trusted_hot_path.changed(
            (input_mix, mhc_scale, mhc_base, grad_out)
        )
        trusted = not changed and self._trusted_validated
        out = head_compute_mix_bwd(
            input_mix, mhc_scale, mhc_base, grad_out,
            self.kernel_profile, self._num_warps,
            self._reduction_mode, self._reduction_block,
            trusted=trusted,
            workspace=self._reduction_workspace,
        )
        self._trusted_validated = True
        return out


class Model(ModelNew):
    """Compatibility name required by the written competition rules."""


batch0 = 2
batch1 = 1024
mhc_mult = 4


def get_inputs():
    return [
        torch.randn(batch0, batch1, mhc_mult, dtype=torch.float32),
        torch.randn(1, dtype=torch.float32),
        torch.randn(mhc_mult, dtype=torch.float32),
        torch.randn(batch0, batch1, mhc_mult, dtype=torch.float32),
    ]


def get_init_inputs():
    return []
