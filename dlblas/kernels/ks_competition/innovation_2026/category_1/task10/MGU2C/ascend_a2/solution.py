"""Fused elementwise sigmoid backward and reductions for Task10."""

import os
from importlib import metadata as importlib_metadata

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.compiler.compiler import CompiledKernel


def _is_triton_ascend_321_distribution():
    """Admit the private launcher ABI only for its audited distribution.

    ``triton-ascend`` 3.2.1 reports ``triton.__version__ == "3.2.0"`` in the
    competition image, so the installed distribution metadata is the
    authoritative release identity.  The path check also rejects an unrelated
    ``triton`` package shadowing the files owned by that distribution.
    """
    distribution = next(
        (
            item
            for item in importlib_metadata.distributions()
            if item.metadata.get("Name", "").lower().replace("_", "-")
            == "triton-ascend"
        ),
        None,
    )
    if distribution is None:
        return False
    version = distribution.version.split("+", 1)[0]
    distribution_triton = os.path.realpath(
        str(distribution.locate_file("triton"))
    )
    imported_triton = os.path.realpath(str(triton.__file__))
    common_path = os.path.commonpath(
        (distribution_triton, imported_triton)
    )
    return (
        version == "3.2.1"
        and common_path == distribution_triton
        and callable(getattr(CompiledKernel, "__getitem__", None))
    )


def _compiled_launch_requested():
    """Return the Task10 launcher A/B choice; compiled is the default arm."""
    value = os.environ.get(
        "KERNELSWIFT_TASK10_COMPILED_LAUNCH", "compiled"
    ).strip().lower()
    if value in ("compiled", "1", "on", "true"):
        return True
    if value in ("jit", "0", "off", "false"):
        return False
    raise ValueError(
        "KERNELSWIFT_TASK10_COMPILED_LAUNCH must be compiled/jit or 1/0"
    )


def _raw_pointer_args_requested():
    """Select the frozen-launcher runtime argument representation.

    The ``ptr`` arm still derives every address from the current-call tensor on
    every invocation.  It never retains a tensor or a raw pointer; it only
    avoids the Ascend launcher's repeated Tensor pointer-attribute lookup.
    """
    value = os.environ.get(
        "KERNELSWIFT_TASK10_RAW_POINTER_ARGS", "ptr"
    ).strip().lower()
    if value in ("ptr", "pointer", "1", "on", "true"):
        return True
    if value in ("tensor", "0", "off", "false"):
        return False
    raise ValueError(
        "KERNELSWIFT_TASK10_RAW_POINTER_ARGS must be ptr/tensor or 1/0"
    )


def _compiled_grid_runner(compiled, grid3):
    """Extract a v3.2.1 grid runner without retaining call-time tensors."""
    if not isinstance(compiled, CompiledKernel):
        return None
    runner = compiled[grid3]
    return runner if callable(runner) else None


def _matches_frozen_task10_abi(tensors):
    """Check every property shared by the frozen exact-shape launchers."""
    specs = (
        ((2, 1024, 4), (4096, 4, 1)),
        ((1,), (1,)),
        ((4,), (1,)),
        ((2, 1024, 4), (4096, 4, 1)),
        ((2, 1024, 4), (4096, 4, 1)),
        ((1,), (1,)),
        ((4,), (1,)),
    )
    if len(tensors) != len(specs):
        return False
    first_device = tensors[0].device
    if first_device.type != "npu":
        return False
    for tensor, (shape, stride) in zip(tensors, specs):
        if (
            tensor.device != first_device
            or tensor.dtype != torch.float32
            or tuple(tensor.shape) != shape
            or tuple(tensor.stride()) != stride
            or tensor.data_ptr() % 16 != 0
        ):
            return False
    return True


def _select_kernel_config():
    """Return ``(profile, num_warps)`` for the persistent reduction."""
    configs = {
        "auto": 4,
        "ascend_a2": 2,
        "metax_c500": 8,
        "iluvatar_bi150": 8,
        "enflame_s60": 4,
        "thead_810e": 4,
        "cambricon_mlu590": 4,
        "hygon_bw1000": 8,
        "mthreads_ph100": 8,
        "kunlun_p800": 4,
        "biren_br106m": 8,
    }
    profile = os.environ.get("KERNELSWIFT_CHIP", "ascend_a2").strip().lower()
    if profile == "default" or profile not in configs:
        profile = "auto"
    default_warps = configs[profile]
    warp_options = {"1": 1, "2": 2, "4": 4}
    warp_key = os.environ.get(
        "KERNELSWIFT_TASK10_WARPS", str(default_warps)
    ).strip()
    return profile, warp_options.get(warp_key, default_warps)


# These are scalar-element tile sizes.  Matrix modes are clamped to at most
# 1024 elements below: the 3.2.1 backend materialises enough hidden temporaries
# for the nominally small 2048-element matrix tile to overflow A2's 192-KiB UB.
_ASCEND_BLOCKS = {
    "128": 128,
    "256": 256,
    "512": 512,
    "1024": 1024,
    "2048": 2048,
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
def _mix_bwd_persistent_matrix_kernel(
    input_ptr, scale_ptr, base_ptr, grad_out_ptr,
    grad_input_ptr, grad_scale_ptr, grad_base_ptr,
    N_ROWS: tl.constexpr, hc_mult: tl.constexpr,
    NUM_TILES: tl.constexpr, BLOCK_ROWS: tl.constexpr,
    SYMMETRIC_DERIVATIVE: tl.constexpr,
):
    """One-launch row-major reduction candidate for A2.

    A single program owns all five reductions, so this needs neither an
    initialization launch, atomics, nor a partial workspace.  Unlike the old
    flat ``tiled`` diagnostic, the live tile is ``[BLOCK_ROWS, hc_mult]``:
    channels stay contiguous for the elementwise work and the four base sums
    are accumulated as a short vector.  Tiling keeps the live set below A2's
    192-KiB UB instead of materialising the full 8192-element expression.
    """
    row_lane = tl.arange(0, BLOCK_ROWS)[:, None]
    channel_1d = tl.arange(0, hc_mult)
    channel = channel_1d[None, :]
    scale = tl.load(scale_ptr).to(tl.float32)
    base = tl.load(base_ptr + channel).to(tl.float32)
    scale_total = 0.0
    base_total = tl.zeros((hc_mult,), dtype=tl.float32)

    for tile in tl.range(0, NUM_TILES):
        row = tile * BLOCK_ROWS + row_lane
        valid = row < N_ROWS
        off = row * hc_mult + channel
        x = tl.load(input_ptr + off, mask=valid, other=0.0).to(tl.float32)
        grad = tl.load(
            grad_out_ptr + off, mask=valid, other=0.0
        ).to(tl.float32)
        z = x * scale + base
        if SYMMETRIC_DERIVATIVE:
            # sigmoid(z) * (1 - sigmoid(z)) is even.  This form keeps the
            # exponential argument non-positive and removes the subtract
            # from one.  It is an algebraic rewrite, not reduced precision.
            exp_abs = tl.exp(-tl.abs(z))
            denominator = 1.0 + exp_abs
            sigmoid_derivative = exp_abs / (denominator * denominator)
        else:
            sig = 1.0 / (1.0 + tl.exp(-z))
            sigmoid_derivative = sig * (1.0 - sig)
        grad_z = grad * sigmoid_derivative
        tl.store(grad_input_ptr + off, grad_z * scale, mask=valid)

        scale_by_row = tl.sum(
            tl.where(valid, grad_z * x, 0.0), axis=1
        )
        scale_total += tl.sum(scale_by_row, axis=0)
        base_total += tl.sum(tl.where(valid, grad_z, 0.0), axis=0)

    tl.store(grad_scale_ptr, scale_total)
    tl.store(grad_base_ptr + channel_1d, base_total)


@triton.jit
def _mix_bwd_persistent_matrix_exact_kernel(
    input_ptr, scale_ptr, base_ptr, grad_out_ptr,
    grad_input_ptr, grad_scale_ptr, grad_base_ptr,
    NUM_TILES: tl.constexpr, BLOCK_ROWS: tl.constexpr,
    SYMMETRIC_DERIVATIVE: tl.constexpr,
):
    """Mask-free fixed ``[2048, 4]`` version used by the audited runner.

    Every supported ``BLOCK_ROWS`` divides 2048.  The strict host ABI guard
    proves the shape and contiguous layout before this entry is reachable, so
    bounds predicates and masked loads/stores would be redundant timed work.
    """
    row_lane = tl.arange(0, BLOCK_ROWS)[:, None]
    channel_1d = tl.arange(0, 4)
    channel = channel_1d[None, :]
    scale = tl.load(scale_ptr).to(tl.float32)
    base = tl.load(base_ptr + channel).to(tl.float32)
    scale_total = 0.0
    base_total = tl.zeros((4,), dtype=tl.float32)

    for tile in tl.range(0, NUM_TILES):
        row = tile * BLOCK_ROWS + row_lane
        off = row * 4 + channel
        x = tl.load(input_ptr + off).to(tl.float32)
        grad = tl.load(grad_out_ptr + off).to(tl.float32)
        z = x * scale + base
        if SYMMETRIC_DERIVATIVE:
            exp_abs = tl.exp(-tl.abs(z))
            denominator = 1.0 + exp_abs
            sigmoid_derivative = exp_abs / (denominator * denominator)
        else:
            sig = 1.0 / (1.0 + tl.exp(-z))
            sigmoid_derivative = sig * (1.0 - sig)
        grad_z = grad * sigmoid_derivative
        tl.store(grad_input_ptr + off, grad_z * scale)

        scale_by_channel = tl.sum(grad_z * x, axis=0)
        scale_total += tl.sum(scale_by_channel, axis=0)
        base_total += tl.sum(grad_z, axis=0)

    tl.store(grad_scale_ptr, scale_total)
    tl.store(grad_base_ptr + channel_1d, base_total)


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
    input_ptr, scale_ptr, base_ptr, grad_out_ptr,
    grad_input_ptr, partial_ptr,
    N_ROWS: tl.constexpr, hc_mult: tl.constexpr,
    NUM_BLOCKS: tl.constexpr, BLOCK_ROWS: tl.constexpr,
):
    """A2 stage one: contiguous ``[rows, hc_mult]`` work and SoA partials.

    The workspace has logical shape ``[1 + hc_mult, NUM_BLOCKS]``.  Keeping
    each reduction stream contiguous makes the tiny second-stage loads
    coalesced.  ``N_ROWS`` and ``NUM_BLOCKS`` are constexprs so all indexing
    and mask comparisons stay in the backend's supported 32-bit integer path.
    """
    block_id = tl.program_id(0)
    row = block_id * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    channel = tl.arange(0, hc_mult)[None, :]
    valid = row < N_ROWS
    off = row * hc_mult + channel

    x = tl.load(input_ptr + off, mask=valid, other=0.0).to(tl.float32)
    grad = tl.load(grad_out_ptr + off, mask=valid, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    base = tl.load(base_ptr + channel).to(tl.float32)

    z = x * scale + base
    sig = 1.0 / (1.0 + tl.exp(-z))
    grad_z = grad * sig * (1.0 - sig)
    tl.store(grad_input_ptr + off, grad_z * scale, mask=valid)

    scale_by_row = tl.sum(tl.where(valid, grad_z * x, 0.0), axis=1)
    tl.store(partial_ptr + block_id, tl.sum(scale_by_row, axis=0))
    base_partial = tl.sum(tl.where(valid, grad_z, 0.0), axis=0)
    tl.store(
        partial_ptr + (1 + tl.arange(0, hc_mult)) * NUM_BLOCKS + block_id,
        base_partial,
    )


@triton.jit
def _mix_bwd_matrix_atomic_kernel(
    input_ptr, scale_ptr, base_ptr, grad_out_ptr,
    grad_input_ptr, grad_scale_ptr, grad_base_ptr,
    N_ROWS: tl.constexpr, hc_mult: tl.constexpr, BLOCK_ROWS: tl.constexpr,
):
    """Contiguous ``[rows, channel]`` stage with five scalar atomics/block."""
    block_id = tl.program_id(0)
    row = block_id * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    channel = tl.arange(0, hc_mult)[None, :]
    valid = row < N_ROWS
    off = row * hc_mult + channel

    x = tl.load(input_ptr + off, mask=valid, other=0.0).to(tl.float32)
    grad = tl.load(grad_out_ptr + off, mask=valid, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    base = tl.load(base_ptr + channel).to(tl.float32)
    z = x * scale + base
    sig = 1.0 / (1.0 + tl.exp(-z))
    grad_z = grad * sig * (1.0 - sig)
    tl.store(grad_input_ptr + off, grad_z * scale, mask=valid)

    scale_by_row = tl.sum(tl.where(valid, grad_z * x, 0.0), axis=1)
    tl.atomic_add(grad_scale_ptr, tl.sum(scale_by_row, axis=0))
    base_partial = tl.sum(tl.where(valid, grad_z, 0.0), axis=0)
    tl.atomic_add(grad_base_ptr + tl.arange(0, hc_mult), base_partial)


@triton.jit
def _mix_bwd_soa_finalize_kernel(
    partial_ptr, grad_scale_ptr, grad_base_ptr,
    NUM_BLOCKS: tl.constexpr, hc_mult: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
):
    """Reduce the five contiguous SoA streams without atomics."""
    block = tl.arange(0, REDUCE_BLOCK)
    valid = block < NUM_BLOCKS
    scale_partial = tl.load(partial_ptr + block, mask=valid, other=0.0)
    tl.store(grad_scale_ptr, tl.sum(scale_partial, axis=0))
    for channel in tl.static_range(0, hc_mult):
        base_partial = tl.load(
            partial_ptr + (1 + channel) * NUM_BLOCKS + block,
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
    ascend_mode, ascend_block,
):
    tensors = (input_mix, mhc_scale, mhc_base, grad_out)
    if any(t.device.type == "cpu" for t in tensors):
        raise ValueError("head_compute_mix_bwd requires accelerator tensors")
    if not all(t.device == input_mix.device for t in tensors):
        raise ValueError("all inputs must be on the same accelerator")
    grad_input = torch.empty_like(input_mix)
    grad_scale = torch.empty_like(mhc_scale)
    grad_base = torch.empty_like(mhc_base)
    hc_mult = input_mix.shape[-1]
    n_elements = input_mix.numel()

    # This chip-specific submission defaults to A2 when the optional profile
    # variable is absent, while still honoring an explicit non-A2 override.
    if profile == "ascend_a2":
        block = ascend_block
        if ascend_mode in ("persistent_matrix", "persistent_symmetric"):
            n_rows = n_elements // hc_mult
            block_rows = block // hc_mult
            num_tiles = triton.cdiv(n_rows, block_rows)
            if _matches_frozen_task10_abi(
                (
                    input_mix, mhc_scale, mhc_base, grad_out,
                    grad_input, grad_scale, grad_base,
                )
            ) and block in (256, 512, 1024):
                _mix_bwd_persistent_matrix_exact_kernel[(1,)](
                    input_mix, mhc_scale, mhc_base, grad_out,
                    grad_input, grad_scale, grad_base,
                    NUM_TILES=num_tiles,
                    BLOCK_ROWS=block_rows,
                    SYMMETRIC_DERIVATIVE=(
                        ascend_mode == "persistent_symmetric"
                    ),
                    num_warps=num_warps,
                )
                return grad_input, grad_scale, grad_base
            _mix_bwd_persistent_matrix_kernel[(1,)](
                input_mix, mhc_scale, mhc_base, grad_out,
                grad_input, grad_scale, grad_base,
                N_ROWS=n_rows,
                hc_mult=hc_mult,
                NUM_TILES=num_tiles,
                BLOCK_ROWS=block_rows,
                SYMMETRIC_DERIVATIVE=(
                    ascend_mode == "persistent_symmetric"
                ),
                num_warps=num_warps,
            )
            return grad_input, grad_scale, grad_base

        if ascend_mode == "atomic":
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

        if ascend_mode == "channel":
            elements_per_channel = triton.cdiv(n_elements, hc_mult)
            blocks_per_channel = triton.cdiv(elements_per_channel, block)
            num_blocks = blocks_per_channel * hc_mult
            partial = torch.empty(
                (num_blocks, 2),
                device=input_mix.device,
                dtype=torch.float32,
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

        if ascend_mode == "tiled":
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

        if ascend_mode == "matrix":
            n_rows = n_elements // hc_mult
            block_rows = block // hc_mult
            num_blocks = triton.cdiv(n_rows, block_rows)
            partial = torch.empty(
                (1 + hc_mult, num_blocks),
                device=input_mix.device,
                dtype=torch.float32,
            )
            _mix_bwd_matrix_partial_kernel[(num_blocks,)](
                input_mix, mhc_scale, mhc_base, grad_out,
                grad_input, partial,
                N_ROWS=n_rows,
                hc_mult=hc_mult,
                NUM_BLOCKS=num_blocks,
                BLOCK_ROWS=block_rows,
                num_warps=num_warps,
            )
            reduce_block = triton.next_power_of_2(num_blocks)
            _mix_bwd_soa_finalize_kernel[(1,)](
                partial, grad_scale, grad_base,
                NUM_BLOCKS=num_blocks,
                hc_mult=hc_mult,
                REDUCE_BLOCK=reduce_block,
                num_warps=num_warps,
            )
            return grad_input, grad_scale, grad_base

        if ascend_mode == "matrix_atomic":
            _zero_reductions_kernel[(1,)](
                grad_scale, grad_base,
                hc_mult=hc_mult,
                num_warps=1,
            )
            n_rows = n_elements // hc_mult
            block_rows = block // hc_mult
            num_blocks = triton.cdiv(n_rows, block_rows)
            _mix_bwd_matrix_atomic_kernel[(num_blocks,)](
                input_mix, mhc_scale, mhc_base, grad_out,
                grad_input, grad_scale, grad_base,
                N_ROWS=n_rows,
                hc_mult=hc_mult,
                BLOCK_ROWS=block_rows,
                num_warps=num_warps,
            )
            return grad_input, grad_scale, grad_base

        num_blocks = triton.cdiv(n_elements, block)
        partial_width = 1 + hc_mult
        partial = torch.empty(
            (num_blocks, partial_width),
            device=input_mix.device,
            dtype=torch.float32,
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
        self._ascend_mode = os.environ.get(
            "KERNELSWIFT_TASK10_MODE", "persistent_matrix"
        ).strip().lower()
        if self._ascend_mode not in (
            "atomic", "channel", "matrix", "matrix_atomic", "tiled",
            "persistent_matrix", "persistent_symmetric", "two_stage"
        ):
            self._ascend_mode = "persistent_matrix"
        default_block = (
            "1024"
            if self._ascend_mode in (
                "persistent_matrix", "persistent_symmetric"
            )
            else "1024" if self._ascend_mode == "two_stage" else "128"
        )
        block_key = os.environ.get(
            "KERNELSWIFT_TASK10_BLOCK", default_block
        ).strip()
        self._ascend_block = _ASCEND_BLOCKS.get(block_key, 128)
        if (
            self._ascend_mode.startswith(("matrix", "persistent"))
            and self._ascend_block > 1024
        ):
            self._ascend_block = 1024

        # This is a frozen, exact-layout optimization for the two kernels in
        # A2's default matrix_atomic path.  A/B=jit bypasses it completely.
        self._use_compiled_launch = (
            _compiled_launch_requested()
            and _is_triton_ascend_321_distribution()
        )
        self._use_raw_pointer_args = _raw_pointer_args_requested()
        self._compiled_config_is_default = (
            self.kernel_profile == "ascend_a2"
            and self._ascend_mode == "matrix_atomic"
            and self._ascend_block == 128
            and self._num_warps == 2
        )
        self._compiled_config_is_persistent_matrix = (
            self.kernel_profile == "ascend_a2"
            and self._ascend_mode in (
                "persistent_matrix", "persistent_symmetric"
            )
            and self._ascend_block in (256, 512, 1024)
            and self._num_warps in (1, 2, 4)
        )
        self._compiled_config_is_two_stage = (
            self.kernel_profile == "ascend_a2"
            and self._ascend_mode == "two_stage"
            and self._ascend_block in (256, 512, 1024)
            and self._num_warps in (1, 2, 4)
        )
        self._compiled_config_is_matrix = (
            self.kernel_profile == "ascend_a2"
            and self._ascend_mode == "matrix"
            and self._ascend_block in (128, 256, 512, 1024)
            and self._num_warps in (1, 2, 4)
        )
        # Only grid-bound executable runners and immutable specialization
        # metadata live across calls.  No tensor, pointer, output, or stream is
        # retained.  CompiledKernel.__getitem__ resolves the current NPU stream
        # inside every runner invocation on triton-ascend 3.2.1.
        self._zero_compiled_launcher = None
        self._matrix_atomic_compiled_launcher = None
        self._persistent_matrix_compiled_launcher = None
        self._partial_compiled_launcher = None
        self._finalize_compiled_launcher = None
        self._matrix_partial_compiled_launcher = None
        self._matrix_finalize_compiled_launcher = None
        self._compiled_device = None
        self._compiled_abi_key = None
        self._compiled_kernel_metadata = None

    def _forward_frozen_matrix_atomic(
        self, input_mix, mhc_scale, mhc_base, grad_out,
    ):
        """Run and lazily publish the exact 910B2 two-kernel specialization."""
        grad_input = torch.empty_like(input_mix)
        grad_scale = torch.empty_like(mhc_scale)
        grad_base = torch.empty_like(mhc_base)
        all_tensors = (
            input_mix, mhc_scale, mhc_base, grad_out,
            grad_input, grad_scale, grad_base,
        )
        if not _matches_frozen_task10_abi(all_tensors):
            return None

        # Fixed competition shape: 8192 elements, four channels, 2048 rows,
        # 32 rows/program at BLOCK=128.  Grids stay explicitly three-axis.
        zero_grid3 = (1, 1, 1)
        matrix_grid3 = (64, 1, 1)
        abi_key = (
            "triton-ascend-3.2.1",
            "task10-matrix-atomic",
            zero_grid3,
            matrix_grid3,
            2048,
            4,
            32,
            self._num_warps,
            "float32",
            ((2, 1024, 4), (1,), (4,), (2, 1024, 4)),
            ((4096, 4, 1), (1,), (1,), (4096, 4, 1)),
            "aligned-16",
        )
        if (
            self._zero_compiled_launcher is not None
            and self._matrix_atomic_compiled_launcher is not None
            and self._compiled_device == input_mix.device
            and self._compiled_abi_key == abi_key
        ):
            # v3.2.1 runner ABI contains non-constexpr arguments only.
            self._zero_compiled_launcher(grad_scale, grad_base)
            self._matrix_atomic_compiled_launcher(
                input_mix, mhc_scale, mhc_base, grad_out,
                grad_input, grad_scale, grad_base,
            )
            return grad_input, grad_scale, grad_base

        # The miss computes this invocation normally.  We publish runners only
        # after both real launches have succeeded, so no partially initialized
        # launcher pair can be observed by a later call.
        zero_compiled = _zero_reductions_kernel[zero_grid3](
            grad_scale, grad_base,
            hc_mult=4,
            num_warps=1,
        )
        matrix_compiled = _mix_bwd_matrix_atomic_kernel[matrix_grid3](
            input_mix, mhc_scale, mhc_base, grad_out,
            grad_input, grad_scale, grad_base,
            N_ROWS=2048,
            hc_mult=4,
            BLOCK_ROWS=32,
            num_warps=self._num_warps,
        )
        zero_launcher = _compiled_grid_runner(zero_compiled, zero_grid3)
        matrix_launcher = _compiled_grid_runner(
            matrix_compiled, matrix_grid3
        )
        if zero_launcher is not None and matrix_launcher is not None:
            self._zero_compiled_launcher = zero_launcher
            self._matrix_atomic_compiled_launcher = matrix_launcher
            self._compiled_device = input_mix.device
            self._compiled_kernel_metadata = (
                getattr(zero_compiled, "metadata", None),
                getattr(matrix_compiled, "metadata", None),
            )
            # Publish the exact ABI key last; it is the direct-path validity
            # marker and contains no tensor identity or pointer value.
            self._compiled_abi_key = abi_key
        return grad_input, grad_scale, grad_base

    def _forward_frozen_persistent_matrix(
        self, input_mix, mhc_scale, mhc_base, grad_out,
    ):
        """Run the exact one-launch 910B2 persistent specialization.

        Only its executable launcher and immutable metadata are cached.  All
        three outputs are freshly allocated on every invocation and the v3.2.1
        runner still resolves the current NPU stream at call time.
        """
        grad_input = torch.empty_like(input_mix)
        grad_scale = torch.empty_like(mhc_scale)
        grad_base = torch.empty_like(mhc_base)
        all_tensors = (
            input_mix, mhc_scale, mhc_base, grad_out,
            grad_input, grad_scale, grad_base,
        )
        if not _matches_frozen_task10_abi(all_tensors):
            return None

        grid3 = (1, 1, 1)
        block_rows = self._ascend_block // 4
        num_tiles = triton.cdiv(2048, block_rows)
        symmetric_derivative = (
            self._ascend_mode == "persistent_symmetric"
        )
        abi_key = (
            "triton-ascend-3.2.1",
            "task10-persistent-matrix",
            grid3,
            2048,
            4,
            block_rows,
            num_tiles,
            symmetric_derivative,
            self._num_warps,
            "float32",
            ((2, 1024, 4), (1,), (4,), (2, 1024, 4)),
            ((4096, 4, 1), (1,), (1,), (4096, 4, 1)),
            "aligned-16",
        )
        if (
            self._persistent_matrix_compiled_launcher is not None
            and self._compiled_device == input_mix.device
            and self._compiled_abi_key == abi_key
        ):
            if self._use_raw_pointer_args:
                # Addresses are reacquired from this invocation's tensors.
                # No pointer, tensor, output, or stream survives the call.
                self._persistent_matrix_compiled_launcher(
                    int(input_mix.data_ptr()),
                    int(mhc_scale.data_ptr()),
                    int(mhc_base.data_ptr()),
                    int(grad_out.data_ptr()),
                    int(grad_input.data_ptr()),
                    int(grad_scale.data_ptr()),
                    int(grad_base.data_ptr()),
                )
            else:
                # Every non-constexpr argument is a current-call tensor.
                self._persistent_matrix_compiled_launcher(
                    input_mix, mhc_scale, mhc_base, grad_out,
                    grad_input, grad_scale, grad_base,
                )
            return grad_input, grad_scale, grad_base

        compiled = _mix_bwd_persistent_matrix_exact_kernel[grid3](
            input_mix, mhc_scale, mhc_base, grad_out,
            grad_input, grad_scale, grad_base,
            NUM_TILES=num_tiles,
            BLOCK_ROWS=block_rows,
            SYMMETRIC_DERIVATIVE=symmetric_derivative,
            num_warps=self._num_warps,
        )
        launcher = _compiled_grid_runner(compiled, grid3)
        if launcher is not None:
            self._persistent_matrix_compiled_launcher = launcher
            self._compiled_device = input_mix.device
            self._compiled_kernel_metadata = getattr(
                compiled, "metadata", None
            )
            self._compiled_abi_key = abi_key
        return grad_input, grad_scale, grad_base

    def _forward_frozen_two_stage(
        self, input_mix, mhc_scale, mhc_base, grad_out,
    ):
        """Run the exact workspace reduction with both launchers frozen."""
        grad_input = torch.empty_like(input_mix)
        grad_scale = torch.empty_like(mhc_scale)
        grad_base = torch.empty_like(mhc_base)
        all_tensors = (
            input_mix, mhc_scale, mhc_base, grad_out,
            grad_input, grad_scale, grad_base,
        )
        if not _matches_frozen_task10_abi(all_tensors):
            return None

        num_blocks = triton.cdiv(8192, self._ascend_block)
        partial = torch.empty(
            (num_blocks, 5),
            device=input_mix.device,
            dtype=torch.float32,
        )
        if (
            tuple(partial.stride()) != (5, 1)
            or partial.data_ptr() % 16 != 0
        ):
            return None
        partial_grid3 = (num_blocks, 1, 1)
        finalize_grid3 = (1, 1, 1)
        reduce_block = triton.next_power_of_2(num_blocks)
        abi_key = (
            "triton-ascend-3.2.1",
            "task10-two-stage",
            partial_grid3,
            finalize_grid3,
            8192,
            4,
            5,
            self._ascend_block,
            num_blocks,
            reduce_block,
            self._num_warps,
            "float32",
            ((2, 1024, 4), (1,), (4,), (2, 1024, 4)),
            ((4096, 4, 1), (1,), (1,), (4096, 4, 1)),
            "aligned-16",
        )
        if (
            self._partial_compiled_launcher is not None
            and self._finalize_compiled_launcher is not None
            and self._compiled_device == input_mix.device
            and self._compiled_abi_key == abi_key
        ):
            self._partial_compiled_launcher(
                input_mix, mhc_scale, mhc_base, grad_out,
                grad_input, partial, 8192,
            )
            self._finalize_compiled_launcher(
                partial, grad_scale, grad_base, num_blocks
            )
            return grad_input, grad_scale, grad_base

        partial_compiled = _mix_bwd_partial_kernel[partial_grid3](
            input_mix, mhc_scale, mhc_base, grad_out,
            grad_input, partial,
            8192,
            hc_mult=4,
            PARTIAL_WIDTH=5,
            BLOCK=self._ascend_block,
            num_warps=self._num_warps,
        )
        finalize_compiled = _mix_bwd_finalize_kernel[finalize_grid3](
            partial, grad_scale, grad_base,
            num_blocks,
            hc_mult=4,
            PARTIAL_WIDTH=5,
            REDUCE_BLOCK=reduce_block,
            num_warps=self._num_warps,
        )
        partial_launcher = _compiled_grid_runner(
            partial_compiled, partial_grid3
        )
        finalize_launcher = _compiled_grid_runner(
            finalize_compiled, finalize_grid3
        )
        if partial_launcher is not None and finalize_launcher is not None:
            self._partial_compiled_launcher = partial_launcher
            self._finalize_compiled_launcher = finalize_launcher
            self._compiled_device = input_mix.device
            self._compiled_kernel_metadata = (
                getattr(partial_compiled, "metadata", None),
                getattr(finalize_compiled, "metadata", None),
            )
            self._compiled_abi_key = abi_key
        return grad_input, grad_scale, grad_base

    def _forward_frozen_matrix(
        self, input_mix, mhc_scale, mhc_base, grad_out,
    ):
        """Run the exact SoA matrix-partial reduction with frozen launchers."""
        grad_input = torch.empty_like(input_mix)
        grad_scale = torch.empty_like(mhc_scale)
        grad_base = torch.empty_like(mhc_base)
        all_tensors = (
            input_mix, mhc_scale, mhc_base, grad_out,
            grad_input, grad_scale, grad_base,
        )
        if not _matches_frozen_task10_abi(all_tensors):
            return None

        block_rows = self._ascend_block // 4
        num_blocks = triton.cdiv(2048, block_rows)
        partial = torch.empty(
            (5, num_blocks),
            device=input_mix.device,
            dtype=torch.float32,
        )
        if (
            tuple(partial.stride()) != (num_blocks, 1)
            or partial.data_ptr() % 16 != 0
        ):
            return None
        partial_grid3 = (num_blocks, 1, 1)
        finalize_grid3 = (1, 1, 1)
        reduce_block = triton.next_power_of_2(num_blocks)
        abi_key = (
            "triton-ascend-3.2.1",
            "task10-matrix-partial",
            partial_grid3,
            finalize_grid3,
            2048,
            4,
            block_rows,
            num_blocks,
            reduce_block,
            self._num_warps,
            "float32",
            ((2, 1024, 4), (1,), (4,), (2, 1024, 4)),
            ((4096, 4, 1), (1,), (1,), (4096, 4, 1)),
            "aligned-16",
        )
        if (
            self._matrix_partial_compiled_launcher is not None
            and self._matrix_finalize_compiled_launcher is not None
            and self._compiled_device == input_mix.device
            and self._compiled_abi_key == abi_key
        ):
            self._matrix_partial_compiled_launcher(
                input_mix, mhc_scale, mhc_base, grad_out,
                grad_input, partial,
            )
            self._matrix_finalize_compiled_launcher(
                partial, grad_scale, grad_base
            )
            return grad_input, grad_scale, grad_base

        partial_compiled = _mix_bwd_matrix_partial_kernel[partial_grid3](
            input_mix, mhc_scale, mhc_base, grad_out,
            grad_input, partial,
            N_ROWS=2048,
            hc_mult=4,
            NUM_BLOCKS=num_blocks,
            BLOCK_ROWS=block_rows,
            num_warps=self._num_warps,
        )
        finalize_compiled = _mix_bwd_soa_finalize_kernel[finalize_grid3](
            partial, grad_scale, grad_base,
            NUM_BLOCKS=num_blocks,
            hc_mult=4,
            REDUCE_BLOCK=reduce_block,
            num_warps=self._num_warps,
        )
        partial_launcher = _compiled_grid_runner(
            partial_compiled, partial_grid3
        )
        finalize_launcher = _compiled_grid_runner(
            finalize_compiled, finalize_grid3
        )
        if partial_launcher is not None and finalize_launcher is not None:
            self._matrix_partial_compiled_launcher = partial_launcher
            self._matrix_finalize_compiled_launcher = finalize_launcher
            self._compiled_device = input_mix.device
            self._compiled_kernel_metadata = (
                getattr(partial_compiled, "metadata", None),
                getattr(finalize_compiled, "metadata", None),
            )
            self._compiled_abi_key = abi_key
        return grad_input, grad_scale, grad_base

    def forward(self, input_mix, mhc_scale, mhc_base, grad_out):
        if self._use_compiled_launch and self._compiled_config_is_matrix:
            frozen_result = self._forward_frozen_matrix(
                input_mix, mhc_scale, mhc_base, grad_out
            )
            if frozen_result is not None:
                return frozen_result
        if self._use_compiled_launch and self._compiled_config_is_two_stage:
            frozen_result = self._forward_frozen_two_stage(
                input_mix, mhc_scale, mhc_base, grad_out
            )
            if frozen_result is not None:
                return frozen_result
        if (
            self._use_compiled_launch
            and self._compiled_config_is_persistent_matrix
        ):
            frozen_result = self._forward_frozen_persistent_matrix(
                input_mix, mhc_scale, mhc_base, grad_out
            )
            if frozen_result is not None:
                return frozen_result
        if self._use_compiled_launch and self._compiled_config_is_default:
            frozen_result = self._forward_frozen_matrix_atomic(
                input_mix, mhc_scale, mhc_base, grad_out
            )
            if frozen_result is not None:
                return frozen_result
        return head_compute_mix_bwd(
            input_mix, mhc_scale, mhc_base, grad_out,
            self.kernel_profile, self._num_warps,
            self._ascend_mode, self._ascend_block,
        )


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
