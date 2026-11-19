"""Fused sigmoid backward with deterministic and atomic MLU590 reductions."""

from __future__ import annotations

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


# (mode, block size, num_warps).  Row4, atomic, and exact-single blocks are
# measured in rows; matrix-tiled blocks are measured in scalar elements.  The
# atomic candidates have a fixed stage count encoded in their public names.
_CHIP_CONFIGS = {
    "auto": ("row4", 128, 1),
    # KernelSwift's MLU590 run passed the structurally identical grid of eight
    # [256,4] programs at W4/S3. Keep that device evidence and remove masks only
    # generic/deterministic candidates stay explicit until measured on the same
    # runner; the generic masked arm is the only platform-proven local choice.
    "cambricon_mlu590": ("atomic", 256, 4),
}


def _get_launch_config():
    chip = os.environ.get("KERNELSWIFT_CHIP", "cambricon_mlu590").strip().lower()
    config = _CHIP_CONFIGS.get(chip, _CHIP_CONFIGS["auto"])
    if chip == "cambricon_mlu590":
        variants = {
            "row4_64x1": ("row4", 64, 1),
            "row4_64x4": ("row4", 64, 4),
            "row4_128x1": ("row4", 128, 1),
            "row4_128x4": ("row4", 128, 4),
            "row4_256x1": ("row4", 256, 1),
            "row4_exp2_64x1": ("row4_exp2", 64, 1),
            "row4_exp2_128x1": ("row4_exp2", 128, 1),
            "atomic256x4s3": ("atomic", 256, 4),
            "atomic256x4s3_exact": ("atomic_exact", 256, 4),
            "atomic256x4s3_packed": ("atomic_packed", 256, 4),
            "atomic512x4s3": ("atomic", 512, 4),
            "single256x1": ("single_exact", 256, 1),
            "single512x1": ("single_exact", 512, 1),
            "single512x4": ("single_exact", 512, 4),
            "single1024x1": ("single_exact", 1024, 1),
            "single1024x4": ("single_exact", 1024, 4),
            "single2048x1": ("single_exact", 2048, 1),
            "two_stage_exact64x1": ("two_stage_exact", 64, 1),
            "two_stage_exact128x1": ("two_stage_exact", 128, 1),
            "two_stage_exact256x1": ("two_stage_exact", 256, 1),
            "matrix_tiled_4096x1": ("matrix_tiled", 4096, 1),
            "matrix_tiled_8192x1": ("matrix_tiled", 8192, 1),
            "matrix_tiled_8192x4": ("matrix_tiled", 8192, 4),
        }
        key = (
            os.environ.get("KERNELSWIFT_TASK10_CONFIG", "atomic256x4s3")
            .strip()
            .lower()
        )
        config = variants.get(key, config)
    return config


@triton.jit
def _mix_bwd_row4_partial_kernel(
    input_ptr,
    scale_ptr,
    base_ptr,
    grad_out_ptr,
    grad_input_ptr,
    partial_ptr,
    n_rows: tl.constexpr,
    hc_mult: tl.constexpr,
    PARTIAL_WIDTH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    block_id = tl.program_id(0)
    row = block_id * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    channel_1d = tl.arange(0, hc_mult)
    channel = channel_1d[None, :]
    valid = row < n_rows
    offset = row * hc_mult + channel
    x = tl.load(input_ptr + offset, mask=valid, other=0.0).to(tl.float32)
    grad = tl.load(grad_out_ptr + offset, mask=valid, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    base = tl.load(base_ptr + channel).to(tl.float32)
    z = x * scale + base
    if USE_EXP2:
        sigmoid = 1.0 / (1.0 + tl.exp2(-z * 1.4426950408889634))
    else:
        sigmoid = 1.0 / (1.0 + tl.exp(-z))
    grad_z = grad * sigmoid * (1.0 - sigmoid)
    tl.store(grad_input_ptr + offset, grad_z * scale, mask=valid)

    partial_row = partial_ptr + block_id * PARTIAL_WIDTH
    scale_by_row = tl.sum(tl.where(valid, grad_z * x, 0.0), axis=1)
    tl.store(partial_row, tl.sum(scale_by_row, axis=0))
    base_partial = tl.sum(tl.where(valid, grad_z, 0.0), axis=0)
    tl.store(partial_row + 1 + channel_1d, base_partial)


@triton.jit
def _mix_bwd_finalize_kernel(
    partial_ptr,
    grad_scale_ptr,
    grad_base_ptr,
    num_blocks,
    hc_mult: tl.constexpr,
    PARTIAL_WIDTH: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
):
    block = tl.arange(0, REDUCE_BLOCK)
    valid = block < num_blocks
    row_offset = block * PARTIAL_WIDTH
    scale_partial = tl.load(partial_ptr + row_offset, mask=valid, other=0.0)
    tl.store(grad_scale_ptr, tl.sum(scale_partial, axis=0))
    for channel in tl.static_range(0, hc_mult):
        base_partial = tl.load(
            partial_ptr + row_offset + 1 + channel,
            mask=valid,
            other=0.0,
        )
        tl.store(grad_base_ptr + channel, tl.sum(base_partial, axis=0))


@triton.jit
def _mix_bwd_atomic_kernel(
    input_ptr,
    scale_ptr,
    base_ptr,
    grad_out_ptr,
    grad_input_ptr,
    grad_scale_ptr,
    grad_base_ptr,
    n_rows: tl.constexpr,
    hc_mult: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """One-launch 256-row tile with FP32 atomic parameter reductions."""
    block_id = tl.program_id(0)
    row = block_id * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    channel_1d = tl.arange(0, hc_mult)
    channel = channel_1d[None, :]
    valid = row < n_rows
    offset = row * hc_mult + channel

    x = tl.load(input_ptr + offset, mask=valid, other=0.0).to(tl.float32)
    grad = tl.load(grad_out_ptr + offset, mask=valid, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    base = tl.load(base_ptr + channel).to(tl.float32)
    z = x * scale + base
    sigmoid = 1.0 / (1.0 + tl.exp(-z))
    grad_z = grad * sigmoid * (1.0 - sigmoid)
    tl.store(grad_input_ptr + offset, grad_z * scale, mask=valid)

    scale_by_row = tl.sum(tl.where(valid, grad_z * x, 0.0), axis=1)
    scale_partial = tl.sum(scale_by_row, axis=0)
    base_partial = tl.sum(tl.where(valid, grad_z, 0.0), axis=0)
    tl.atomic_add(grad_scale_ptr, scale_partial)
    tl.atomic_add(grad_base_ptr + channel_1d, base_partial)


@triton.jit
def _mix_bwd_exact_atomic_kernel(
    input_ptr,
    scale_ptr,
    base_ptr,
    grad_out_ptr,
    grad_input_ptr,
    grad_scale_ptr,
    grad_base_ptr,
    BLOCK_ROWS: tl.constexpr,
):
    """Mask-free version of the platform-proven eight-program geometry."""
    block_id = tl.program_id(0)
    row = block_id * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    channel_1d = tl.arange(0, 4)
    channel = channel_1d[None, :]
    offset = row * 4 + channel
    x = tl.load(input_ptr + offset).to(tl.float32)
    grad = tl.load(grad_out_ptr + offset).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    base = tl.load(base_ptr + channel_1d).to(tl.float32)[None, :]
    z = x * scale + base
    sigmoid = 1.0 / (1.0 + tl.exp(-z))
    grad_z = grad * sigmoid * (1.0 - sigmoid)
    tl.store(grad_input_ptr + offset, grad_z * scale)

    scale_by_row = tl.sum(grad_z * x, axis=1)
    tl.atomic_add(grad_scale_ptr, tl.sum(scale_by_row, axis=0))
    tl.atomic_add(grad_base_ptr + channel_1d, tl.sum(grad_z, axis=0))


@triton.jit
def _mix_bwd_exact_single_kernel(
    input_ptr,
    scale_ptr,
    base_ptr,
    grad_out_ptr,
    grad_input_ptr,
    grad_scale_ptr,
    grad_base_ptr,
    NUM_TILES: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """One program owns the official contiguous ``[2048,4]`` matrix.

    The fixed task contract makes every tile exact, so this candidate has no
    bounds masks, atomics, partial workspace, or finalizer launch.  BLOCK_ROWS
    bounds the per-iteration NRAM live set; 256/512/1024/2048 are exposed for
    hardware measurement rather than selected from input values.
    """
    row_lane = tl.arange(0, BLOCK_ROWS)[:, None]
    channel_1d = tl.arange(0, 4)
    channel = channel_1d[None, :]
    scale = tl.load(scale_ptr).to(tl.float32)
    base = tl.load(base_ptr + channel_1d).to(tl.float32)[None, :]
    scale_total = 0.0
    base_total = tl.zeros((4,), dtype=tl.float32)

    for tile in tl.static_range(0, NUM_TILES):
        row = tile * BLOCK_ROWS + row_lane
        offset = row * 4 + channel
        x = tl.load(input_ptr + offset).to(tl.float32)
        grad = tl.load(grad_out_ptr + offset).to(tl.float32)
        z = x * scale + base
        sigmoid = 1.0 / (1.0 + tl.exp(-z))
        grad_z = grad * sigmoid * (1.0 - sigmoid)
        tl.store(grad_input_ptr + offset, grad_z * scale)

        scale_by_row = tl.sum(grad_z * x, axis=1)
        scale_total += tl.sum(scale_by_row, axis=0)
        base_total += tl.sum(grad_z, axis=0)

    tl.store(grad_scale_ptr, scale_total)
    tl.store(grad_base_ptr + channel_1d, base_total)


@triton.jit
def _mix_bwd_exact_partial_kernel(
    input_ptr,
    scale_ptr,
    base_ptr,
    grad_out_ptr,
    grad_input_ptr,
    partial_ptr,
    PARTIAL_WIDTH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """Exact-mask-free first stage for the official ``[2048,4]`` input."""
    block_id = tl.program_id(0)
    row = block_id * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    channel_1d = tl.arange(0, 4)
    channel = channel_1d[None, :]
    offset = row * 4 + channel
    x = tl.load(input_ptr + offset).to(tl.float32)
    grad = tl.load(grad_out_ptr + offset).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    base = tl.load(base_ptr + channel_1d).to(tl.float32)[None, :]
    z = x * scale + base
    sigmoid = 1.0 / (1.0 + tl.exp(-z))
    grad_z = grad * sigmoid * (1.0 - sigmoid)
    tl.store(grad_input_ptr + offset, grad_z * scale)

    partial_row = partial_ptr + block_id * PARTIAL_WIDTH
    scale_by_row = tl.sum(grad_z * x, axis=1)
    tl.store(partial_row, tl.sum(scale_by_row, axis=0))
    tl.store(partial_row + 1 + channel_1d, tl.sum(grad_z, axis=0))


@triton.jit
def _mix_bwd_matrix_tiled_kernel(
    input_ptr,
    scale_ptr,
    base_ptr,
    grad_out_ptr,
    grad_input_ptr,
    grad_scale_ptr,
    grad_base_ptr,
    n_rows: tl.constexpr,
    hc_mult: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """One-program alternative that removes the partial-buffer launch."""
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
        grad = tl.load(grad_out_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        base = tl.load(base_ptr + channels).to(tl.float32)
        z = x * scale + base
        sigmoid = 1.0 / (1.0 + tl.exp(-z))
        grad_z = grad * sigmoid * (1.0 - sigmoid)
        tl.store(grad_input_ptr + offsets, grad_z * scale, mask=valid)
        scale_by_row = tl.sum(tl.where(valid, grad_z * x, 0.0), axis=1)
        scale_total += tl.sum(scale_by_row, axis=0)
        base_total += tl.sum(tl.where(valid, grad_z, 0.0), axis=0)
    tl.store(grad_scale_ptr, scale_total)
    tl.store(grad_base_ptr + channels, base_total)


def head_compute_mix_bwd(
    input_mix,
    mhc_scale,
    mhc_base,
    grad_out,
    kernel_config,
):
    mode, block, num_warps = kernel_config
    tensors = (input_mix, mhc_scale, mhc_base, grad_out)
    if any(t.device.type == "cpu" for t in tensors):
        raise ValueError("Task10 requires accelerator tensors supported by Triton")
    if not all(t.device == input_mix.device for t in tensors):
        raise ValueError("all inputs must be on the same accelerator")
    hc_mult = input_mix.shape[-1]
    if hc_mult != 4:
        raise ValueError("the official Task10 contract fixes mhc_mult=4")
    if grad_out.shape != input_mix.shape:
        raise ValueError("grad_out must have the same shape as input_mix")

    exact_official = (
        input_mix.shape == (2, 1024, 4)
        and grad_out.shape == (2, 1024, 4)
        and mhc_scale.shape == (1,)
        and mhc_base.shape == (4,)
        and input_mix.is_contiguous()
        and grad_out.is_contiguous()
        and mhc_scale.is_contiguous()
        and mhc_base.is_contiguous()
    )
    if (
        mode in ("atomic_exact", "single_exact", "two_stage_exact")
        and not exact_official
    ):
        raise ValueError(
            "exact Task10 candidates require contiguous [2,1024,4], [1], [4]"
        )

    grad_input = torch.empty_like(input_mix)
    n_rows = input_mix.numel() // hc_mult

    if mode in ("atomic", "atomic_exact", "atomic_packed"):
        # Atomic accumulation requires identity initialization.  No other
        # candidate pays for this zero-fill.  The packed variant is a host-side
        # allocation experiment; the platform-matched default keeps the two
        # independently zeroed outputs used by the passing generated program.
        if mode == "atomic_packed":
            reduction = torch.zeros(
                (1 + hc_mult,),
                device=input_mix.device,
                dtype=input_mix.dtype,
            )
            grad_scale = reduction[:1].view_as(mhc_scale)
            grad_base = reduction[1:].view_as(mhc_base)
        else:
            grad_scale = torch.zeros_like(mhc_scale)
            grad_base = torch.zeros_like(mhc_base)
        num_blocks = triton.cdiv(n_rows, block)
        if mode == "atomic_exact":
            _mix_bwd_exact_atomic_kernel[(num_blocks,)](
                input_mix,
                mhc_scale,
                mhc_base,
                grad_out,
                grad_input,
                grad_scale,
                grad_base,
                BLOCK_ROWS=block,
                num_warps=num_warps,
                num_stages=3,
            )
        else:
            _mix_bwd_atomic_kernel[(num_blocks,)](
                input_mix,
                mhc_scale,
                mhc_base,
                grad_out,
                grad_input,
                grad_scale,
                grad_base,
                n_rows=n_rows,
                hc_mult=hc_mult,
                BLOCK_ROWS=block,
                num_warps=num_warps,
                num_stages=3,
            )
        return grad_input, grad_scale, grad_base

    grad_scale = torch.empty_like(mhc_scale)
    grad_base = torch.empty_like(mhc_base)

    if mode == "single_exact":
        _mix_bwd_exact_single_kernel[(1,)](
            input_mix,
            mhc_scale,
            mhc_base,
            grad_out,
            grad_input,
            grad_scale,
            grad_base,
            NUM_TILES=2048 // block,
            BLOCK_ROWS=block,
            num_warps=num_warps,
            num_stages=1,
        )
        return grad_input, grad_scale, grad_base

    if mode == "two_stage_exact":
        num_blocks = 2048 // block
        partial_width = 1 + hc_mult
        partial = torch.empty(
            (num_blocks, partial_width),
            device=input_mix.device,
            dtype=torch.float32,
        )
        _mix_bwd_exact_partial_kernel[(num_blocks,)](
            input_mix,
            mhc_scale,
            mhc_base,
            grad_out,
            grad_input,
            partial,
            PARTIAL_WIDTH=partial_width,
            BLOCK_ROWS=block,
            num_warps=num_warps,
            num_stages=1,
        )
        _mix_bwd_finalize_kernel[(1,)](
            partial,
            grad_scale,
            grad_base,
            num_blocks,
            hc_mult=hc_mult,
            PARTIAL_WIDTH=partial_width,
            REDUCE_BLOCK=triton.next_power_of_2(num_blocks),
            num_warps=1,
            num_stages=1,
        )
        return grad_input, grad_scale, grad_base

    if mode == "matrix_tiled":
        block_rows = block // hc_mult
        num_tiles = triton.cdiv(n_rows, block_rows)
        _mix_bwd_matrix_tiled_kernel[(1,)](
            input_mix,
            mhc_scale,
            mhc_base,
            grad_out,
            grad_input,
            grad_scale,
            grad_base,
            n_rows=n_rows,
            hc_mult=hc_mult,
            NUM_TILES=num_tiles,
            BLOCK_ROWS=block_rows,
            num_warps=num_warps,
            num_stages=1,
        )
        return grad_input, grad_scale, grad_base

    num_blocks = triton.cdiv(n_rows, block)
    partial_width = 1 + hc_mult
    partial = torch.empty(
        (num_blocks, partial_width),
        device=input_mix.device,
        dtype=torch.float32,
    )
    _mix_bwd_row4_partial_kernel[(num_blocks,)](
        input_mix,
        mhc_scale,
        mhc_base,
        grad_out,
        grad_input,
        partial,
        n_rows=n_rows,
        hc_mult=hc_mult,
        PARTIAL_WIDTH=partial_width,
        BLOCK_ROWS=block,
        USE_EXP2=mode == "row4_exp2",
        num_warps=num_warps,
        num_stages=1,
    )
    _mix_bwd_finalize_kernel[(1,)](
        partial,
        grad_scale,
        grad_base,
        num_blocks,
        hc_mult=hc_mult,
        PARTIAL_WIDTH=partial_width,
        REDUCE_BLOCK=triton.next_power_of_2(num_blocks),
        num_warps=1,
        num_stages=1,
    )
    return grad_input, grad_scale, grad_base


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._kernel_config = _get_launch_config()

    def forward(self, input_mix, mhc_scale, mhc_base, grad_out):
        return head_compute_mix_bwd(
            input_mix,
            mhc_scale,
            mhc_base,
            grad_out,
            self._kernel_config,
        )


class Model(ModelNew):
    """Compatibility name for runners that import ``Model``."""


batch0 = 2
batch1 = 1024
mhc_mult = 4


def get_inputs():
    device = _target_device()
    return [
        torch.randn(batch0, batch1, mhc_mult, dtype=torch.float32, device=device),
        torch.randn(1, dtype=torch.float32, device=device),
        torch.randn(mhc_mult, dtype=torch.float32, device=device),
        torch.randn(batch0, batch1, mhc_mult, dtype=torch.float32, device=device),
    ]


def get_init_inputs():
    return []
