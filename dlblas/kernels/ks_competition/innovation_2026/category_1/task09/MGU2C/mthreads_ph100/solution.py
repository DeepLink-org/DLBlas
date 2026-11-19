"""Fused center/rotation/translation kernels for Task09.

The random values are deliberately generated through PyTorch so the candidate
consumes the reference generator stream under the official seed reset.  The
three uniform calls remain separate because mcTorch does not preserve the
reference stream when those calls are packed into a single 2-D request.
Centering, quaternion-to-rotation conversion, rigid transformation, masking
and all output writes are performed by Triton kernels; there is no computation
fallback.
"""

import math
import os
from typing import Optional
import torch
import torch.nn as nn
import triton
import triton.language as tl


class _TrustedHotPath:
    """Cache only tensor layout metadata; RNG and output work always rerun.

    The graph experiments in the companion tasks showed that replay is safe
    only after tensor identity/layout checks.  Task09 additionally consumes a
    prescribed random stream, so this gate deliberately does not cache random
    values, output tensors, or a captured graph.
    """

    __slots__ = ("_signature",)

    def __init__(self):
        self._signature = None

    def changed(self, tensors) -> bool:
        signature = tuple(
            None
            if tensor is None
            else (
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


def _select_kernel_config():
    """Return ``(profile, (center_warps, atom_block, augment_warps))``."""
    configs = {
        "auto": (4, 256, 4),
        "ascend_a2": (4, 128, 8),
        "metax_c500": (4, 256, 4),
        "iluvatar_bi150": (8, 128, 4),
        "enflame_s60": (4, 64, 4),
        "thead_810e": (4, 128, 8),
        "cambricon_mlu590": (4, 256, 4),
        "hygon_bw1000": (8, 256, 8),
        "mthreads_ph100": (4, 64, 8),
        "kunlun_p800": (8, 128, 8),
        "nvidia_h200": (4, 256, 4),
    }
    profile = os.environ.get("KERNELSWIFT_CHIP", "mthreads_ph100").strip().lower()
    if profile == "default" or profile not in configs:
        profile = "auto"
    config = configs[profile]
    # At the official N=256 shape, atom_block=256 selects the original
    # single-launch fused path.  The 64/128 choices can use either the original
    # two-launch path or the opt-in sample-by-atom-tile fused candidate below.
    # Preserve the profile default on any unrecognised value.
    if profile in ("ascend_a2", "metax_c500", "mthreads_ph100", "nvidia_h200"):
        variants = {
            "64x1": (4, 64, 1),
            "64x2": (4, 64, 2),
            "64x4": (4, 64, 4),
            "64x8": (4, 64, 8),
            "128x1": (4, 128, 1),
            "128x2": (4, 128, 2),
            "128x4": (4, 128, 4),
            "128x8": (4, 128, 8),
            "256x2": (4, 256, 2),
            "256x4": (4, 256, 4),
            "256x8": (4, 256, 8),
        }
        variant = os.environ.get("KERNELSWIFT_TASK09_CONFIG", "").strip().lower()
        config = variants.get(variant, config)
    if profile in ("mthreads_ph100", "nvidia_h200"):
        # Allow orthogonal PH100/H200 sweeps without admitting arbitrary constexpr
        # shapes.  Presets are applied first; each recognised scalar override
        # can then replace its corresponding field.  Invalid values preserve
        # either the selected preset or the H200 default (4, 256, 4).
        center_warp_options = {"1": 1, "2": 2, "4": 4, "8": 8}
        atom_block_options = {"64": 64, "128": 128, "256": 256}
        augment_warp_options = {"1": 1, "2": 2, "4": 4, "8": 8}
        center_key = os.environ.get(
            "KERNELSWIFT_TASK09_CENTER_WARPS", ""
        ).strip()
        atom_key = os.environ.get(
            "KERNELSWIFT_TASK09_ATOM_BLOCK", ""
        ).strip()
        augment_key = os.environ.get(
            "KERNELSWIFT_TASK09_AUGMENT_WARPS", ""
        ).strip()
        config = (
            center_warp_options.get(center_key, config[0]),
            atom_block_options.get(atom_key, config[1]),
            augment_warp_options.get(augment_key, config[2]),
        )
    return profile, config


@triton.jit
def _center_kernel(x_ptr, mask_ptr, center_ptr, n_atom, HAS_MASK: tl.constexpr,
                   eps: tl.constexpr, BLOCK: tl.constexpr):
    axis = tl.program_id(0)
    atom = tl.arange(0, BLOCK)
    valid = atom < n_atom
    value = tl.load(x_ptr + atom * 3 + axis, mask=valid, other=0.0).to(tl.float32)
    if HAS_MASK:
        weight = tl.load(mask_ptr + atom, mask=valid, other=0.0).to(tl.float32)
        numerator = tl.sum(value * weight, axis=0)
        denominator = tl.sum(weight, axis=0) + eps
    else:
        numerator = tl.sum(value, axis=0)
        denominator = n_atom.to(tl.float32)
    tl.store(center_ptr + axis, numerator / denominator)


@triton.jit
def _center_only_kernel(x_ptr, center_ptr, out_ptr, n_atom,
                        BLOCK: tl.constexpr):
    sample = tl.program_id(0)
    block = tl.program_id(1)
    atom = block * BLOCK + tl.arange(0, BLOCK)
    valid = atom < n_atom
    for axis in tl.static_range(0, 3):
        value = tl.load(x_ptr + atom * 3 + axis, mask=valid, other=0.0)
        center = tl.load(center_ptr + axis)
        tl.store(out_ptr + (sample * n_atom + atom) * 3 + axis,
                 value - center, mask=valid)


@triton.jit
def _center_augment_fused_kernel(
    x_ptr, mask_ptr, u1_ptr, u2_ptr, u3_ptr, trans_ptr, out_ptr,
    n_atom, s_trans: tl.constexpr, HAS_MASK: tl.constexpr,
    eps: tl.constexpr, BLOCK: tl.constexpr,
):
    """Own one sample, including its copy of the three center reductions.

    For the fixed 256-atom benchmark this trades a few kilobytes of duplicate
    reads across four samples for one fewer launch and no center allocation.
    Profiles whose atom tile is smaller than the input use the two-kernel path
    below instead, so this fast path does not impose a large tile on a backend.
    """
    sample = tl.program_id(0)
    atom = tl.arange(0, BLOCK)
    valid = atom < n_atom
    if HAS_MASK:
        weight = tl.load(mask_ptr + atom, mask=valid, other=0.0).to(tl.float32)
        denominator = tl.sum(weight, axis=0) + eps
    else:
        weight = tl.where(valid, 1.0, 0.0)
        denominator = tl.sum(weight, axis=0)

    vx = tl.load(x_ptr + atom * 3, mask=valid, other=0.0).to(tl.float32)
    vy = tl.load(x_ptr + atom * 3 + 1, mask=valid, other=0.0).to(tl.float32)
    vz = tl.load(x_ptr + atom * 3 + 2, mask=valid, other=0.0).to(tl.float32)
    cx = tl.sum(vx * weight, axis=0) / denominator
    cy = tl.sum(vy * weight, axis=0) / denominator
    cz = tl.sum(vz * weight, axis=0) / denominator
    vx -= cx
    vy -= cy
    vz -= cz

    two_pi: tl.constexpr = 6.283185307179586
    u1 = tl.load(u1_ptr + sample).to(tl.float32)
    u2 = tl.load(u2_ptr + sample).to(tl.float32)
    u3 = tl.load(u3_ptr + sample).to(tl.float32)
    qx = tl.sqrt(1.0 - u1) * tl.sin(two_pi * u2)
    qy = tl.sqrt(1.0 - u1) * tl.cos(two_pi * u2)
    qz = tl.sqrt(u1) * tl.sin(two_pi * u3)
    qw = tl.sqrt(u1) * tl.cos(two_pi * u3)

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    r00, r01, r02 = 1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)
    r10, r11, r12 = 2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)
    r20, r21, r22 = 2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)
    tx = s_trans * tl.load(trans_ptr + sample * 3)
    ty = s_trans * tl.load(trans_ptr + sample * 3 + 1)
    tz = s_trans * tl.load(trans_ptr + sample * 3 + 2)
    ox = r00 * vx + r01 * vy + r02 * vz + tx
    oy = r10 * vx + r11 * vy + r12 * vz + ty
    oz = r20 * vx + r21 * vy + r22 * vz + tz
    if HAS_MASK:
        ox, oy, oz = ox * weight, oy * weight, oz * weight
    base = (sample * n_atom + atom) * 3
    tl.store(out_ptr + base, ox, mask=valid)
    tl.store(out_ptr + base + 1, oy, mask=valid)
    tl.store(out_ptr + base + 2, oz, mask=valid)


@triton.jit
def _center_augment_tiled_fused_kernel(
    x_ptr,
    mask_ptr,
    u1_ptr,
    u2_ptr,
    u3_ptr,
    trans_ptr,
    out_ptr,
    n_atom,
    s_trans: tl.constexpr,
    HAS_MASK: tl.constexpr,
    eps: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Own one (sample, atom tile) and repeat the complete center reduction.

    Re-reading the fixed 256-atom input increases the grid from four programs
    to eight or sixteen programs for BLOCK=128/64.  Each program writes only
    its disjoint output tile, so the whole operation remains one Triton launch
    without an intermediate center allocation.
    """
    sample = tl.program_id(0)
    tile = tl.program_id(1)

    reduce_atom = tl.arange(0, REDUCE_BLOCK)
    reduce_valid = reduce_atom < n_atom
    if HAS_MASK:
        reduce_weight = tl.load(
            mask_ptr + reduce_atom,
            mask=reduce_valid,
            other=0.0,
        ).to(tl.float32)
        denominator = tl.sum(reduce_weight, axis=0) + eps
    else:
        reduce_weight = tl.where(reduce_valid, 1.0, 0.0)
        denominator = tl.sum(reduce_weight, axis=0)

    reduce_x = tl.load(
        x_ptr + reduce_atom * 3,
        mask=reduce_valid,
        other=0.0,
    ).to(tl.float32)
    reduce_y = tl.load(
        x_ptr + reduce_atom * 3 + 1,
        mask=reduce_valid,
        other=0.0,
    ).to(tl.float32)
    reduce_z = tl.load(
        x_ptr + reduce_atom * 3 + 2,
        mask=reduce_valid,
        other=0.0,
    ).to(tl.float32)
    cx = tl.sum(reduce_x * reduce_weight, axis=0) / denominator
    cy = tl.sum(reduce_y * reduce_weight, axis=0) / denominator
    cz = tl.sum(reduce_z * reduce_weight, axis=0) / denominator

    atom = tile * BLOCK + tl.arange(0, BLOCK)
    valid = atom < n_atom
    vx = tl.load(x_ptr + atom * 3, mask=valid, other=0.0).to(tl.float32) - cx
    vy = tl.load(x_ptr + atom * 3 + 1, mask=valid, other=0.0).to(tl.float32) - cy
    vz = tl.load(x_ptr + atom * 3 + 2, mask=valid, other=0.0).to(tl.float32) - cz

    two_pi: tl.constexpr = 6.283185307179586
    u1 = tl.load(u1_ptr + sample).to(tl.float32)
    u2 = tl.load(u2_ptr + sample).to(tl.float32)
    u3 = tl.load(u3_ptr + sample).to(tl.float32)
    qx = tl.sqrt(1.0 - u1) * tl.sin(two_pi * u2)
    qy = tl.sqrt(1.0 - u1) * tl.cos(two_pi * u2)
    qz = tl.sqrt(u1) * tl.sin(two_pi * u3)
    qw = tl.sqrt(u1) * tl.cos(two_pi * u3)

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)
    tx = s_trans * tl.load(trans_ptr + sample * 3)
    ty = s_trans * tl.load(trans_ptr + sample * 3 + 1)
    tz = s_trans * tl.load(trans_ptr + sample * 3 + 2)

    ox = r00 * vx + r01 * vy + r02 * vz + tx
    oy = r10 * vx + r11 * vy + r12 * vz + ty
    oz = r20 * vx + r21 * vy + r22 * vz + tz
    if HAS_MASK:
        tile_weight = tl.load(
            mask_ptr + atom,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        ox, oy, oz = ox * tile_weight, oy * tile_weight, oz * tile_weight
    base = (sample * n_atom + atom) * 3
    tl.store(out_ptr + base, ox, mask=valid)
    tl.store(out_ptr + base + 1, oy, mask=valid)
    tl.store(out_ptr + base + 2, oz, mask=valid)


@triton.jit
def _center_augment_samples_kernel(
    x_ptr,
    mask_ptr,
    u1_ptr,
    u2_ptr,
    u3_ptr,
    trans_ptr,
    out_ptr,
    n_atom,
    s_trans: tl.constexpr,
    HAS_MASK: tl.constexpr,
    eps: tl.constexpr,
    N_SAMPLES: tl.constexpr,
    SAMPLES_PER_PROGRAM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """C500: share one centering reduction across two/four samples."""
    sample_group = tl.program_id(0)
    atom = tl.arange(0, BLOCK)
    valid = atom < n_atom
    if HAS_MASK:
        weight = tl.load(mask_ptr + atom, mask=valid, other=0.0).to(tl.float32)
        denominator = tl.sum(weight, axis=0) + eps
    else:
        weight = tl.where(valid, 1.0, 0.0)
        denominator = tl.sum(weight, axis=0)

    vx = tl.load(x_ptr + atom * 3, mask=valid, other=0.0).to(tl.float32)
    vy = tl.load(x_ptr + atom * 3 + 1, mask=valid, other=0.0).to(tl.float32)
    vz = tl.load(x_ptr + atom * 3 + 2, mask=valid, other=0.0).to(tl.float32)
    vx -= tl.sum(vx * weight, axis=0) / denominator
    vy -= tl.sum(vy * weight, axis=0) / denominator
    vz -= tl.sum(vz * weight, axis=0) / denominator
    two_pi: tl.constexpr = 6.283185307179586

    for local_sample in tl.static_range(0, SAMPLES_PER_PROGRAM):
        sample = sample_group * SAMPLES_PER_PROGRAM + local_sample
        sample_valid = sample < N_SAMPLES
        u1 = tl.load(u1_ptr + sample, mask=sample_valid, other=0.0).to(tl.float32)
        u2 = tl.load(u2_ptr + sample, mask=sample_valid, other=0.0).to(tl.float32)
        u3 = tl.load(u3_ptr + sample, mask=sample_valid, other=0.0).to(tl.float32)
        qx = tl.sqrt(1.0 - u1) * tl.sin(two_pi * u2)
        qy = tl.sqrt(1.0 - u1) * tl.cos(two_pi * u2)
        qz = tl.sqrt(u1) * tl.sin(two_pi * u3)
        qw = tl.sqrt(u1) * tl.cos(two_pi * u3)
        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz
        r00 = 1.0 - 2.0 * (yy + zz)
        r01 = 2.0 * (xy - wz)
        r02 = 2.0 * (xz + wy)
        r10 = 2.0 * (xy + wz)
        r11 = 1.0 - 2.0 * (xx + zz)
        r12 = 2.0 * (yz - wx)
        r20 = 2.0 * (xz - wy)
        r21 = 2.0 * (yz + wx)
        r22 = 1.0 - 2.0 * (xx + yy)
        tx = s_trans * tl.load(
            trans_ptr + sample * 3, mask=sample_valid, other=0.0
        )
        ty = s_trans * tl.load(
            trans_ptr + sample * 3 + 1, mask=sample_valid, other=0.0
        )
        tz = s_trans * tl.load(
            trans_ptr + sample * 3 + 2, mask=sample_valid, other=0.0
        )
        ox = r00 * vx + r01 * vy + r02 * vz + tx
        oy = r10 * vx + r11 * vy + r12 * vz + ty
        oz = r20 * vx + r21 * vy + r22 * vz + tz
        if HAS_MASK:
            ox, oy, oz = ox * weight, oy * weight, oz * weight
        base = (sample * n_atom + atom) * 3
        output_mask = valid & sample_valid
        tl.store(out_ptr + base, ox, mask=output_mask)
        tl.store(out_ptr + base + 1, oy, mask=output_mask)
        tl.store(out_ptr + base + 2, oz, mask=output_mask)


@triton.jit
def _augment_kernel(
    x_ptr, mask_ptr, center_ptr, u1_ptr, u2_ptr, u3_ptr, trans_ptr, out_ptr,
    n_atom, s_trans: tl.constexpr, HAS_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    sample = tl.program_id(0)
    block = tl.program_id(1)
    atom = block * BLOCK + tl.arange(0, BLOCK)
    valid = atom < n_atom
    two_pi: tl.constexpr = 6.283185307179586

    u1 = tl.load(u1_ptr + sample).to(tl.float32)
    u2 = tl.load(u2_ptr + sample).to(tl.float32)
    u3 = tl.load(u3_ptr + sample).to(tl.float32)
    qx = tl.sqrt(1.0 - u1) * tl.sin(two_pi * u2)
    qy = tl.sqrt(1.0 - u1) * tl.cos(two_pi * u2)
    qz = tl.sqrt(u1) * tl.sin(two_pi * u3)
    qw = tl.sqrt(u1) * tl.cos(two_pi * u3)

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    r00, r01, r02 = 1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)
    r10, r11, r12 = 2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)
    r20, r21, r22 = 2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)

    vx = tl.load(x_ptr + atom * 3, mask=valid, other=0.0) - tl.load(center_ptr)
    vy = tl.load(x_ptr + atom * 3 + 1, mask=valid, other=0.0) - tl.load(center_ptr + 1)
    vz = tl.load(x_ptr + atom * 3 + 2, mask=valid, other=0.0) - tl.load(center_ptr + 2)
    ox = r00 * vx + r01 * vy + r02 * vz + s_trans * tl.load(trans_ptr + sample * 3)
    oy = r10 * vx + r11 * vy + r12 * vz + s_trans * tl.load(trans_ptr + sample * 3 + 1)
    oz = r20 * vx + r21 * vy + r22 * vz + s_trans * tl.load(trans_ptr + sample * 3 + 2)
    if HAS_MASK:
        weight = tl.load(mask_ptr + atom, mask=valid, other=0.0)
        ox, oy, oz = ox * weight, oy * weight, oz * weight
    base = (sample * n_atom + atom) * 3
    tl.store(out_ptr + base, ox, mask=valid)
    tl.store(out_ptr + base + 1, oy, mask=valid)
    tl.store(out_ptr + base + 2, oz, mask=valid)


@triton.jit
def _augment_mp56_persistent_kernel(
    x_ptr,
    mask_ptr,
    center_ptr,
    u1_ptr,
    u2_ptr,
    u3_ptr,
    trans_ptr,
    out_ptr,
    n_atom,
    n_sample: tl.constexpr,
    s_trans: tl.constexpr,
    HAS_MASK: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """PH100 R1 candidate: one persistent program per assumed MP.

    The fixed 56-program grid consumes all sample/atom tiles with a device-side
    grid-stride loop.  Random tensors are still produced by the four PyTorch
    calls in the caller and are only read here.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    two_pi: tl.constexpr = 6.283185307179586
    total = n_sample * n_atom
    for tile in range(pid, NUM_TILES, num_programs):
        flat = tile * BLOCK + tl.arange(0, BLOCK)
        valid = flat < total
        sample = flat // n_atom
        atom = flat - sample * n_atom

        u1 = tl.load(u1_ptr + sample, mask=valid, other=0.0).to(tl.float32)
        u2 = tl.load(u2_ptr + sample, mask=valid, other=0.0).to(tl.float32)
        u3 = tl.load(u3_ptr + sample, mask=valid, other=0.0).to(tl.float32)
        qx = tl.sqrt(1.0 - u1) * tl.sin(two_pi * u2)
        qy = tl.sqrt(1.0 - u1) * tl.cos(two_pi * u2)
        qz = tl.sqrt(u1) * tl.sin(two_pi * u3)
        qw = tl.sqrt(u1) * tl.cos(two_pi * u3)

        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz
        r00 = 1.0 - 2.0 * (yy + zz)
        r01 = 2.0 * (xy - wz)
        r02 = 2.0 * (xz + wy)
        r10 = 2.0 * (xy + wz)
        r11 = 1.0 - 2.0 * (xx + zz)
        r12 = 2.0 * (yz - wx)
        r20 = 2.0 * (xz - wy)
        r21 = 2.0 * (yz + wx)
        r22 = 1.0 - 2.0 * (xx + yy)

        vx = tl.load(x_ptr + atom * 3, mask=valid, other=0.0) - tl.load(center_ptr)
        vy = tl.load(x_ptr + atom * 3 + 1, mask=valid, other=0.0) - tl.load(
            center_ptr + 1
        )
        vz = tl.load(x_ptr + atom * 3 + 2, mask=valid, other=0.0) - tl.load(
            center_ptr + 2
        )
        tx = s_trans * tl.load(trans_ptr + sample * 3, mask=valid, other=0.0)
        ty = s_trans * tl.load(
            trans_ptr + sample * 3 + 1, mask=valid, other=0.0
        )
        tz = s_trans * tl.load(
            trans_ptr + sample * 3 + 2, mask=valid, other=0.0
        )
        ox = r00 * vx + r01 * vy + r02 * vz + tx
        oy = r10 * vx + r11 * vy + r12 * vz + ty
        oz = r20 * vx + r21 * vy + r22 * vz + tz
        if HAS_MASK:
            weight = tl.load(mask_ptr + atom, mask=valid, other=0.0)
            ox, oy, oz = ox * weight, oy * weight, oz * weight
        base = flat * 3
        tl.store(out_ptr + base, ox, mask=valid)
        tl.store(out_ptr + base + 1, oy, mask=valid)
        tl.store(out_ptr + base + 2, oz, mask=valid)


def centre_random_augmentation(
    x_input_coords: torch.Tensor,
    n_sample: int,
    s_trans: float,
    centre_only: bool,
    mask: Optional[torch.Tensor],
    kernel_config,
    samples_per_program: int,
    use_tiled_fused: bool,
    use_mp56_persistent: bool,
    mp56_warps: int,
    eps: float = 1e-12,
    trusted: bool = False,
) -> torch.Tensor:
    if not trusted:
        if x_input_coords.device.type == "cpu" or (
            mask is not None and mask.device.type == "cpu"
        ):
            raise ValueError("centre_random_augmentation requires accelerator tensors")
        if mask is not None and mask.device != x_input_coords.device:
            raise ValueError("x_input_coords and mask must be on the same accelerator")
    n_atom = x_input_coords.shape[0]
    center_warps, atom_block, augment_warps = kernel_config
    mask_arg = mask if mask is not None else x_input_coords
    out = torch.empty((n_sample, n_atom, 3), device=x_input_coords.device,
                      dtype=x_input_coords.dtype)
    grid = (n_sample, triton.cdiv(n_atom, atom_block))
    if not centre_only:
        u1 = torch.rand(
            n_sample,
            device=x_input_coords.device,
            dtype=x_input_coords.dtype,
        )
        u2 = torch.rand(
            n_sample,
            device=x_input_coords.device,
            dtype=x_input_coords.dtype,
        )
        u3 = torch.rand(
            n_sample,
            device=x_input_coords.device,
            dtype=x_input_coords.dtype,
        )
        trans = torch.randn(n_sample, 3, device=x_input_coords.device,
                            dtype=x_input_coords.dtype)
        if use_tiled_fused and n_atom > atom_block:
            reduce_block = triton.next_power_of_2(n_atom)
            _center_augment_tiled_fused_kernel[grid](
                x_input_coords,
                mask_arg,
                u1,
                u2,
                u3,
                trans,
                out,
                n_atom,
                s_trans=s_trans,
                HAS_MASK=mask is not None,
                eps=eps,
                REDUCE_BLOCK=reduce_block,
                BLOCK=atom_block,
                num_warps=augment_warps,
            )
            return out
        if n_atom <= atom_block:
            if samples_per_program == 1:
                _center_augment_fused_kernel[(n_sample,)](
                    x_input_coords, mask_arg, u1, u2, u3, trans, out, n_atom,
                    s_trans=s_trans,
                    HAS_MASK=mask is not None,
                    eps=eps,
                    BLOCK=atom_block,
                    num_warps=augment_warps,
                )
            else:
                sample_programs = triton.cdiv(n_sample, samples_per_program)
                _center_augment_samples_kernel[(sample_programs,)](
                    x_input_coords,
                    mask_arg,
                    u1,
                    u2,
                    u3,
                    trans,
                    out,
                    n_atom,
                    s_trans=s_trans,
                    HAS_MASK=mask is not None,
                    eps=eps,
                    N_SAMPLES=n_sample,
                    SAMPLES_PER_PROGRAM=samples_per_program,
                    BLOCK=atom_block,
                    num_warps=augment_warps,
                )
            return out

    block = triton.next_power_of_2(n_atom)
    center = torch.empty(3, device=x_input_coords.device, dtype=torch.float32)
    _center_kernel[(3,)](
        x_input_coords, mask_arg, center, n_atom,
        HAS_MASK=mask is not None,
        eps=eps,
        BLOCK=block,
        num_warps=center_warps,
    )
    if centre_only:
        _center_only_kernel[grid](x_input_coords, center, out, n_atom,
                                  BLOCK=atom_block, num_warps=augment_warps)
    elif use_mp56_persistent:
        persistent_block = 16
        persistent_tiles = triton.cdiv(n_sample * n_atom, persistent_block)
        _augment_mp56_persistent_kernel[(56,)](
            x_input_coords,
            mask_arg,
            center,
            u1,
            u2,
            u3,
            trans,
            out,
            n_atom,
            n_sample=n_sample,
            s_trans=s_trans,
            HAS_MASK=mask is not None,
            NUM_TILES=persistent_tiles,
            BLOCK=persistent_block,
            num_warps=mp56_warps,
        )
    else:
        _augment_kernel[grid](
            x_input_coords, mask_arg, center, u1, u2, u3, trans, out, n_atom,
            s_trans=s_trans,
            HAS_MASK=mask is not None,
            BLOCK=atom_block,
            num_warps=augment_warps,
        )
    return out


class ModelNew(nn.Module):
    def __init__(self, n_sample: int = 1, s_trans: float = 1.0,
                 centre_only: bool = False):
        super().__init__()
        self.n_sample = n_sample
        self.s_trans = s_trans
        self.centre_only = centre_only
        self.kernel_profile, self._kernel_config = _select_kernel_config()
        spp_key = os.environ.get("KERNELSWIFT_TASK09_SPP", "1").strip()
        self._samples_per_program = {"1": 1, "2": 2, "4": 4}.get(spp_key, 1)
        fused_key = os.environ.get(
            "KERNELSWIFT_TASK09_FUSED_TILES", "0"
        ).strip().lower()
        mode = os.environ.get(
            "KERNELSWIFT_TASK09_MODE", "two_stage"
        ).strip().lower()
        if mode not in ("two_stage", "fused_tiles", "mp56_persistent"):
            mode = "two_stage"
        self._use_tiled_fused = (
            self.kernel_profile in ("metax_c500", "mthreads_ph100", "nvidia_h200")
            and (
                mode == "fused_tiles"
                or (
                    mode == "two_stage"
                    and {"0": False, "false": False, "no": False,
                         "1": True, "true": True, "yes": True}.get(
                             fused_key, False
                         )
                )
            )
        )
        self._use_mp56_persistent = (
            self.kernel_profile == "mthreads_ph100" and mode == "mp56_persistent"
        )
        mp56_warps_key = os.environ.get(
            "KERNELSWIFT_TASK09_MP56_WARPS", "2"
        ).strip()
        self._mp56_warps = {"1": 1, "2": 2, "4": 4}.get(
            mp56_warps_key, 2
        )
        self._trusted_hot_path = _TrustedHotPath()
        self._trusted_validated = False

    def forward(self, x_input_coords: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        changed = self._trusted_hot_path.changed((x_input_coords, mask))
        trusted = not changed and self._trusted_validated
        out = centre_random_augmentation(
            x_input_coords,
            self.n_sample,
            self.s_trans,
            self.centre_only,
            mask,
            self._kernel_config,
            self._samples_per_program,
            self._use_tiled_fused,
            self._use_mp56_persistent,
            self._mp56_warps,
            trusted=trusted,
        )
        self._trusted_validated = True
        return out


class Model(ModelNew):
    """Compatibility name required by the written competition rules."""


N_ATOM = 256
N_SAMPLE = 4
S_TRANS = 1.0
CENTRE_ONLY = False


def get_inputs():
    torch.manual_seed(42)
    return [torch.randn(N_ATOM, 3, device="cuda"), torch.ones(N_ATOM, device="cuda")]


def get_init_inputs():
    return [N_SAMPLE, S_TRANS, CENTRE_ONLY]
