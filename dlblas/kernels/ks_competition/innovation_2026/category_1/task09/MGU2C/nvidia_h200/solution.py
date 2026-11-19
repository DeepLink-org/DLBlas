"""Fused center/rotation/translation kernels for Task09.

The fixed H200 competition shape reconstructs PyTorch CUDA's four Philox calls
bit-for-bit inside the fused Triton kernel while reserving the same generator
offset.  Unsupported shapes and profiles retain the separate PyTorch RNG path.
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
    profile = os.environ.get("KERNELSWIFT_CHIP", "nvidia_h200").strip().lower()
    if profile == "default" or profile not in configs:
        profile = "auto"
    config = configs[profile]
    # At the official N=256 shape, atom_block=256 selects the original
    # single-launch fused path.  The 64/128 choices can use either the original
    # two-launch path or the opt-in sample-by-atom-tile fused candidate below.
    # Preserve the profile default on any unrecognised value.
    if profile in ("ascend_a2", "metax_c500", "nvidia_h200"):
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
    if profile == "nvidia_h200":
        # Allow orthogonal H200 sweeps without admitting arbitrary constexpr
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
def _curand_uniform_from_uint(value):
    """Bit-match CUDA curand's uint32-to-float conversion."""
    inv_2pow32: tl.constexpr = 2.3283064365386963e-10
    half_ulp: tl.constexpr = 1.1641532182693481e-10
    result = tl.fma(value.to(tl.float32), inv_2pow32, half_ulp)
    # PyTorch's uniform kernel reverses curand's (0, 1] bound to [0, 1).
    return tl.where(result == 1.0, 0.0, result)


@triton.jit
def _cuda_sin_approx(value):
    """Bit-match the sine half of CUDA ``__sincosf``."""
    return tl.inline_asm_elementwise(
        "sin.approx.f32 $0, $1;",
        "=f,f",
        [value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _task09_curand_pair(seed, counter, subsequence):
    """Return curand uniform and curand_normal4.x for one CUDA thread."""
    counter = counter.to(tl.uint64)
    counter_lo = counter.to(tl.uint32)
    counter_hi = (counter >> 32).to(tl.uint32)
    subsequence = subsequence.to(tl.uint32)
    zero = subsequence * 0
    random_x, random_y, _, _ = tl.philox(
        seed,
        counter_lo,
        counter_hi,
        subsequence,
        zero,
    )
    uniform = _curand_uniform_from_uint(random_x)
    inv_2pow32: tl.constexpr = 2.3283064365386963e-10
    half_ulp: tl.constexpr = 1.1641532182693481e-10
    inv_2pow32_2pi: tl.constexpr = 1.4629180792671596e-9
    half_ulp_2pi: tl.constexpr = 7.314590396335798e-10
    radius_uniform = tl.fma(
        random_x.to(tl.float32), inv_2pow32, half_ulp
    )
    angle = tl.fma(
        random_y.to(tl.float32), inv_2pow32_2pi, half_ulp_2pi
    )
    normal = (
        tl.sqrt_rn(-2.0 * tl.log(radius_uniform))
        * _cuda_sin_approx(angle)
    )
    return uniform, normal


@triton.jit
def _task09_philox_rng_kernel(seed, base_offset, rng_ptr, BLOCK: tl.constexpr):
    """Reconstruct the three rand(4) calls and randn(4, 3) in one launch."""
    lane = tl.arange(0, BLOCK)
    base_counter = base_offset.to(tl.uint64) >> 2
    is_uniform = lane < 12
    uniform_call = lane >> 2
    uniform_subsequence = lane & 3
    normal_subsequence = lane - 12
    counter = base_counter + tl.where(is_uniform, uniform_call, 3).to(tl.uint64)
    subsequence = tl.where(
        is_uniform,
        uniform_subsequence,
        normal_subsequence,
    )
    uniform, normal = _task09_curand_pair(seed, counter, subsequence)
    value = tl.where(is_uniform, uniform, normal)
    tl.store(rng_ptr + lane, value, mask=lane < 24)


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


@triton.jit(do_not_specialize=["seed", "base_offset"])
def _center_augment_philox_fused_kernel(
    x_ptr,
    mask_ptr,
    out_ptr,
    seed: tl.uint64,
    base_offset: tl.uint64,
    n_atom,
    s_trans: tl.constexpr,
    HAS_MASK: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Generate exact PyTorch CUDA randoms and augment in one launch."""
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

    # Six useful lanes reconstruct u1/u2/u3/tx/ty/tz for this sample.
    # Reducing a single non-zero lane broadcasts each result to the atom tile.
    rng_lane = tl.arange(0, 8)
    is_uniform = rng_lane < 3
    base_counter = base_offset.to(tl.uint64) >> 2
    rng_counter = base_counter + tl.where(is_uniform, rng_lane, 3).to(tl.uint64)
    rng_subsequence = tl.where(
        is_uniform,
        sample,
        sample * 3 + rng_lane - 3,
    )
    uniform, normal = _task09_curand_pair(seed, rng_counter, rng_subsequence)
    random_value = tl.where(is_uniform, uniform, normal)
    u1 = tl.sum(tl.where(rng_lane == 0, random_value, 0.0), axis=0)
    u2 = tl.sum(tl.where(rng_lane == 1, random_value, 0.0), axis=0)
    u3 = tl.sum(tl.where(rng_lane == 2, random_value, 0.0), axis=0)
    tx = s_trans * tl.sum(
        tl.where(rng_lane == 3, random_value, 0.0), axis=0
    )
    ty = s_trans * tl.sum(
        tl.where(rng_lane == 4, random_value, 0.0), axis=0
    )
    tz = s_trans * tl.sum(
        tl.where(rng_lane == 5, random_value, 0.0), axis=0
    )

    two_pi: tl.constexpr = 6.283185307179586
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
def _center_augment_rng_buffer_kernel(
    x_ptr,
    mask_ptr,
    rng_ptr,
    out_ptr,
    n_atom,
    s_trans: tl.constexpr,
    HAS_MASK: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Consume the one-launch, one-allocation packed Philox buffer."""
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
    u1 = tl.load(rng_ptr + sample).to(tl.float32)
    u2 = tl.load(rng_ptr + 4 + sample).to(tl.float32)
    u3 = tl.load(rng_ptr + 8 + sample).to(tl.float32)
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
    trans_base = 12 + sample * 3
    tx = s_trans * tl.load(rng_ptr + trans_base)
    ty = s_trans * tl.load(rng_ptr + trans_base + 1)
    tz = s_trans * tl.load(rng_ptr + trans_base + 2)
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
    """Share one centering reduction across two/four output samples."""
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


def _reserve_default_cuda_philox(
    device: torch.device,
    increment: int,
) -> tuple[int, int]:
    """Return (seed, old_offset) and reserve a Philox range.

    PyTorch does not expose its atomic ``philox_cuda_state`` reservation in
    Python in all supported releases.  The H200 official-shape fast path uses
    the default CUDA generator's seed/offset accessors.  The official runner
    is single-threaded; callers sharing a generator across Python threads must
    provide their own serialization.
    """
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    generator = torch.cuda.default_generators[device_index]
    seed = generator.initial_seed()
    offset = generator.get_offset()
    generator.set_offset(offset + increment)
    return seed, offset


def centre_random_augmentation(
    x_input_coords: torch.Tensor,
    n_sample: int,
    s_trans: float,
    centre_only: bool,
    mask: Optional[torch.Tensor],
    kernel_config,
    samples_per_program: int,
    use_tiled_fused: bool,
    use_packed_rng: bool,
    philox_mode: str = "off",
    eps: float = 1e-12,
) -> torch.Tensor:
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
        use_fixed_philox = (
            philox_mode in ("buffer", "direct")
            and x_input_coords.device.type == "cuda"
            and x_input_coords.dtype == torch.float32
            and n_sample == 4
            and n_atom == 256
            and atom_block == 256
            and samples_per_program == 1
            and not use_tiled_fused
            and x_input_coords.ndim == 2
            and x_input_coords.shape[1] == 3
            and x_input_coords.is_contiguous()
            and (mask is None or mask.is_contiguous())
        )
        if use_fixed_philox:
            seed, base_offset = _reserve_default_cuda_philox(
                x_input_coords.device,
                16,
            )
            if philox_mode == "direct":
                _center_augment_philox_fused_kernel[(n_sample,)](
                    x_input_coords,
                    mask_arg,
                    out,
                    seed,
                    base_offset,
                    n_atom,
                    s_trans=s_trans,
                    HAS_MASK=mask is not None,
                    eps=eps,
                    BLOCK=atom_block,
                    num_warps=augment_warps,
                )
                return out
            rng_backing = torch.empty(
                24,
                device=x_input_coords.device,
                dtype=torch.float32,
            )
            _task09_philox_rng_kernel[(1,)](
                seed,
                base_offset,
                rng_backing,
                BLOCK=32,
                num_warps=1,
            )
            _center_augment_rng_buffer_kernel[(n_sample,)](
                x_input_coords,
                mask_arg,
                rng_backing,
                out,
                n_atom,
                s_trans=s_trans,
                HAS_MASK=mask is not None,
                eps=eps,
                BLOCK=atom_block,
                num_warps=augment_warps,
            )
            return out
        if use_packed_rng:
            # Keep four distinct RNG calls with the exact reference shapes and
            # order, but back their disjoint outputs with one allocation.
            # Allocation never consumes Philox state, and ``out=`` preserves
            # the generator calls while removing three allocator operations.
            rng_backing = torch.empty(
                6 * n_sample,
                device=x_input_coords.device,
                dtype=x_input_coords.dtype,
            )
            u1 = rng_backing[:n_sample]
            u2 = rng_backing[n_sample:2 * n_sample]
            u3 = rng_backing[2 * n_sample:3 * n_sample]
            trans = rng_backing[3 * n_sample:].view(n_sample, 3)
            torch.rand((n_sample,), out=u1)
            torch.rand((n_sample,), out=u2)
            torch.rand((n_sample,), out=u3)
            torch.randn((n_sample, 3), out=trans)
        else:
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
            trans = torch.randn(
                n_sample,
                3,
                device=x_input_coords.device,
                dtype=x_input_coords.dtype,
            )
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
        # ``KERNELSWIFT_TASK09_SPP`` is the canonical selector.  Accept the
        # longer historical spelling as a compatibility alias because the
        # first H200 sweep JSON used it; canonical SPP wins if both are set.
        spp_key = os.environ.get("KERNELSWIFT_TASK09_SPP")
        if spp_key is None:
            spp_key = os.environ.get(
                "KERNELSWIFT_TASK09_SAMPLES_PER_PROGRAM", "1"
            )
        spp_key = spp_key.strip()
        self._samples_per_program = {"1": 1, "2": 2, "4": 4}.get(spp_key, 1)
        fused_key = os.environ.get(
            "KERNELSWIFT_TASK09_FUSED_TILES", "0"
        ).strip().lower()
        self._use_tiled_fused = (
            self.kernel_profile in ("metax_c500", "nvidia_h200")
            and {"0": False, "false": False, "no": False,
                 "1": True, "true": True, "yes": True}.get(
                     fused_key, False
                 )
        )
        packed_rng_key = os.environ.get(
            "KERNELSWIFT_TASK09_PACKED_RNG", "0"
        ).strip().lower()
        self._use_packed_rng = {
            "0": False,
            "false": False,
            "no": False,
            "1": True,
            "true": True,
            "yes": True,
        }.get(packed_rng_key, False)
        philox_env = os.environ.get("KERNELSWIFT_TASK09_PHILOX_MODE")
        if philox_env is None:
            philox_key = (
                "direct"
                if self.kernel_profile == "nvidia_h200" and not self._use_packed_rng
                else "off"
            )
        else:
            philox_key = philox_env.strip().lower()
        if (
            self.kernel_profile == "nvidia_h200"
            and philox_env is not None
            and philox_key == "off"
        ):
            # ``off`` is the complete rollback switch, even if an old packed
            # RNG selector remains in the caller's environment.
            self._use_packed_rng = False
        self._philox_mode = (
            {"off": "off", "buffer": "buffer", "direct": "direct"}.get(
                philox_key, "off"
            )
            if self.kernel_profile == "nvidia_h200"
            else "off"
        )
        triton36 = (
            str(getattr(triton, "__version__", "")).split("+", 1)[0]
            == "3.6.0"
        )
        compiled_default = (
            "1" if self.kernel_profile == "nvidia_h200" and triton36 else "0"
        )
        compiled_key = os.environ.get(
            "KERNELSWIFT_TASK09_COMPILED_LAUNCH", compiled_default
        ).strip().lower()
        self._use_compiled_launch = (
            self.kernel_profile == "nvidia_h200"
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
            self.n_sample == 4
            and self.s_trans == 1.0
            and not self.centre_only
            and self._kernel_config == (4, 256, 4)
            and self._samples_per_program == 1
            and not self._use_tiled_fused
            and not self._use_packed_rng
            and self._philox_mode == "direct"
        )
        graph_rng_key = os.environ.get(
            "KERNELSWIFT_TASK09_GRAPH_RNG", "1"
        ).strip().lower()
        if graph_rng_key not in ("0", "false", "no", "1", "true", "yes"):
            raise ValueError("Task09 GRAPH_RNG must be 0 or 1")
        self._use_graph_rng = (
            self.kernel_profile == "nvidia_h200"
            and graph_rng_key in ("1", "true", "yes")
            and torch.cuda.is_available()
            and self.n_sample == 4
            and self.s_trans == 1.0
            and not self.centre_only
            and self._kernel_config == (4, 256, 4)
            and self._samples_per_program == 1
            and not self._use_tiled_fused
            and not self._use_packed_rng
        )
        self._rng_graph = None
        self._rng_graph_inputs = None
        self._rng_graph_output = None

    def forward(self, x_input_coords: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        rng_graph = self._rng_graph
        rng_graph_inputs = self._rng_graph_inputs
        if (
            rng_graph is not None
            and rng_graph_inputs is not None
            and rng_graph_inputs[0] is x_input_coords
            and rng_graph_inputs[1] is mask
        ):
            # PyTorch's CUDAGraph RNG integration advances the default CUDA
            # generator by the captured increment on every replay.  Random
            # values and the full subsequent Philox sequence therefore remain
            # identical to the four eager rand/randn calls.
            rng_graph.replay()
            return self._rng_graph_output

        outer_capture = (
            x_input_coords.device.type == "cuda"
            and torch.cuda.is_current_stream_capturing()
        )
        graph_rng_official = (
            self._use_graph_rng
            and not outer_capture
            and x_input_coords.shape == (256, 3)
            and x_input_coords.dtype == torch.float32
            and x_input_coords.is_contiguous()
            and x_input_coords.device.type == "cuda"
            and mask is not None
            and mask.shape == (256,)
            and mask.dtype == torch.float32
            and mask.is_contiguous()
            and mask.device == x_input_coords.device
        )
        if graph_rng_official:
            device_index = x_input_coords.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            generator = torch.cuda.default_generators[device_index]
            saved_state = generator.get_state().clone()

            # Compile/allocate outside capture.  Restore the exact state so
            # this setup call is invisible to the task's RNG contract.
            centre_random_augmentation(
                x_input_coords,
                4,
                1.0,
                False,
                mask,
                (4, 256, 4),
                1,
                False,
                False,
                "off",
            )
            torch.cuda.synchronize()
            generator.set_state(saved_state)

            rng_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(rng_graph):
                output = centre_random_augmentation(
                    x_input_coords,
                    4,
                    1.0,
                    False,
                    mask,
                    (4, 256, 4),
                    1,
                    False,
                    False,
                    "off",
                )
            self._rng_graph = rng_graph
            self._rng_graph_inputs = (x_input_coords, mask)
            self._rng_graph_output = output
            # Capture records but does not execute the graph.  One replay is
            # the single logical call and advances Philox by exactly 16.
            rng_graph.replay()
            return output

        if (
            self._use_compiled_launch
            and not outer_capture
            and self._compiled_config_is_official
            and x_input_coords.shape == (256, 3)
            and x_input_coords.dtype == torch.float32
            and x_input_coords.is_contiguous()
            and x_input_coords.device.type == "cuda"
            and mask is not None
            and mask.shape == (256,)
            and mask.dtype == torch.float32
            and mask.is_contiguous()
            and mask.device == x_input_coords.device
        ):
            # Preserve PyTorch's four-call Philox contract exactly: reserve
            # the same 16 counter units on every invocation, including cached
            # launcher invocations.  Only compilation/binding metadata is
            # reused; random values, generator state and outputs are not.
            device_index = x_input_coords.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            generator = torch.cuda.default_generators[device_index]
            seed = generator.initial_seed()
            base_offset = generator.get_offset()
            generator.set_offset(base_offset + 16)
            out = x_input_coords.new_empty((4, 256, 3))
            if (
                self._compiled_launcher is not None
                and self._compiled_device == x_input_coords.device
            ):
                self._compiled_launcher(
                    x_input_coords,
                    mask,
                    out,
                    seed,
                    base_offset,
                    256,
                    1.0,
                    True,
                    1e-12,
                    256,
                )
            else:
                compiled = _center_augment_philox_fused_kernel[(4,)](
                    x_input_coords,
                    mask,
                    out,
                    seed,
                    base_offset,
                    256,
                    s_trans=1.0,
                    HAS_MASK=True,
                    eps=1e-12,
                    BLOCK=256,
                    num_warps=4,
                )
                if self._compiled_launcher is None and compiled is not None:
                    self._compiled_launcher = compiled[(4, 1, 1)]
                    self._compiled_device = x_input_coords.device
            return out
        return centre_random_augmentation(
            x_input_coords,
            self.n_sample,
            self.s_trans,
            self.centre_only,
            mask,
            self._kernel_config,
            self._samples_per_program,
            self._use_tiled_fused,
            self._use_packed_rng,
            self._philox_mode,
        )


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
