"""Triton implementation of Task07 (mhc_post).

The evaluated forward path always launches a Triton kernel.  PyTorch is used
only for tensor ownership/allocation and the ``nn.Module`` interface.
"""

import os
import torch
import torch.nn as nn
import triton
import triton.language as tl


_A2_VECTOR_CORE_COUNTS = {}


def _select_kernel_config():
    """Return ``(profile, launch_config)`` for the target.

    The final two fields are C500's sequence-tile width and hidden-tile
    height.  They stay zero on the portable hidden-vector path.
    """
    configs = {
        "auto": (128, 8, False, 0, 0),
        # A2 is a vector-only path.  Launch exactly the physical vector-core
        # count and let each core consume many rows in a device-side loop.
        # This avoids the 40,960 tiny-program launches of the direct mapping.
        "ascend_a2": (256, 4, True, 0, 0),
        # CPU->C500 migration gives 4-D tensors a channels-last stride.  Tile
        # sequence as the contiguous dimension and hidden as a small outer
        # dimension; 128x8 exposes 1024 independent elements per program.
        "metax_c500": (1024, 8, False, 128, 16),
        # BI150's official inputs are contiguous.  Own all four output
        # channels in one program so x/residual are read once instead of four
        # times; 512 values map naturally to eight 64-lane vectors while four
        # warps keep enough CTAs resident on the 16-MP device.
        "iluvatar_bi150": (512, 4, False, 0, 0),
        "enflame_s60": (128, 8, False, 0, 0),
        "thead_810e": (256, 4, False, 0, 0),
        "cambricon_mlu590": (128, 4, False, 0, 0),
        "hygon_bw1000": (512, 4, False, 0, 0),
        "mthreads_ph100": (256, 8, False, 0, 0),
        "kunlun_p800": (128, 8, False, 0, 0),
        "biren_br106m": (512, 8, False, 0, 0),
    }
    profile = os.environ.get("KERNELSWIFT_CHIP", "iluvatar_bi150").strip().lower()
    if profile == "default" or profile not in configs:
        profile = "auto"
    config = configs[profile]
    if profile == "ascend_a2":
        # These overrides make real-device A/B sweeps possible without
        # changing the submitted math or adding an eager-framework fallback.
        block_override = os.environ.get("KERNELSWIFT_TASK07_BLOCK_H")
        if block_override is not None:
            block_h = int(block_override)
            if block_h not in (128, 256, 512, 1024):
                raise ValueError("Task07 BLOCK_H must be 128, 256, 512, or 1024")
            config = (block_h, config[1], config[2], 0, 0)
    elif profile in ("metax_c500", "iluvatar_bi150"):
        # C500 and BI150 stay on portable direct/stride-aware kernels.  In
        # particular, no value here can enable the A2 vector-core persistent
        # path or query Ascend-only device properties.
        variants = {
            "512x2": (512, 2, False, 0, 0),
            "128x4": (128, 4, False, 0, 0),
            "128x8": (128, 8, False, 0, 0),
            "256x4": (256, 4, False, 0, 0),
            "256x8": (256, 8, False, 0, 0),
            "512x4": (512, 4, False, 0, 0),
            "512x8": (512, 8, False, 0, 0),
            "1024x4": (1024, 4, False, 0, 0),
            "1024x8": (1024, 8, False, 0, 0),
            "1024x2": (1024, 2, False, 0, 0),
            "seq64x8_w2": (1024, 2, False, 64, 8),
            "seq64x8_w4": (1024, 4, False, 64, 8),
            "seq64x16_w4": (1024, 4, False, 64, 16),
            "seq32x32_w4": (1024, 4, False, 32, 32),
            "seq64x32_w4": (1024, 4, False, 64, 32),
            "seq64x32_w8": (1024, 8, False, 64, 32),
            "seq128x16_w4": (1024, 4, False, 128, 16),
            "seq128x16_w8": (1024, 8, False, 128, 16),
            "seq256x8_w4": (1024, 4, False, 256, 8),
            "seq256x8_w8": (1024, 8, False, 256, 8),
            "seq128x8_w2": (1024, 2, False, 128, 8),
            "seq128x8_w4": (1024, 4, False, 128, 8),
            "seq128x8_w8": (1024, 8, False, 128, 8),
            "seq256x4_w4": (1024, 4, False, 256, 4),
        }
        variant = os.environ.get("KERNELSWIFT_TASK07_CONFIG", "").strip().lower()
        config = variants.get(variant, config)
    return profile, config


def _ascend_vector_core_count(device_index: int) -> int:
    """Read A2's physical vector-core count, with an explicit sweep override."""
    override = os.environ.get("KERNELSWIFT_TASK07_PROGRAMS")
    cache_key = (device_index, override)
    if cache_key in _A2_VECTOR_CORE_COUNTS:
        return _A2_VECTOR_CORE_COUNTS[cache_key]

    if override is not None:
        count = int(override)
        if count < 1 or count > 65535:
            raise ValueError("Task07 program count must be in [1, 65535]")
    else:
        properties = triton.runtime.driver.active.utils.get_device_properties(device_index)
        count = int(properties["num_vectorcore"])
    _A2_VECTOR_CORE_COUNTS[cache_key] = count
    return count


@triton.jit
def _mhc_post_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    H_BLOCKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # mcTriton 3.0.0 on C500 mis-lowers this workload with a two-dimensional
    # grid.  Flattening (row, hidden tile) preserves exactly the same ownership
    # while using the most portable one-dimensional program-id mapping.
    pid = tl.program_id(0)
    row = pid // H_BLOCKS
    h_block = pid - row * H_BLOCKS
    h_off = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = h_off < hidden_size
    # Force row coefficients to have the same vector shape as the hidden tile.
    # Scalar-to-vector broadcast is mis-lowered by mcTriton 3.0.0 for this
    # multi-accumulator kernel; zero-offset vector loads preserve the semantics
    # and let the hardware cache service the repeated address efficiently.
    coeff_lane = tl.zeros((BLOCK_H,), tl.int32)

    post_base = row * hc_mult
    post0 = tl.load(post_ptr + post_base + coeff_lane).to(tl.float32)
    post1 = tl.load(post_ptr + post_base + 1 + coeff_lane).to(tl.float32)
    post2 = tl.load(post_ptr + post_base + 2 + coeff_lane).to(tl.float32)
    post3 = tl.load(post_ptr + post_base + 3 + coeff_lane).to(tl.float32)

    comb_base = row * hc_mult * hc_mult
    c00 = tl.load(comb_ptr + comb_base + coeff_lane).to(tl.float32)
    c01 = tl.load(comb_ptr + comb_base + 1 + coeff_lane).to(tl.float32)
    c02 = tl.load(comb_ptr + comb_base + 2 + coeff_lane).to(tl.float32)
    c03 = tl.load(comb_ptr + comb_base + 3 + coeff_lane).to(tl.float32)
    c10 = tl.load(comb_ptr + comb_base + 4 + coeff_lane).to(tl.float32)
    c11 = tl.load(comb_ptr + comb_base + 5 + coeff_lane).to(tl.float32)
    c12 = tl.load(comb_ptr + comb_base + 6 + coeff_lane).to(tl.float32)
    c13 = tl.load(comb_ptr + comb_base + 7 + coeff_lane).to(tl.float32)
    c20 = tl.load(comb_ptr + comb_base + 8 + coeff_lane).to(tl.float32)
    c21 = tl.load(comb_ptr + comb_base + 9 + coeff_lane).to(tl.float32)
    c22 = tl.load(comb_ptr + comb_base + 10 + coeff_lane).to(tl.float32)
    c23 = tl.load(comb_ptr + comb_base + 11 + coeff_lane).to(tl.float32)
    c30 = tl.load(comb_ptr + comb_base + 12 + coeff_lane).to(tl.float32)
    c31 = tl.load(comb_ptr + comb_base + 13 + coeff_lane).to(tl.float32)
    c32 = tl.load(comb_ptr + comb_base + 14 + coeff_lane).to(tl.float32)
    c33 = tl.load(comb_ptr + comb_base + 15 + coeff_lane).to(tl.float32)

    x = tl.load(
        x_ptr + row * hidden_size + h_off,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    acc0 = x * post0
    acc1 = x * post1
    acc2 = x * post2
    acc3 = x * post3

    residual_base = row * hc_mult * hidden_size + h_off
    r0 = tl.load(residual_ptr + residual_base, mask=mask, other=0.0).to(tl.float32)
    acc0 += r0 * c00
    acc1 += r0 * c01
    acc2 += r0 * c02
    acc3 += r0 * c03
    r1 = tl.load(
        residual_ptr + residual_base + hidden_size, mask=mask, other=0.0
    ).to(tl.float32)
    acc0 += r1 * c10
    acc1 += r1 * c11
    acc2 += r1 * c12
    acc3 += r1 * c13
    r2 = tl.load(
        residual_ptr + residual_base + 2 * hidden_size, mask=mask, other=0.0
    ).to(tl.float32)
    acc0 += r2 * c20
    acc1 += r2 * c21
    acc2 += r2 * c22
    acc3 += r2 * c23
    r3 = tl.load(
        residual_ptr + residual_base + 3 * hidden_size, mask=mask, other=0.0
    ).to(tl.float32)
    acc0 += r3 * c30
    acc1 += r3 * c31
    acc2 += r3 * c32
    acc3 += r3 * c33

    output_base = row * hc_mult * hidden_size + h_off
    tl.store(out_ptr + output_base, acc0, mask=mask)
    tl.store(out_ptr + output_base + hidden_size, acc1, mask=mask)
    tl.store(out_ptr + output_base + 2 * hidden_size, acc2, mask=mask)
    tl.store(out_ptr + output_base + 3 * hidden_size, acc3, mask=mask)


@triton.jit
def _mhc_post_c500_channel_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    seq_size: tl.constexpr,
    x_stride_0: tl.constexpr,
    x_stride_1: tl.constexpr,
    x_stride_h: tl.constexpr,
    residual_stride_0: tl.constexpr,
    residual_stride_1: tl.constexpr,
    residual_stride_m: tl.constexpr,
    residual_stride_h: tl.constexpr,
    post_stride_0: tl.constexpr,
    post_stride_1: tl.constexpr,
    post_stride_m: tl.constexpr,
    comb_stride_0: tl.constexpr,
    comb_stride_1: tl.constexpr,
    comb_stride_m: tl.constexpr,
    comb_stride_n: tl.constexpr,
    out_stride_0: tl.constexpr,
    out_stride_1: tl.constexpr,
    out_stride_m: tl.constexpr,
    out_stride_h: tl.constexpr,
    H_BLOCKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """C500: one output-channel owner, honoring mcPyTorch tensor strides."""
    pid = tl.program_id(0)
    h_block = pid % H_BLOCKS
    row_channel = pid // H_BLOCKS
    out_channel = row_channel % hc_mult
    row = row_channel // hc_mult
    batch = row // seq_size
    seq = row - batch * seq_size
    h_off = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = h_off < hidden_size
    coeff_lane = tl.zeros((BLOCK_H,), tl.int32)

    post = tl.load(
        post_ptr
        + batch * post_stride_0
        + seq * post_stride_1
        + out_channel * post_stride_m
        + coeff_lane
    ).to(tl.float32)
    x = tl.load(
        x_ptr
        + batch * x_stride_0
        + seq * x_stride_1
        + h_off * x_stride_h,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    acc = x * post

    residual_base = (
        batch * residual_stride_0
        + seq * residual_stride_1
        + h_off * residual_stride_h
    )
    comb_base = (
        batch * comb_stride_0
        + seq * comb_stride_1
        + out_channel * comb_stride_n
    )
    r0 = tl.load(
        residual_ptr + residual_base, mask=mask, other=0.0
    ).to(tl.float32)
    c0 = tl.load(comb_ptr + comb_base + coeff_lane).to(tl.float32)
    acc += r0 * c0
    r1 = tl.load(
        residual_ptr + residual_base + residual_stride_m,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    c1 = tl.load(comb_ptr + comb_base + comb_stride_m + coeff_lane).to(tl.float32)
    acc += r1 * c1
    r2 = tl.load(
        residual_ptr + residual_base + 2 * residual_stride_m,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    c2 = tl.load(
        comb_ptr + comb_base + 2 * comb_stride_m + coeff_lane
    ).to(tl.float32)
    acc += r2 * c2
    r3 = tl.load(
        residual_ptr + residual_base + 3 * residual_stride_m,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    c3 = tl.load(
        comb_ptr + comb_base + 3 * comb_stride_m + coeff_lane
    ).to(tl.float32)
    acc += r3 * c3

    output_base = (
        batch * out_stride_0
        + seq * out_stride_1
        + out_channel * out_stride_m
        + h_off * out_stride_h
    )
    tl.store(out_ptr + output_base, acc, mask=mask)


@triton.jit
def _mhc_post_c500_sequence_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    seq_size: tl.constexpr,
    x_stride_0: tl.constexpr,
    x_stride_1: tl.constexpr,
    x_stride_h: tl.constexpr,
    residual_stride_0: tl.constexpr,
    residual_stride_1: tl.constexpr,
    residual_stride_m: tl.constexpr,
    residual_stride_h: tl.constexpr,
    post_stride_0: tl.constexpr,
    post_stride_1: tl.constexpr,
    post_stride_m: tl.constexpr,
    comb_stride_0: tl.constexpr,
    comb_stride_1: tl.constexpr,
    comb_stride_m: tl.constexpr,
    comb_stride_n: tl.constexpr,
    out_stride_0: tl.constexpr,
    out_stride_1: tl.constexpr,
    out_stride_m: tl.constexpr,
    out_stride_h: tl.constexpr,
    S_BLOCKS: tl.constexpr,
    H_BLOCKS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H_TILE: tl.constexpr,
):
    """C500 channels-last kernel, contiguous along the sequence dimension."""
    pid = tl.program_id(0)
    s_block = pid % S_BLOCKS
    pid = pid // S_BLOCKS
    h_block = pid % H_BLOCKS
    row_channel = pid // H_BLOCKS
    out_channel = row_channel % hc_mult
    batch = row_channel // hc_mult

    s = s_block * BLOCK_S + tl.arange(0, BLOCK_S)[None, :]
    h = h_block * BLOCK_H_TILE + tl.arange(0, BLOCK_H_TILE)[:, None]
    s_mask = s < seq_size
    mask = s_mask & (h < hidden_size)

    post_base = (
        batch * post_stride_0
        + s * post_stride_1
        + out_channel * post_stride_m
    )
    post = tl.load(post_ptr + post_base, mask=s_mask, other=0.0).to(tl.float32)
    x_base = batch * x_stride_0 + s * x_stride_1 + h * x_stride_h
    x = tl.load(x_ptr + x_base, mask=mask, other=0.0).to(tl.float32)
    acc = x * post

    residual_base = (
        batch * residual_stride_0 + s * residual_stride_1 + h * residual_stride_h
    )
    comb_base = (
        batch * comb_stride_0
        + s * comb_stride_1
        + out_channel * comb_stride_n
    )
    r0 = tl.load(residual_ptr + residual_base, mask=mask, other=0.0).to(tl.float32)
    c0 = tl.load(comb_ptr + comb_base, mask=s_mask, other=0.0).to(tl.float32)
    acc += r0 * c0
    r1 = tl.load(
        residual_ptr + residual_base + residual_stride_m,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    c1 = tl.load(
        comb_ptr + comb_base + comb_stride_m, mask=s_mask, other=0.0
    ).to(tl.float32)
    acc += r1 * c1
    r2 = tl.load(
        residual_ptr + residual_base + 2 * residual_stride_m,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    c2 = tl.load(
        comb_ptr + comb_base + 2 * comb_stride_m, mask=s_mask, other=0.0
    ).to(tl.float32)
    acc += r2 * c2
    r3 = tl.load(
        residual_ptr + residual_base + 3 * residual_stride_m,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    c3 = tl.load(
        comb_ptr + comb_base + 3 * comb_stride_m, mask=s_mask, other=0.0
    ).to(tl.float32)
    acc += r3 * c3

    output_base = (
        batch * out_stride_0
        + s * out_stride_1
        + out_channel * out_stride_m
        + h * out_stride_h
    )
    tl.store(out_ptr + output_base, acc, mask=mask)


@triton.jit
def _mhc_post_a2_persistent_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    H_BLOCKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Persistent A2 kernel: one Triton program per physical vector core."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Assign complete rows, rather than independent (row, hidden-tile) tasks,
    # to a core.  The 20 row-wise FP32 mixing coefficients are then loaded
    # once and reused across all hidden tiles.
    for row in range(pid, NUM_ROWS, num_programs):
        post_base = row * hc_mult
        post0 = tl.load(post_ptr + post_base).to(tl.float32)
        post1 = tl.load(post_ptr + post_base + 1).to(tl.float32)
        post2 = tl.load(post_ptr + post_base + 2).to(tl.float32)
        post3 = tl.load(post_ptr + post_base + 3).to(tl.float32)

        comb_base = row * hc_mult * hc_mult
        c00 = tl.load(comb_ptr + comb_base).to(tl.float32)
        c01 = tl.load(comb_ptr + comb_base + 1).to(tl.float32)
        c02 = tl.load(comb_ptr + comb_base + 2).to(tl.float32)
        c03 = tl.load(comb_ptr + comb_base + 3).to(tl.float32)
        c10 = tl.load(comb_ptr + comb_base + 4).to(tl.float32)
        c11 = tl.load(comb_ptr + comb_base + 5).to(tl.float32)
        c12 = tl.load(comb_ptr + comb_base + 6).to(tl.float32)
        c13 = tl.load(comb_ptr + comb_base + 7).to(tl.float32)
        c20 = tl.load(comb_ptr + comb_base + 8).to(tl.float32)
        c21 = tl.load(comb_ptr + comb_base + 9).to(tl.float32)
        c22 = tl.load(comb_ptr + comb_base + 10).to(tl.float32)
        c23 = tl.load(comb_ptr + comb_base + 11).to(tl.float32)
        c30 = tl.load(comb_ptr + comb_base + 12).to(tl.float32)
        c31 = tl.load(comb_ptr + comb_base + 13).to(tl.float32)
        c32 = tl.load(comb_ptr + comb_base + 14).to(tl.float32)
        c33 = tl.load(comb_ptr + comb_base + 15).to(tl.float32)

        for h_block in range(0, H_BLOCKS):
            h_off = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
            mask = h_off < hidden_size
            x = tl.load(
                x_ptr + row * hidden_size + h_off,
                mask=mask,
                other=0.0,
            ).to(tl.float32)

            acc0 = x * post0
            acc1 = x * post1
            acc2 = x * post2
            acc3 = x * post3

            residual_base = row * hc_mult * hidden_size + h_off
            r0 = tl.load(
                residual_ptr + residual_base,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc0 += r0 * c00
            acc1 += r0 * c01
            acc2 += r0 * c02
            acc3 += r0 * c03

            r1 = tl.load(
                residual_ptr + residual_base + hidden_size,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc0 += r1 * c10
            acc1 += r1 * c11
            acc2 += r1 * c12
            acc3 += r1 * c13

            r2 = tl.load(
                residual_ptr + residual_base + 2 * hidden_size,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc0 += r2 * c20
            acc1 += r2 * c21
            acc2 += r2 * c22
            acc3 += r2 * c23

            r3 = tl.load(
                residual_ptr + residual_base + 3 * hidden_size,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc0 += r3 * c30
            acc1 += r3 * c31
            acc2 += r3 * c32
            acc3 += r3 * c33

            output_base = row * hc_mult * hidden_size + h_off
            tl.store(out_ptr + output_base, acc0, mask=mask)
            tl.store(out_ptr + output_base + hidden_size, acc1, mask=mask)
            tl.store(out_ptr + output_base + 2 * hidden_size, acc2, mask=mask)
            tl.store(out_ptr + output_base + 3 * hidden_size, acc3, mask=mask)


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    kernel_config,
    kernel_profile: str,
    compiled_runners=None,
) -> torch.Tensor:
    if x.ndim != 3 or residual.ndim != 4:
        raise ValueError("expected x [n0,n1,h] and residual [n0,n1,m,h]")
    tensors = (x, residual, post_layer_mix, comb_res_mix)
    if any(t.device.type == "cpu" for t in tensors):
        raise ValueError("mhc_post requires accelerator tensors supported by Triton")
    if not all(t.device == x.device for t in tensors):
        raise ValueError("all inputs must be on the same accelerator")
    n0, n1, hidden_size = x.shape
    hc_mult = residual.shape[2]
    if hc_mult != 4:
        raise ValueError("the Task07 reference contract requires hc_mult=4")
    if residual.shape != (n0, n1, hc_mult, hidden_size):
        raise ValueError("residual shape is inconsistent with x")
    if post_layer_mix.shape != (n0, n1, hc_mult, 1):
        raise ValueError("post_layer_mix must have shape [n0,n1,m,1]")
    if comb_res_mix.shape != (n0, n1, hc_mult, hc_mult):
        raise ValueError("comb_res_mix must have shape [n0,n1,m,m]")

    block_h, num_warps, persistent, block_s, block_h_tile = kernel_config
    c500_sequence_layout = (
        kernel_profile == "metax_c500"
        and block_s > 0
        and residual.stride(1) == 1
        and post_layer_mix.stride(1) == 1
        and comb_res_mix.stride(1) == 1
    )
    if c500_sequence_layout:
        # Preserve the C500 channels-last layout without materializing a copy.
        out = torch.empty_like(residual)
    else:
        out = torch.empty(
            (n0, n1, hc_mult, hidden_size),
            device=x.device,
            dtype=torch.bfloat16,
        )

    if c500_sequence_layout:
        s_blocks = triton.cdiv(n1, block_s)
        h_blocks = triton.cdiv(hidden_size, block_h_tile)
        grid = (n0 * hc_mult * h_blocks * s_blocks,)
        _mhc_post_c500_sequence_kernel[grid](
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            out,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            seq_size=n1,
            x_stride_0=x.stride(0),
            x_stride_1=x.stride(1),
            x_stride_h=x.stride(2),
            residual_stride_0=residual.stride(0),
            residual_stride_1=residual.stride(1),
            residual_stride_m=residual.stride(2),
            residual_stride_h=residual.stride(3),
            post_stride_0=post_layer_mix.stride(0),
            post_stride_1=post_layer_mix.stride(1),
            post_stride_m=post_layer_mix.stride(2),
            comb_stride_0=comb_res_mix.stride(0),
            comb_stride_1=comb_res_mix.stride(1),
            comb_stride_m=comb_res_mix.stride(2),
            comb_stride_n=comb_res_mix.stride(3),
            out_stride_0=out.stride(0),
            out_stride_1=out.stride(1),
            out_stride_m=out.stride(2),
            out_stride_h=out.stride(3),
            S_BLOCKS=s_blocks,
            H_BLOCKS=h_blocks,
            BLOCK_S=block_s,
            BLOCK_H_TILE=block_h_tile,
            num_warps=num_warps,
        )
    elif kernel_profile == "metax_c500":
        h_blocks = triton.next_power_of_2(triton.cdiv(hidden_size, block_h))
        grid = (n0 * n1 * hc_mult * h_blocks,)
        _mhc_post_c500_channel_kernel[grid](
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            out,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            seq_size=n1,
            x_stride_0=x.stride(0),
            x_stride_1=x.stride(1),
            x_stride_h=x.stride(2),
            residual_stride_0=residual.stride(0),
            residual_stride_1=residual.stride(1),
            residual_stride_m=residual.stride(2),
            residual_stride_h=residual.stride(3),
            post_stride_0=post_layer_mix.stride(0),
            post_stride_1=post_layer_mix.stride(1),
            post_stride_m=post_layer_mix.stride(2),
            comb_stride_0=comb_res_mix.stride(0),
            comb_stride_1=comb_res_mix.stride(1),
            comb_stride_m=comb_res_mix.stride(2),
            comb_stride_n=comb_res_mix.stride(3),
            out_stride_0=out.stride(0),
            out_stride_1=out.stride(1),
            out_stride_m=out.stride(2),
            out_stride_h=out.stride(3),
            H_BLOCKS=h_blocks,
            BLOCK_H=block_h,
            num_warps=num_warps,
        )
    elif persistent:
        device_index = x.device.index if x.device.index is not None else 0
        num_programs = min(
            n0 * n1,
            _ascend_vector_core_count(device_index),
        )
        h_blocks = triton.cdiv(hidden_size, block_h)
        _mhc_post_a2_persistent_kernel[(num_programs,)](
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            out,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            NUM_ROWS=n0 * n1,
            H_BLOCKS=h_blocks,
            BLOCK_H=block_h,
            num_warps=num_warps,
        )
    else:
        h_blocks = triton.cdiv(hidden_size, block_h)
        grid = (n0 * n1 * h_blocks,)
        compiled_grid = (grid[0], 1, 1)
        runner = False
        if compiled_runners is not None:
            runner_key = (
                compiled_grid,
                block_h,
                num_warps,
                hidden_size,
                hc_mult,
                h_blocks,
                x.device,
                x.dtype,
                tuple(x.shape),
                x.stride(),
                x.data_ptr() & 15,
                residual.device,
                residual.dtype,
                tuple(residual.shape),
                residual.stride(),
                residual.data_ptr() & 15,
                post_layer_mix.device,
                post_layer_mix.dtype,
                tuple(post_layer_mix.shape),
                post_layer_mix.stride(),
                post_layer_mix.data_ptr() & 15,
                comb_res_mix.device,
                comb_res_mix.dtype,
                tuple(comb_res_mix.shape),
                comb_res_mix.stride(),
                comb_res_mix.data_ptr() & 15,
                out.device,
                out.dtype,
                tuple(out.shape),
                out.stride(),
                out.data_ptr() & 15,
            )
            runner = compiled_runners.get(runner_key)
            if runner is not False:
                if runner is None and hasattr(_mhc_post_kernel, "warmup"):
                    compiled = _mhc_post_kernel.warmup(
                        x,
                        residual,
                        post_layer_mix,
                        comb_res_mix,
                        out,
                        hidden_size,
                        hc_mult,
                        h_blocks,
                        block_h,
                        grid=compiled_grid,
                        num_warps=num_warps,
                    )
                    runner = compiled[compiled_grid]
                    compiled_runners[runner_key] = runner
                elif runner is None:
                    compiled_runners[runner_key] = False
                    runner = False
                if runner is not False:
                    # CoreX's direct-kernel signature is exactly five pointers.
                    runner(x, residual, post_layer_mix, comb_res_mix, out)
        if runner is False:
            _mhc_post_kernel[grid](
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                out,
                hidden_size=hidden_size,
                hc_mult=hc_mult,
                H_BLOCKS=h_blocks,
                BLOCK_H=block_h,
                num_warps=num_warps,
            )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_profile, self._kernel_config = _select_kernel_config()
        self._use_compiled_runner = self.kernel_profile == "iluvatar_bi150"
        self._use_fixed_shape_runner = (
            os.environ.get("KERNELSWIFT_TASK07_FIXED_RUNNER", "1").strip()
            != "0"
        )
        self._fixed_direct_runner = None
        self._fixed_direct_runner_device = None
        # Compiled launchers, never output tensors or computed values.
        self._compiled_runners = {}
        graph_value = os.environ.get("KERNELSWIFT_TASK07_GRAPH_REPLAY")
        if graph_value is None:
                graph_value = os.environ.get("KERNELSWIFT_BI150_GRAPH_REPLAY", "1")
        self._use_graph_replay = (
            self.kernel_profile == "iluvatar_bi150"
            and graph_value.strip().lower() in ("1", "true", "yes")
        )
        self._graph = None
        self._graph_inputs = None
        self._graph_output = None

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        if not self._use_graph_replay or x.device.type == "cpu":
            return self._forward_impl(
                x, residual, post_layer_mix, comb_res_mix
            )

        graph_inputs = self._graph_inputs
        if (
            self._graph is not None
            and graph_inputs is not None
            and graph_inputs[0] is x
            and graph_inputs[1] is residual
            and graph_inputs[2] is post_layer_mix
            and graph_inputs[3] is comb_res_mix
        ):
            self._graph.replay()
            return self._graph_output

        self._forward_impl(x, residual, post_layer_mix, comb_res_mix)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self._forward_impl(
                x, residual, post_layer_mix, comb_res_mix
            )
        self._graph = graph
        self._graph_inputs = (x, residual, post_layer_mix, comb_res_mix)
        self._graph_output = output
        graph.replay()
        return output

    def _forward_impl(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        # The published BI150 case has a single dense layout.  Avoid rebuilding
        # and hashing the full generic specialization key on every timed call,
        # while retaining the validated generic path for every other layout.
        fixed_layout = (
            self._use_compiled_runner
            and self._use_fixed_shape_runner
            and self._kernel_config == (512, 4, False, 0, 0)
            and x.shape == (2, 4096, 1280)
            and residual.shape == (2, 4096, 4, 1280)
            and post_layer_mix.shape == (2, 4096, 4, 1)
            and comb_res_mix.shape == (2, 4096, 4, 4)
            and x.dtype == torch.bfloat16
            and residual.dtype == torch.bfloat16
            and post_layer_mix.dtype == torch.float32
            and comb_res_mix.dtype == torch.float32
            and x.is_contiguous()
            and residual.is_contiguous()
            and post_layer_mix.is_contiguous()
            and comb_res_mix.is_contiguous()
            and residual.device == x.device
            and post_layer_mix.device == x.device
            and comb_res_mix.device == x.device
            and x.device.type != "cpu"
            and hasattr(_mhc_post_kernel, "warmup")
        )
        if fixed_layout:
            out = torch.empty(
                (2, 4096, 4, 1280),
                device=x.device,
                dtype=torch.bfloat16,
            )
            if (
                self._fixed_direct_runner is None
                or self._fixed_direct_runner_device != x.device
            ):
                compiled_grid = (24576, 1, 1)
                compiled = _mhc_post_kernel.warmup(
                    x,
                    residual,
                    post_layer_mix,
                    comb_res_mix,
                    out,
                    1280,
                    4,
                    3,
                    512,
                    grid=compiled_grid,
                    num_warps=4,
                )
                self._fixed_direct_runner = compiled[compiled_grid]
                self._fixed_direct_runner_device = x.device
            self._fixed_direct_runner(
                x, residual, post_layer_mix, comb_res_mix, out
            )
            return out

        return mhc_post(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            self._kernel_config,
            self.kernel_profile,
            self._compiled_runners if self._use_compiled_runner else None,
        )


class Model(ModelNew):
    """Compatibility name required by the written competition rules."""


n0 = 2
n1 = 4096
h = 1280
mhc_mult = 4


def get_inputs():
    x = torch.randn((n0, n1, h), dtype=torch.bfloat16)
    residual = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16)
    post_layer_mix = torch.randn((n0, n1, mhc_mult, 1), dtype=torch.float32)
    comb_res_mix = torch.randn((n0, n1, mhc_mult, mhc_mult), dtype=torch.float32)
    return [x, residual, post_layer_mix, comb_res_mix]


def get_init_inputs():
    return []
