"""Triton implementation of Task07 (mhc_post).

The evaluated forward path always launches a Triton kernel.  PyTorch is used
only for tensor ownership/allocation and the ``nn.Module`` interface.
"""

import os
import torch
import torch.nn as nn
import triton
import triton.language as tl


class _TrustedHotPath:
    """Cache layout validation only; each call allocates and fills fresh output."""

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


_A2_VECTOR_CORE_COUNTS = {}


def _select_kernel_config():
    """Return ``(profile, launch_config)`` for the target.

    The final three fields are C500's sequence-tile width, hidden-tile
    height, and whether one program owns all four output channels.  The tile
    fields stay zero on the portable hidden-vector path.
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
        "metax_c500": (1024, 8, False, 128, 8, True),
        "iluvatar_bi150": (512, 8, False, 0, 0),
        "enflame_s60": (128, 8, False, 0, 0),
        "thead_810e": (256, 4, False, 0, 0),
        "cambricon_mlu590": (128, 4, False, 0, 0),
        "hygon_bw1000": (512, 4, False, 0, 0),
        "mthreads_ph100": (256, 8, False, 0, 0),
        "kunlun_p800": (128, 8, False, 0, 0),
        "nvidia_h200": (256, 1, False, 0, 0),
    }
    profile = os.environ.get("KERNELSWIFT_CHIP", "mthreads_ph100").strip().lower()
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
    elif profile == "mthreads_ph100":
        # Keep PH100 on hidden-contiguous, backend-neutral kernels.  ``exact``
        # uses two direct hidden tiles (1024 + a masked 256 tail) for H=1280.
        # The default is the 56-program row-persistent path: the public S5000
        # runtime example reports 56 MPs, and one program can reuse each row's
        # 20 mixing coefficients across all hidden tiles.  Direct/exact remain
        # explicit A/B paths because the real Triton-MUSA occupancy decision
        # must be made on the target card.
        variants = {
            "64x4": (64, 4, False, 0, 0),
            "64x8": (64, 8, False, 0, 0),
            "128x4": (128, 4, False, 0, 0),
            "256x4": (256, 4, False, 0, 0),
            "256x8": (256, 8, False, 0, 0),
            "512x4": (512, 4, False, 0, 0),
            "512x8": (512, 8, False, 0, 0),
            "1024x4": (1024, 4, False, 0, 0),
            "1024x8": (1024, 8, False, 0, 0),
        }
        variant = os.environ.get("KERNELSWIFT_TASK07_CONFIG", "").strip().lower()
        config = variants.get(variant, config)

        mode = os.environ.get(
            "KERNELSWIFT_TASK07_MODE", "mp56_persistent"
        ).strip().lower()
        if mode not in (
            "direct", "exact", "row_persistent", "mp56_persistent", "trusted_graph"
        ):
            mode = "mp56_persistent"
        # ``trusted_graph`` is a descriptive alias for the same safe
        # persistent ownership plan.  It never captures values or reuses an
        # output tensor; it only selects the static launch geometry once.
        if mode == "trusted_graph":
            mode = "mp56_persistent"
        block_options = {
            "64": 64, "128": 128, "256": 256, "512": 512, "1024": 1024
        }
        warp_options = {"2": 2, "4": 4, "8": 8}
        block_key = os.environ.get("KERNELSWIFT_TASK07_BLOCK_H", "").strip()
        warp_key = os.environ.get("KERNELSWIFT_TASK07_WARPS", "").strip()
        selected_block = block_options.get(block_key, config[0])
        if mode == "exact" and block_key == "":
            selected_block = 1024
        config = (
            selected_block,
            warp_options.get(warp_key, config[1]),
            56 if mode == "mp56_persistent" else mode == "row_persistent",
            0,
            0,
        )
    elif profile == "nvidia_h200":
        # Keep H200 on the portable contiguous hidden-vector kernel.  In
        # particular, do not opt it into the C500 channel-last sequence
        # kernels below: their launch geometry was designed around a
        # different host-to-device stride transformation.  These bounded
        # presets and scalar overrides expose only BLOCK_H and num_warps of
        # the already-selected portable path.  Unknown values retain the
        # measured H200 default (256, 1).
        variants = {
            "128x1": (128, 1, False, 0, 0),
            "128x2": (128, 2, False, 0, 0),
            "128x4": (128, 4, False, 0, 0),
            "128x8": (128, 8, False, 0, 0),
            "256x1": (256, 1, False, 0, 0),
            "256x2": (256, 2, False, 0, 0),
            "256x4": (256, 4, False, 0, 0),
            "256x8": (256, 8, False, 0, 0),
            "512x1": (512, 1, False, 0, 0),
            "512x2": (512, 2, False, 0, 0),
            "512x4": (512, 4, False, 0, 0),
            "512x8": (512, 8, False, 0, 0),
            "1024x1": (1024, 1, False, 0, 0),
            "1024x4": (1024, 4, False, 0, 0),
            "1024x8": (1024, 8, False, 0, 0),
        }
        variant = os.environ.get("KERNELSWIFT_TASK07_CONFIG", "").strip().lower()
        config = variants.get(variant, config)

        # The row-persistent candidate is explicit opt-in: the unconfigured
        # H200 submission remains on the direct path.  Both modes use the
        # ordinary hidden-contiguous layout; neither can select a C500
        # channel-last kernel.  Unknown values safely retain the default.
        mode = os.environ.get("KERNELSWIFT_TASK07_MODE", "direct").strip().lower()
        if mode not in ("direct", "row_persistent"):
            mode = "direct"

        block_options = {"128": 128, "256": 256, "512": 512, "1024": 1024}
        warp_options = {"1": 1, "2": 2, "4": 4, "8": 8}
        block_key = os.environ.get("KERNELSWIFT_TASK07_BLOCK_H", "").strip()
        warp_key = os.environ.get("KERNELSWIFT_TASK07_WARPS", "").strip()
        config = (
            block_options.get(block_key, config[0]),
            warp_options.get(warp_key, config[1]),
            mode == "row_persistent",
            0,
            0,
        )
    elif profile == "metax_c500":
        # C500 uses stride-aware Triton paths, with the four-output sequence
        # kernel as the measured default.  No value here can enable the A2
        # vector-core persistent path or query Ascend-only device properties.
        variants = {
            "512x2": (512, 2, False, 0, 0, False),
            "128x4": (128, 4, False, 0, 0, False),
            "128x8": (128, 8, False, 0, 0, False),
            "256x4": (256, 4, False, 0, 0, False),
            "256x8": (256, 8, False, 0, 0, False),
            "512x4": (512, 4, False, 0, 0, False),
            "512x8": (512, 8, False, 0, 0, False),
            "1024x4": (1024, 4, False, 0, 0, False),
            "1024x8": (1024, 8, False, 0, 0, False),
            "1024x2": (1024, 2, False, 0, 0, False),
            "seq64x8_w2": (1024, 2, False, 64, 8, False),
            "seq64x8_w4": (1024, 4, False, 64, 8, False),
            "seq64x16_w4": (1024, 4, False, 64, 16, False),
            "seq32x32_w4": (1024, 4, False, 32, 32, False),
            "seq64x32_w4": (1024, 4, False, 64, 32, False),
            "seq64x32_w8": (1024, 8, False, 64, 32, False),
            "seq128x16_w4": (1024, 4, False, 128, 16, False),
            "seq128x16_w8": (1024, 8, False, 128, 16, False),
            "seq256x8_w4": (1024, 4, False, 256, 8, False),
            "seq256x8_w8": (1024, 8, False, 256, 8, False),
            "seq128x8_w2": (1024, 2, False, 128, 8, False),
            "seq128x8_w4": (1024, 4, False, 128, 8, False),
            "seq128x8_w8": (1024, 8, False, 128, 8, False),
            "seq256x4_w4": (1024, 4, False, 256, 4, False),
            # One program owns all four output channels for a sequence/hidden
            # tile.  This loads x and the four residual channels only once.
            "seq4_32x8_w2": (1024, 2, False, 32, 8, True),
            "seq4_32x8_w4": (1024, 4, False, 32, 8, True),
            "seq4_32x16_w4": (1024, 4, False, 32, 16, True),
            "seq4_64x8_w4": (1024, 4, False, 64, 8, True),
            "seq4_64x8_w8": (1024, 8, False, 64, 8, True),
            "seq4_64x16_w8": (1024, 8, False, 64, 16, True),
            "seq4_64x16_w16": (1024, 16, False, 64, 16, True),
            "seq4_64x32_w8": (1024, 8, False, 64, 32, True),
            "seq4_128x8_w8": (1024, 8, False, 128, 8, True),
            "seq4_128x8_w16": (1024, 16, False, 128, 8, True),
            "seq4_128x16_w8": (1024, 8, False, 128, 16, True),
            "seq4_128x16_w16": (1024, 16, False, 128, 16, True),
            "seq4_128x4_w4": (1024, 4, False, 128, 4, True),
            "seq4_128x4_w8": (1024, 8, False, 128, 4, True),
            "seq4_256x4_w8": (1024, 8, False, 256, 4, True),
            "seq4_256x4_w16": (1024, 16, False, 256, 4, True),
            "seq4_256x8_w8": (1024, 8, False, 256, 8, True),
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
def _mhc_post_c500_sequence4_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
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
    """C500 channels-last kernel owning all four output channels per tile."""
    pid = tl.program_id(0)
    s_block = pid % S_BLOCKS
    pid = pid // S_BLOCKS
    h_block = pid % H_BLOCKS
    batch = pid // H_BLOCKS

    s = s_block * BLOCK_S + tl.arange(0, BLOCK_S)[None, :]
    h = h_block * BLOCK_H_TILE + tl.arange(0, BLOCK_H_TILE)[:, None]
    s_mask = s < seq_size
    mask = s_mask & (h < hidden_size)

    x_base = batch * x_stride_0 + s * x_stride_1 + h * x_stride_h
    # Keep the reused tiles in native BF16 registers.  Multiplication by the
    # FP32 coefficients promotes the arithmetic exactly where it is needed.
    x = tl.load(x_ptr + x_base, mask=mask, other=0.0)
    post_base = batch * post_stride_0 + s * post_stride_1
    comb_base = batch * comb_stride_0 + s * comb_stride_1
    output_base = batch * out_stride_0 + s * out_stride_1 + h * out_stride_h

    # Build four output accumulators, then stream one residual channel at a
    # time through all of them.  This mirrors the well-behaved portable
    # multi-accumulator lowering and keeps only one residual tile live.
    p0 = tl.load(post_ptr + post_base, mask=s_mask, other=0.0).to(tl.float32)
    p1 = tl.load(
        post_ptr + post_base + post_stride_m, mask=s_mask, other=0.0
    ).to(tl.float32)
    p2 = tl.load(
        post_ptr + post_base + 2 * post_stride_m, mask=s_mask, other=0.0
    ).to(tl.float32)
    p3 = tl.load(
        post_ptr + post_base + 3 * post_stride_m, mask=s_mask, other=0.0
    ).to(tl.float32)

    acc0 = x * p0
    acc1 = x * p1
    acc2 = x * p2
    acc3 = x * p3

    residual_base = (
        batch * residual_stride_0 + s * residual_stride_1 + h * residual_stride_h
    )
    r = tl.load(residual_ptr + residual_base, mask=mask, other=0.0)
    c00 = tl.load(comb_ptr + comb_base, mask=s_mask, other=0.0).to(tl.float32)
    c01 = tl.load(
        comb_ptr + comb_base + comb_stride_n, mask=s_mask, other=0.0
    ).to(tl.float32)
    c02 = tl.load(
        comb_ptr + comb_base + 2 * comb_stride_n, mask=s_mask, other=0.0
    ).to(tl.float32)
    c03 = tl.load(
        comb_ptr + comb_base + 3 * comb_stride_n, mask=s_mask, other=0.0
    ).to(tl.float32)
    acc0 += r * c00
    acc1 += r * c01
    acc2 += r * c02
    acc3 += r * c03

    r = tl.load(
        residual_ptr + residual_base + residual_stride_m,
        mask=mask,
        other=0.0,
    )
    c10 = tl.load(
        comb_ptr + comb_base + comb_stride_m, mask=s_mask, other=0.0
    ).to(tl.float32)
    c13 = tl.load(
        comb_ptr + comb_base + comb_stride_m + 3 * comb_stride_n,
        mask=s_mask,
        other=0.0,
    ).to(tl.float32)
    c11 = tl.load(
        comb_ptr + comb_base + comb_stride_m + comb_stride_n,
        mask=s_mask,
        other=0.0,
    ).to(tl.float32)
    c12 = tl.load(
        comb_ptr + comb_base + comb_stride_m + 2 * comb_stride_n,
        mask=s_mask,
        other=0.0,
    ).to(tl.float32)
    acc0 += r * c10
    acc1 += r * c11
    acc2 += r * c12
    acc3 += r * c13

    r = tl.load(
        residual_ptr + residual_base + 2 * residual_stride_m,
        mask=mask,
        other=0.0,
    )
    c20 = tl.load(
        comb_ptr + comb_base + 2 * comb_stride_m, mask=s_mask, other=0.0
    ).to(tl.float32)
    c21 = tl.load(
        comb_ptr + comb_base + 2 * comb_stride_m + comb_stride_n,
        mask=s_mask,
        other=0.0,
    ).to(tl.float32)
    c22 = tl.load(
        comb_ptr + comb_base + 2 * comb_stride_m + 2 * comb_stride_n,
        mask=s_mask,
        other=0.0,
    ).to(tl.float32)
    c23 = tl.load(
        comb_ptr + comb_base + 2 * comb_stride_m + 3 * comb_stride_n,
        mask=s_mask,
        other=0.0,
    ).to(tl.float32)
    acc0 += r * c20
    acc1 += r * c21
    acc2 += r * c22
    acc3 += r * c23

    r = tl.load(
        residual_ptr + residual_base + 3 * residual_stride_m,
        mask=mask,
        other=0.0,
    )
    c30 = tl.load(
        comb_ptr + comb_base + 3 * comb_stride_m, mask=s_mask, other=0.0
    ).to(tl.float32)
    c31 = tl.load(
        comb_ptr + comb_base + 3 * comb_stride_m + comb_stride_n,
        mask=s_mask,
        other=0.0,
    ).to(tl.float32)
    c32 = tl.load(
        comb_ptr + comb_base + 3 * comb_stride_m + 2 * comb_stride_n,
        mask=s_mask,
        other=0.0,
    ).to(tl.float32)
    c33 = tl.load(
        comb_ptr + comb_base + 3 * comb_stride_m + 3 * comb_stride_n,
        mask=s_mask,
        other=0.0,
    ).to(tl.float32)
    acc0 += r * c30
    acc1 += r * c31
    acc2 += r * c32
    acc3 += r * c33

    tl.store(out_ptr + output_base, acc0, mask=mask)
    tl.store(out_ptr + output_base + out_stride_m, acc1, mask=mask)
    tl.store(out_ptr + output_base + 2 * out_stride_m, acc2, mask=mask)
    tl.store(out_ptr + output_base + 3 * out_stride_m, acc3, mask=mask)


@triton.jit
def _mhc_post_row_persistent_kernel(
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
    """Reuse row coefficients while iterating over hidden-vector tiles.

    The launch policy is deliberately host-specific.  A2 launches one
    program per physical vector core and lets it consume multiple rows.  The
    H200 opt-in launches one program per row, so its official H=1280 shape and
    BLOCK_H=256 execute five tiles while keeping the four post coefficients
    and sixteen mixing coefficients live only once.
    """
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
    trusted: bool = False,
) -> torch.Tensor:
    tensors = (x, residual, post_layer_mix, comb_res_mix)
    if not trusted:
        if x.ndim != 3 or residual.ndim != 4:
            raise ValueError("expected x [n0,n1,h] and residual [n0,n1,m,h]")
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
    else:
        n0, n1, hidden_size = x.shape
        hc_mult = residual.shape[2]

    if len(kernel_config) == 5:
        block_h, num_warps, persistent, block_s, block_h_tile = kernel_config
        multi_channel = False
    else:
        block_h, num_warps, persistent, block_s, block_h_tile, multi_channel = (
            kernel_config
        )
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

    if c500_sequence_layout and multi_channel:
        s_blocks = triton.cdiv(n1, block_s)
        h_blocks = triton.cdiv(hidden_size, block_h_tile)
        grid = (n0 * h_blocks * s_blocks,)
        _mhc_post_c500_sequence4_kernel[grid](
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            out,
            hidden_size=hidden_size,
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

    elif c500_sequence_layout:
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
        if kernel_profile in ("mthreads_ph100", "nvidia_h200"):
            # One persistent program owns complete rows.  At the official shape
            # a 56-program grid processes the 8,192 rows instead of direct
            # BH256's 40,960-program grid, while retaining device-wide parallelism.
            num_programs = (
                min(n0 * n1, 56)
                if kernel_profile == "mthreads_ph100" and persistent == 56
                else n0 * n1
            )
        else:
            device_index = x.device.index if x.device.index is not None else 0
            num_programs = min(
                n0 * n1,
                _ascend_vector_core_count(device_index),
            )
        h_blocks = triton.cdiv(hidden_size, block_h)
        _mhc_post_row_persistent_kernel[(num_programs,)](
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
        self._trusted_hot_path = _TrustedHotPath()
        self._trusted_validated = False

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        changed = self._trusted_hot_path.changed(
            (x, residual, post_layer_mix, comb_res_mix)
        )
        trusted = not changed and self._trusted_validated
        out = mhc_post(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            self._kernel_config,
            self.kernel_profile,
            trusted=trusted,
        )
        self._trusted_validated = True
        return out


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
