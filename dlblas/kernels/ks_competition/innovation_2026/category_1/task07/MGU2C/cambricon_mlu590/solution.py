"""Fused MLU590-oriented Triton 3.2 implementation of Task07."""

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


# (BLOCK_S, BLOCK_H, num_warps)
_CHIP_CONFIGS = {
    "auto": (2, 128, 1),
    # Sixteen adjacent rows share each program while 256 hidden values remain
    # contiguous. This keeps the coefficient reuse of the large tile while
    # reducing the live set and exposing 2,560 fixed owners at the official
    # shape; the larger S32 tile remains an opt-in bandwidth experiment.
    "cambricon_mlu590": (16, 256, 1),
}


def _get_launch_config():
    chip = os.environ.get("KERNELSWIFT_CHIP", "cambricon_mlu590").strip().lower()
    config = _CHIP_CONFIGS.get(chip, _CHIP_CONFIGS["auto"])
    if chip == "cambricon_mlu590":
        variants = {
            "s1h128x1": (1, 128, 1),
            "s1h256x1": (1, 256, 1),
            "s2h256x1": (2, 256, 1),
            "s4h128x1": (4, 128, 1),
            "s4h256x1": (4, 256, 1),
            "s4h256x4": (4, 256, 4),
            "s8h128x1": (8, 128, 1),
            "s8h128x4": (8, 128, 4),
            "s8h256x1": (8, 256, 1),
            "s16h64x1": (16, 64, 1),
            "s16h256x1": (16, 256, 1),
            "s16h256x4": (16, 256, 4),
            "s32h256x1": (32, 256, 1),
        }
        key = os.environ.get("KERNELSWIFT_TASK07_CONFIG", "s16h256x1").strip().lower()
        config = variants.get(key, config)
    return config


@triton.jit
def _mhc_post_tiled_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    seq_size: tl.constexpr,
    S_BLOCKS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    batch_sequence_block = tl.program_id(0)
    hidden_block = tl.program_id(1)
    batch = batch_sequence_block // S_BLOCKS
    sequence_block = batch_sequence_block % S_BLOCKS

    sequence = sequence_block * BLOCK_S + tl.arange(0, BLOCK_S)[:, None]
    hidden = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H)[None, :]
    row = batch * seq_size + sequence
    row_mask = sequence < seq_size
    mask = row_mask & (hidden < hidden_size)

    x = tl.load(
        x_ptr + row * hidden_size + hidden,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    post_base = row * hc_mult
    post0 = tl.load(post_ptr + post_base, mask=row_mask, other=0.0).to(tl.float32)
    post1 = tl.load(post_ptr + post_base + 1, mask=row_mask, other=0.0).to(tl.float32)
    post2 = tl.load(post_ptr + post_base + 2, mask=row_mask, other=0.0).to(tl.float32)
    post3 = tl.load(post_ptr + post_base + 3, mask=row_mask, other=0.0).to(tl.float32)
    # The contraction size is an official constant (4), so keep all four
    # paths explicit.  In particular there is no Python list/list-comprehension
    # in the JIT body: Triton 3.2 rejects the dynamic accumulator-list pattern
    residual_base = row * hc_mult * hidden_size + hidden
    comb_row_base = row * hc_mult * hc_mult

    residual0 = tl.load(
        residual_ptr + residual_base,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    c00 = tl.load(comb_ptr + comb_row_base, mask=row_mask, other=0.0).to(tl.float32)
    c01 = tl.load(comb_ptr + comb_row_base + 1, mask=row_mask, other=0.0).to(tl.float32)
    c02 = tl.load(comb_ptr + comb_row_base + 2, mask=row_mask, other=0.0).to(tl.float32)
    c03 = tl.load(comb_ptr + comb_row_base + 3, mask=row_mask, other=0.0).to(tl.float32)
    acc0 = residual0 * c00
    acc1 = residual0 * c01
    acc2 = residual0 * c02
    acc3 = residual0 * c03

    residual1 = tl.load(
        residual_ptr + residual_base + hidden_size,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    c10 = tl.load(comb_ptr + comb_row_base + 4, mask=row_mask, other=0.0).to(tl.float32)
    c11 = tl.load(comb_ptr + comb_row_base + 5, mask=row_mask, other=0.0).to(tl.float32)
    c12 = tl.load(comb_ptr + comb_row_base + 6, mask=row_mask, other=0.0).to(tl.float32)
    c13 = tl.load(comb_ptr + comb_row_base + 7, mask=row_mask, other=0.0).to(tl.float32)
    acc0 += residual1 * c10
    acc1 += residual1 * c11
    acc2 += residual1 * c12
    acc3 += residual1 * c13

    residual2 = tl.load(
        residual_ptr + residual_base + 2 * hidden_size,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    c20 = tl.load(comb_ptr + comb_row_base + 8, mask=row_mask, other=0.0).to(tl.float32)
    c21 = tl.load(comb_ptr + comb_row_base + 9, mask=row_mask, other=0.0).to(tl.float32)
    c22 = tl.load(comb_ptr + comb_row_base + 10, mask=row_mask, other=0.0).to(
        tl.float32
    )
    c23 = tl.load(comb_ptr + comb_row_base + 11, mask=row_mask, other=0.0).to(
        tl.float32
    )
    acc0 += residual2 * c20
    acc1 += residual2 * c21
    acc2 += residual2 * c22
    acc3 += residual2 * c23

    residual3 = tl.load(
        residual_ptr + residual_base + 3 * hidden_size,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    c30 = tl.load(comb_ptr + comb_row_base + 12, mask=row_mask, other=0.0).to(
        tl.float32
    )
    c31 = tl.load(comb_ptr + comb_row_base + 13, mask=row_mask, other=0.0).to(
        tl.float32
    )
    c32 = tl.load(comb_ptr + comb_row_base + 14, mask=row_mask, other=0.0).to(
        tl.float32
    )
    c33 = tl.load(comb_ptr + comb_row_base + 15, mask=row_mask, other=0.0).to(
        tl.float32
    )
    acc0 += residual3 * c30
    acc1 += residual3 * c31
    acc2 += residual3 * c32
    acc3 += residual3 * c33

    # Match platform_input.py exactly: finish residual contraction first, then
    # add the x*post branch, and round once on the BF16 output store.
    acc0 += x * post0
    acc1 += x * post1
    acc2 += x * post2
    acc3 += x * post3

    output_base = row * hc_mult * hidden_size + hidden
    tl.store(out_ptr + output_base, acc0, mask=mask)
    tl.store(out_ptr + output_base + hidden_size, acc1, mask=mask)
    tl.store(out_ptr + output_base + 2 * hidden_size, acc2, mask=mask)
    tl.store(out_ptr + output_base + 3 * hidden_size, acc3, mask=mask)


def mhc_post(x, residual, post_layer_mix, comb_res_mix, kernel_config):
    if x.ndim != 3 or residual.ndim != 4:
        raise ValueError("expected x [n0,n1,h] and residual [n0,n1,m,h]")
    if (
        x.device.type == "cpu"
        or residual.device.type == "cpu"
        or post_layer_mix.device.type == "cpu"
        or comb_res_mix.device.type == "cpu"
    ):
        raise ValueError("Task07 requires accelerator tensors supported by Triton")
    if (
        residual.device != x.device
        or post_layer_mix.device != x.device
        or comb_res_mix.device != x.device
    ):
        raise ValueError("all inputs must be on the same accelerator")

    n0, n1, hidden_size = x.shape
    hc_mult = residual.shape[2]
    if hc_mult != 4:
        raise ValueError("the official Task07 contract fixes hc_mult=4")
    if residual.shape != (n0, n1, hc_mult, hidden_size):
        raise ValueError("residual shape is inconsistent with x")
    if post_layer_mix.shape != (n0, n1, hc_mult, 1):
        raise ValueError("post_layer_mix must have shape [n0,n1,4,1]")
    if comb_res_mix.shape != (n0, n1, hc_mult, hc_mult):
        raise ValueError("comb_res_mix must have shape [n0,n1,4,4]")

    output = torch.empty(
        (n0, n1, hc_mult, hidden_size),
        device=x.device,
        dtype=torch.bfloat16,
    )
    block_s, block_h, num_warps = kernel_config
    s_blocks = triton.cdiv(n1, block_s)
    grid = (n0 * s_blocks, triton.cdiv(hidden_size, block_h))
    _mhc_post_tiled_kernel[grid](
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        output,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        seq_size=n1,
        S_BLOCKS=s_blocks,
        BLOCK_S=block_s,
        BLOCK_H=block_h,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._kernel_config = _get_launch_config()

    def forward(self, x, residual, post_layer_mix, comb_res_mix):
        return mhc_post(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            self._kernel_config,
        )


class Model(ModelNew):
    """Compatibility name for runners that import ``Model``."""


n0 = 2
n1 = 4096
h = 1280
mhc_mult = 4


def get_inputs():
    device = _target_device()
    return [
        torch.randn((n0, n1, h), dtype=torch.bfloat16, device=device),
        torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16, device=device),
        torch.randn((n0, n1, mhc_mult, 1), dtype=torch.float32, device=device),
        torch.randn(
            (n0, n1, mhc_mult, mhc_mult), dtype=torch.float32, device=device
        ),
    ]


def get_init_inputs():
    return []
