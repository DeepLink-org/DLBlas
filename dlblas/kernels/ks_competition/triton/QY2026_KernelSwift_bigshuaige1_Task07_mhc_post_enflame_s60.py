"""Task07 fused FP32 mix and BF16 cast candidate for Enflame S60."""

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl


PLATFORM = "enflame_s60"
BLOCK_H = 256
ROW_WARPS = 1


@triton.jit
def _mhc_post_fused4_kernel(x_ptr, residual_ptr, post_ptr, comb_ptr,
                            out_ptr, total_tokens: tl.constexpr,
                            hidden_size: tl.constexpr, BLOCK_T: tl.constexpr,
                            BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    tiles_h = (hidden_size + BLOCK_H - 1) // BLOCK_H
    tile_h = pid % tiles_h
    token_tile = pid // tiles_h
    token_flat = token_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    h = tile_h * BLOCK_H + tl.arange(0, BLOCK_H)
    valid_t = token_flat < total_tokens
    valid_h = h < hidden_size
    x = tl.load(x_ptr + token_flat[:, None] * hidden_size + h[None, :],
                mask=valid_t[:, None] & valid_h[None, :], other=0.0).to(tl.float32)
    c = tl.arange(0, 4)
    post = tl.load(post_ptr + token_flat[:, None] * 4 + c[None, :],
                   mask=valid_t[:, None], other=0.0).to(tl.float32)
    acc = post[:, :, None] * x[:, None, :]
    for j in range(4):
        r = tl.load(
            residual_ptr + token_flat[:, None] * 4 * hidden_size
            + j * hidden_size + h[None, :],
            mask=valid_t[:, None] & valid_h[None, :], other=0.0,
        ).to(tl.float32)
        mix = tl.load(comb_ptr + token_flat[:, None] * 16 + j * 4 + c[None, :],
                      mask=valid_t[:, None], other=0.0).to(tl.float32)
        acc += mix[:, :, None] * r[:, None, :]
    tl.store(
        out_ptr + token_flat[:, None, None] * 4 * hidden_size
        + c[None, :, None] * hidden_size + h[None, None, :],
        acc.to(tl.bfloat16),
        mask=valid_t[:, None, None] & valid_h[None, None, :],
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, residual, post_layer_mix, comb_res_mix):
        b, s, h = x.shape
        out = torch.empty((b, s, 4, h), dtype=torch.bfloat16, device=x.device)
        block_t = int(os.getenv("S60_T7_BLOCK_T", "64"))
        block_h = int(os.getenv("S60_T7_BLOCK_H", "256"))
        warps = int(os.getenv("S60_T7_WARPS", "1"))
        _mhc_post_fused4_kernel[
            (triton.cdiv(b * s, block_t) * triton.cdiv(h, block_h),)
        ](
            x, residual, post_layer_mix, comb_res_mix, out,
            total_tokens=b * s, hidden_size=h, BLOCK_T=block_t, BLOCK_H=block_h,
            num_warps=warps, num_stages=1,
        )
        return out


class Model(ModelNew):
    pass


def get_inputs():
    torch.manual_seed(0)
    return [
        torch.randn((2, 4096, 1280), dtype=torch.bfloat16, device="cuda"),
        torch.randn((2, 4096, 4, 1280), dtype=torch.bfloat16, device="cuda"),
        torch.randn((2, 4096, 4, 1), dtype=torch.float32, device="cuda"),
        torch.randn((2, 4096, 4, 4), dtype=torch.float32, device="cuda"),
    ]


def get_init_inputs():
    return []
