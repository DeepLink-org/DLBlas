"""Zhenwu 810E Task07: fuse all four output components per hidden tile."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


BLOCK_H = 256
KERNEL_WARPS = 2


@triton.jit
def _mhc_post_fused4_kernel(x_ptr, residual_ptr, post_ptr, comb_ptr, out_ptr, n_tokens: tl.constexpr, hidden_size: tl.constexpr, BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    tiles_h = (hidden_size + BLOCK_H - 1) // BLOCK_H
    tile_h = pid % tiles_h
    token_flat = pid // tiles_h
    h = tile_h * BLOCK_H + tl.arange(0, BLOCK_H)
    valid = h < hidden_size
    x = tl.load(x_ptr + token_flat * hidden_size + h, mask=valid, other=0.0).to(tl.float32)
    c = tl.arange(0, 4)
    post = tl.load(post_ptr + token_flat * 4 + c).to(tl.float32)
    acc = post[:, None] * x[None, :]
    for j in range(4):
        r = tl.load(residual_ptr + token_flat * 4 * hidden_size + j * hidden_size + h, mask=valid, other=0.0).to(tl.float32)
        mix = tl.load(comb_ptr + token_flat * 16 + j * 4 + c).to(tl.float32)
        acc += mix[:, None] * r[None, :]
    tl.store(out_ptr + token_flat * 4 * hidden_size + c[:, None] * hidden_size + h[None, :], acc.to(tl.bfloat16), mask=valid[None, :])


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, residual, post_layer_mix, comb_res_mix):
        b, s, h = x.shape
        out = torch.empty((b, s, 4, h), dtype=torch.bfloat16, device=x.device)
        _mhc_post_fused4_kernel[(b * s * triton.cdiv(h, BLOCK_H),)](x, residual, post_layer_mix, comb_res_mix, out, n_tokens=s, hidden_size=h, BLOCK_H=BLOCK_H, num_warps=KERNEL_WARPS, num_stages=1)
        return out


class Model(ModelNew):
    pass


def get_inputs():
    torch.manual_seed(0)
    return [torch.randn((2, 4096, 1280), dtype=torch.bfloat16, device="cuda"), torch.randn((2, 4096, 4, 1280), dtype=torch.bfloat16, device="cuda"), torch.randn((2, 4096, 4, 1), dtype=torch.float32, device="cuda"), torch.randn((2, 4096, 4, 4), dtype=torch.float32, device="cuda")]


def get_init_inputs():
    return []
