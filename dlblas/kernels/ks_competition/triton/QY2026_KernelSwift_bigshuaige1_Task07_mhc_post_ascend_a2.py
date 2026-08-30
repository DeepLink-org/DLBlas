"""KernelSwift Task07 fused mHC post operation for Ascend A2.

One program owns a 128-wide hidden tile for one (batch, sequence, mix) row.
The four residual terms stay in registers and the result is written as BF16.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


PLATFORM = "ascend_a2"


@triton.jit
def _mhc_post_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // NUM_TILES
    tile = pid - row * NUM_TILES
    token = row // 4
    mix_idx = row - token * 4
    h = tile * BLOCK + tl.arange(0, BLOCK)
    valid = (row < n_rows) & (h < 1280)

    x = tl.load(x_ptr + token * 1280 + h, mask=valid, other=0.0).to(tl.float32)
    post = tl.load(post_ptr + token * 4 + mix_idx, mask=row < n_rows, other=0.0).to(
        tl.float32
    )
    acc = x * post
    for j in range(4):
        residual = tl.load(
            residual_ptr + token * 4 * 1280 + j * 1280 + h,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        coeff = tl.load(
            comb_ptr + token * 16 + j * 4 + mix_idx,
            mask=row < n_rows,
            other=0.0,
        ).to(tl.float32)
        acc += coeff * residual
    tl.store(
        out_ptr + token * 4 * 1280 + mix_idx * 1280 + h,
        acc.to(tl.bfloat16),
        mask=valid,
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, residual, post_layer_mix, comb_res_mix):
        b, s, hidden = x.shape
        out = torch.empty((b, s, 4, hidden), dtype=torch.bfloat16, device=x.device)
        num_tiles = triton.cdiv(hidden, 128)
        _mhc_post_kernel[(b * s * 4 * num_tiles,)](
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            out,
            n_rows=b * s * 4,
            NUM_TILES=num_tiles,
            BLOCK=128,
            num_warps=1,
            num_stages=1,
        )
        return out


class Model(ModelNew):
    pass


def get_inputs():
    torch.manual_seed(0)
    return [
        torch.randn((2, 4096, 1280), dtype=torch.bfloat16, device="npu"),
        torch.randn((2, 4096, 4, 1280), dtype=torch.bfloat16, device="npu"),
        torch.randn((2, 4096, 4, 1), dtype=torch.float32, device="npu"),
        torch.randn((2, 4096, 4, 4), dtype=torch.float32, device="npu"),
    ]


def get_init_inputs():
    return []
