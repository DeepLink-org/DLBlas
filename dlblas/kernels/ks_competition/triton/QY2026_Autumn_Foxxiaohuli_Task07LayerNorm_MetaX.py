import os

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(
    x_ptr,
    y_ptr,
    stride_x,
    stride_y,
    eps: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    row_x_ptr = x_ptr + pid * stride_x + offs
    vals = tl.load(row_x_ptr, mask=mask, other=0.0)

    invD = 1.0 / D
    mean = tl.sum(vals, axis=0) * invD
    mean_sq = tl.sum(vals * vals, axis=0) * invD
    var = mean_sq - mean * mean

    inv_std = tl.math.rsqrt(var + eps)
    shift = -mean * inv_std
    out = vals * inv_std + shift

    row_y_ptr = y_ptr + pid * stride_y + offs
    tl.store(row_y_ptr, out, mask=mask)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self._y = None
        self._rows = -1
        self._D = -1

    def forward(self, x):
        D = x.shape[-1]
        x_2d = x.view(-1, D) if x.is_contiguous() else x.contiguous().view(-1, D)
        B = x_2d.shape[0]

        if self._rows != B or self._D != D:
            self._y = torch.empty(B, D, device=x.device, dtype=x.dtype)
            self._rows = B
            self._D = D

        _layernorm_kernel[(B,)](
            x_2d,
            self._y,
            D,
            D,
            eps=1e-5,
            D=D,
            BLOCK_SIZE=max(32, ((D + 31) // 32) * 32),
            num_warps=1,
            num_stages=1,
        )

        return self._y.view(x.shape)


# ==========================================
# Hyperparameters & Data Generation
# ==========================================


def get_inputs():
    D = int(os.environ.get("LN_DIM", "10"))
    x = torch.rand(10, D)
    return [x]


def get_init_inputs():
    return []
