"""KernelSwift Task10 single-kernel sigmoid backward for hygon_bw1000."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


PLATFORM = "tianshu_bi150"
ROW_WARPS = 8


@triton.jit
def _mix_bwd_single_kernel(
    input_mix,
    scale_ptr,
    base_ptr,
    grad_out,
    grad_input,
    grad_scale,
    grad_base,
    BLOCK: tl.constexpr,
):
    offset = tl.arange(0, BLOCK)
    component = offset % 4
    x = tl.load(input_mix + offset).to(tl.float32)
    go = tl.load(grad_out + offset).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    base = tl.load(base_ptr + component).to(tl.float32)
    z = x * scale + base
    sigmoid = 1.0 / (1.0 + tl.exp(-z))
    grad_z = go * sigmoid * (1.0 - sigmoid)
    tl.store(grad_input + offset, grad_z * scale)
    g0 = tl.sum(tl.where(component == 0, grad_z, 0.0), axis=0)
    g1 = tl.sum(tl.where(component == 1, grad_z, 0.0), axis=0)
    g2 = tl.sum(tl.where(component == 2, grad_z, 0.0), axis=0)
    g3 = tl.sum(tl.where(component == 3, grad_z, 0.0), axis=0)
    c = tl.arange(0, 4)
    base_grad = tl.where(c == 0, g0, tl.where(c == 1, g1, tl.where(c == 2, g2, g3)))
    tl.store(grad_base + c, base_grad)
    tl.store(grad_scale, tl.sum(grad_z * x, axis=0))


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_mix: torch.Tensor,
        mhc_scale: torch.Tensor,
        mhc_base: torch.Tensor,
        grad_out: torch.Tensor,
    ):
        grad_input = torch.empty_like(input_mix)
        grad_scale = torch.empty_like(mhc_scale)
        grad_base = torch.empty_like(mhc_base)
        _mix_bwd_single_kernel[(1,)](
            input_mix,
            mhc_scale,
            mhc_base,
            grad_out,
            grad_input,
            grad_scale,
            grad_base,
            BLOCK=8192,
            num_warps=ROW_WARPS,
            num_stages=1,
        )
        return grad_input, grad_scale, grad_base


class Model(ModelNew):
    pass


def get_inputs():
    return [
        torch.randn(2, 1024, 4, dtype=torch.float32),
        torch.randn(1, dtype=torch.float32),
        torch.randn(4, dtype=torch.float32),
        torch.randn(2, 1024, 4, dtype=torch.float32),
    ]


def get_init_inputs():
    return []
