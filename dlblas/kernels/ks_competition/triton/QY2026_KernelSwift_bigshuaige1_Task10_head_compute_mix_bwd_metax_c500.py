"""KernelSwift Task10 fused sigmoid-backward reductions for metax_c500."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


PLATFORM = "metax_c500"
ROW_WARPS = 4


@triton.jit
def _mix_bwd_kernel(input_mix, scale_ptr, base_ptr, grad_out, grad_input, grad_base, scale_parts, rows: tl.constexpr, BLOCK: tl.constexpr):
    component = tl.program_id(0)
    row = tl.arange(0, BLOCK)
    valid = row < rows
    offset = row * 4 + component
    x = tl.load(input_mix + offset, mask=valid, other=0.0).to(tl.float32)
    go = tl.load(grad_out + offset, mask=valid, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    base = tl.load(base_ptr + component).to(tl.float32)
    z = x * scale + base
    sigmoid = 1.0 / (1.0 + tl.exp(-z))
    grad_z = go * sigmoid * (1.0 - sigmoid)
    tl.store(grad_input + offset, grad_z * scale, mask=valid)
    tl.store(grad_base + component, tl.sum(grad_z, axis=0))
    tl.store(scale_parts + component, tl.sum(grad_z * x, axis=0))


@triton.jit
def _sum_scale_kernel(parts, output):
    offsets = tl.arange(0, 4)
    tl.store(output, tl.sum(tl.load(parts + offsets), axis=0))


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_mix: torch.Tensor, mhc_scale: torch.Tensor, mhc_base: torch.Tensor, grad_out: torch.Tensor):
        grad_input = torch.empty_like(input_mix)
        grad_base = torch.empty_like(mhc_base)
        parts = torch.empty(4, dtype=torch.float32, device=input_mix.device)
        grad_scale = torch.empty_like(mhc_scale)
        rows = input_mix.numel() // 4
        _mix_bwd_kernel[(4,)](input_mix, mhc_scale, mhc_base, grad_out, grad_input, grad_base, parts, rows=rows, BLOCK=2048, num_warps=ROW_WARPS, num_stages=1)
        _sum_scale_kernel[(1,)](parts, grad_scale, num_warps=1, num_stages=1)
        return grad_input, grad_scale, grad_base


class Model(ModelNew):
    pass


def get_inputs():
    return [torch.randn(2, 1024, 4, dtype=torch.float32), torch.randn(1, dtype=torch.float32), torch.randn(4, dtype=torch.float32), torch.randn(2, 1024, 4, dtype=torch.float32)]


def get_init_inputs():
    return []
