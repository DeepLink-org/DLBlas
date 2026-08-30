"""KernelSwift Task10 chunked sigmoid backward for Ascend A2.

The 8192-element benchmark is split into small partial reductions so the
intermediate state remains within the 910B unified buffer.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


PLATFORM = "ascend_a2"
ROW_WARPS = 1
PARTIAL_BLOCK = 256


@triton.jit
def _mix_bwd_chunk_kernel(
    input_mix,
    scale_ptr,
    base_ptr,
    grad_out,
    grad_input,
    partial_scale,
    partial_base,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    valid = off < n_elements
    component = off % 4
    x = tl.load(input_mix + off, mask=valid, other=0.0).to(tl.float32)
    go = tl.load(grad_out + off, mask=valid, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    base = tl.load(base_ptr + component).to(tl.float32)
    z = x * scale + base
    sigmoid = 1.0 / (1.0 + tl.exp(-z))
    grad_z = go * sigmoid * (1.0 - sigmoid)
    tl.store(grad_input + off, grad_z * scale, mask=valid)

    c = tl.arange(0, 4)
    base_grads = tl.sum(
        tl.where(
            valid[:, None] & (component[:, None] == c[None, :]), grad_z[:, None], 0.0
        ),
        axis=0,
    )
    tl.store(partial_base + pid * 4 + c, base_grads)
    tl.store(partial_scale + pid, tl.sum(tl.where(valid, grad_z * x, 0.0), axis=0))


@triton.jit
def _mix_bwd_reduce_kernel(
    partial_scale,
    partial_base,
    grad_scale,
    grad_base,
    n_partials: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.arange(0, BLOCK)
    valid = off < n_partials
    ps = tl.load(partial_scale + off, mask=valid, other=0.0).to(tl.float32)
    c = tl.arange(0, 4)
    pb = tl.load(
        partial_base + off[:, None] * 4 + c[None, :], mask=valid[:, None], other=0.0
    ).to(tl.float32)
    tl.store(grad_scale, tl.sum(ps, axis=0))
    tl.store(grad_base + c, tl.sum(pb, axis=0))


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_mix, mhc_scale, mhc_base, grad_out):
        grad_input = torch.empty_like(input_mix)
        n_elements = input_mix.numel()
        n_partials = triton.cdiv(n_elements, PARTIAL_BLOCK)
        partial_scale = torch.empty(
            (n_partials,), dtype=input_mix.dtype, device=input_mix.device
        )
        partial_base = torch.empty(
            (n_partials, 4), dtype=input_mix.dtype, device=input_mix.device
        )
        _mix_bwd_chunk_kernel[(n_partials,)](
            input_mix,
            mhc_scale,
            mhc_base,
            grad_out,
            grad_input,
            partial_scale,
            partial_base,
            n_elements=n_elements,
            BLOCK=PARTIAL_BLOCK,
            num_warps=ROW_WARPS,
            num_stages=1,
        )
        grad_scale = torch.empty_like(mhc_scale)
        grad_base = torch.empty_like(mhc_base)
        _mix_bwd_reduce_kernel[(1,)](
            partial_scale,
            partial_base,
            grad_scale,
            grad_base,
            n_partials=n_partials,
            BLOCK=triton.next_power_of_2(n_partials),
            num_warps=ROW_WARPS,
            num_stages=1,
        )
        return grad_input, grad_scale, grad_base


class Model(ModelNew):
    pass


def get_inputs():
    return [
        torch.randn(2, 1024, 4, dtype=torch.float32, device="npu"),
        torch.randn(1, dtype=torch.float32, device="npu"),
        torch.randn(4, dtype=torch.float32, device="npu"),
        torch.randn(2, 1024, 4, dtype=torch.float32, device="npu"),
    ]


def get_init_inputs():
    return []
