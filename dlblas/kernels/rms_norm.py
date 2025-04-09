# https://github.com/InternLM/lmdeploy/blob/v0.6.1/lmdeploy/pytorch/kernels/cuda/rms_norm.py
import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _compute_rms_norm(x, w, eps: tl.constexpr, N_COLS: tl.constexpr):
    """compute rms norm."""
    xf = x.to(tl.float32)

    var = tl.sum(xf * xf, 0) * float(1.0 / N_COLS)
    out = xf * tl.math.rsqrt(var + eps)
    out = (w * out).to(x.dtype)
    return out


@triton.jit
def rms_norm_kernel(
    input,
    weight,
    output,
    input_row_stride: tl.constexpr,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """rms norm kernel."""
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    w = tl.load(weight + offsets, mask=offsets < N_COLS, other=0.0)

    x_ptr = input + prog_id * input_row_stride
    x = tl.load(x_ptr + offsets, mask=offsets < N_COLS, other=0.0)
    out = _compute_rms_norm(x, w, eps, N_COLS)

    out_ptr = output + prog_id * input_row_stride
    tl.store(out_ptr + offsets, out, mask=offsets < N_COLS)


@triton.jit
def add_rms_norm_kernel(
    input,
    weight,
    residual,
    output,
    out_residual,
    input_row_stride: tl.constexpr,
    residual_row_stride: tl.constexpr,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """rms norm kernel."""
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    w = tl.load(weight + offsets, mask=offsets < N_COLS, other=0.0)

    x_ptr = input + prog_id * input_row_stride
    x = tl.load(x_ptr + offsets, mask=offsets < N_COLS, other=0.0)

    res_ptr = residual + prog_id * residual_row_stride
    res = tl.load(res_ptr + offsets, mask=offsets < N_COLS, other=0.0)

    new_x = x + res
    out_res_ptr = out_residual + prog_id * residual_row_stride
    tl.store(out_res_ptr + offsets, new_x, mask=offsets < N_COLS)

    out = _compute_rms_norm(new_x, w, eps, N_COLS)

    out_ptr = output + prog_id * input_row_stride
    tl.store(out_ptr + offsets, out, mask=offsets < N_COLS)


def rms_norm(
    hidden_states: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
    residual: Tensor = None,
    out: Tensor = None,
    out_residual: Tensor = None,
):
    """rms norm."""
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()

    feat_size = weight.shape[0]
    seq_len = hidden_states.numel() // hidden_states.size(-1)
    input_stride = hidden_states.stride(-2)

    BLOCK_N = triton.next_power_of_2(feat_size)

    if out is None:
        out = torch.empty_like(hidden_states)

    grid = (seq_len,)

    if residual is None:
        rms_norm_kernel[grid](
            hidden_states,
            weight,
            out,
            input_row_stride=input_stride,
            eps=eps,
            N_COLS=feat_size,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2,
        )
        return out
    else:
        if out_residual is None:
            out_residual = torch.empty_like(hidden_states)

        res_stride = residual.stride(-2)
        add_rms_norm_kernel[grid](
            hidden_states,
            weight,
            residual,
            out,
            out_residual,
            input_row_stride=input_stride,
            residual_row_stride=res_stride,
            eps=eps,
            N_COLS=feat_size,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2,
        )
        return out, out_residual

