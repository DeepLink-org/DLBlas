import torch
import triton
from torch import Tensor
import triton.language as tl
from dlblas.utils.device_utils import NUM_CORES


@triton.jit
def rms_norm_kernel(
    input,
    weight,
    output,
    n_rows,
    input_row_stride: tl.constexpr,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, N_COLS)
    w = tl.load(weight + offsets)
    for row_idx in range(pid, n_rows, NUM_CORES):
        x_ptr = input + row_idx * input_row_stride
        x = tl.load(x_ptr + offsets)
        xf = x.to(tl.float32)
        var = tl.sum(xf * xf, 0) / N_COLS
        out = xf * tl.math.rsqrt(var + eps)
        out = w * out.to(x.dtype)
        out_ptr = output + row_idx * input_row_stride
        tl.store(out_ptr + offsets, out)


@triton.jit
def rms_norm_block_kernel(
    input,
    weight,
    output,
    n_rows,
    input_row_stride: tl.constexpr,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    cols_offsets = tl.arange(0, N_COLS)
    w = tl.load(weight + cols_offsets)
    w = tl.expand_dims(w, 0)
    w = tl.broadcast_to(w, (BLOCK, N_COLS))
    NUM_BLOCKS = tl.cdiv(n_rows, BLOCK)
    for row_block_id in range(pid, NUM_BLOCKS, NUM_CORES):
        pos_offset = row_block_id * BLOCK + tl.arange(0, BLOCK)
        pos_mask = (pos_offset < n_rows)[:, None]
        base_offset = pos_offset[:, None] * input_row_stride + cols_offsets[None, :]
        x = tl.load(input + base_offset, mask=pos_mask)
        xf = x.to(tl.float32)
        var = tl.sum(xf * xf, 1) / N_COLS
        qrt = tl.expand_dims(tl.math.rsqrt(var + eps), 1)
        out = xf * tl.broadcast_to(qrt, (BLOCK, N_COLS))
        out = w * out.to(x.dtype)
        tl.store(output + base_offset, out, mask=pos_mask)


def rms_norm_triton(
    hidden_states: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
):
    assert hidden_states.is_contiguous()
    feat_size = weight.shape[0]
    seq_len = hidden_states.numel() // hidden_states.size(-1)
    input_stride = hidden_states.stride(-2)
    out = torch.empty_like(hidden_states)
    rms_norm_kernel[(NUM_CORES,)](
        hidden_states,
        weight,
        out,
        n_rows=seq_len,
        input_row_stride=input_stride,
        eps=eps,
        N_COLS=feat_size,
        NUM_CORES=NUM_CORES,
    )
    return out


def rms_norm_block_triton(
    hidden_states: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
):
    assert hidden_states.is_contiguous()
    feat_size = weight.shape[0]
    seq_len = hidden_states.numel() // hidden_states.size(-1)
    input_stride = hidden_states.stride(-2)
    out = torch.empty_like(hidden_states)
    rms_norm_block_kernel[(NUM_CORES,)](
        hidden_states,
        weight,
        out,
        n_rows=seq_len,
        input_row_stride=input_stride,
        eps=eps,
        N_COLS=feat_size,
        BLOCK=16,
        NUM_CORES=NUM_CORES,
    )
    return out
