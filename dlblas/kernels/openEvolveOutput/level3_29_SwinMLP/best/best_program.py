import math
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=8, num_warps=8),
        triton.Config({}, num_stages=1),
        triton.Config({}, num_stages=2),
        triton.Config({}, num_stages=4),
        triton.Config({}, num_stages=8),
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BLOCK_SIZE"],
)

@triton.jit
def _quantize_rowwise(
    x_ptr,
    output_ptr,
    output_scale,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    row_mask = arange < BLOCK_SIZE
    x = tl.load(x_ptr + offsets, mask=row_mask, other=0.0)

    abs_x = tl.abs(x)
    max_val = tl.max(tl.where(row_mask, abs_x, 0), axis=0)
    max_val = tl.maximum(max_val, 1e-4)
    scale = 127.0 / max_val
    scaled = x * scale
    clipped = tl.minimum(tl.maximum(scaled, -127.0), 127.0)
    integer_value = libdevice.llrint(clipped)
    tl.store(output_ptr + offsets, integer_value.to(tl.int8), mask=row_mask)
    tl.store(output_scale + pid, max_val / 127.0)

def per_row_quantize_int8(x: torch.Tensor):
    x = x.contiguous()
    output = torch.empty(*x.shape, device=x.device, dtype=torch.int8)
    output_scale = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    row_length = x.shape[1]
    P2_val = 1 << (row_length - 1).bit_length()
    grid = (x.shape[0],)
    _quantize_rowwise[grid](x, output, output_scale, BLOCK_SIZE=row_length, P2=P2_val)
    return output, output_scale