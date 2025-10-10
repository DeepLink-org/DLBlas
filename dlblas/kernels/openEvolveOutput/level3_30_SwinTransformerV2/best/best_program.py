import math
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32),
    ],
    key=['n_cols'],
)
@triton.jit
def _quantize_rowwise(
    x_ptr,
    output_ptr,
    output_scale,
    n_cols,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * stride_row
    max_val = tl.zeros((1,), tl.float32)
    
    # Compute max absolute value in chunks
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        chunk = tl.load(x_ptr + row_start + col_offsets, mask, 0.0)
        abs_chunk = tl.abs(chunk)
        chunk_max = tl.max(tl.where(mask, abs_chunk, 0.0), 0)
        max_val = tl.maximum(max_val, chunk_max)
    
    max_val = tl.maximum(max_val, 1e-4)
    scale = 127.0 / max_val
    
    # Quantize in chunks
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        chunk = tl.load(x_ptr + row_start + col_offsets, mask, 0.0)
        quantized = libdevice.llrint(chunk * scale)
        quantized = tl.minimum(tl.maximum(quantized, -127), 127)
        tl.store(output_ptr + row_start + col_offsets, quantized.to(tl.int8), mask)
    
    tl.store(output_scale + pid, max_val / 127.0)

def per_row_quantize_int8(x: torch.Tensor):
    x = x.contiguous()
    n_rows, n_cols = x.shape
    output = torch.empty_like(x, dtype=torch.int8)
    output_scale = torch.empty(n_rows, device=x.device, dtype=torch.float32)
    
    if n_cols == 0:
        return output, output_scale
        
    grid = (n_rows,)
    stride_row = x.stride(0)
    _quantize_rowwise[grid](x, output, output_scale, n_cols, stride_row)
    
    return output, output_scale