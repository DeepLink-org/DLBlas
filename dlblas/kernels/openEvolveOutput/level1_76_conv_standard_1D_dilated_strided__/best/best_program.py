import math
import torch
import triton
import triton.language as tl

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
    key=["P2", "BLOCK_SIZE"],
)

@triton.jit
def _quantize_rowwise(
    x_ptr,
    output_ptr,
    output_scale_ptr,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, P2)
    row_mask = tl.arange(0, P2) < BLOCK_SIZE
    x = tl.load(x_ptr + offsets, mask=row_mask, other=0.0)

    abs_x = tl.abs(x)
    max_val = tl.max(abs_x, axis=0)
    max_val = tl.maximum(max_val, 1e-4)
    scale = 127.0 / max_val
    scaled = x * scale
    clamped = tl.minimum(tl.maximum(scaled, -127.0), 127.0)
    quantized = tl.llrint(clamped).to(tl.int8)
    
    tl.store(output_ptr + offsets, quantized, mask=row_mask)
    tl.store(output_scale_ptr + pid, max_val / 127.0)

def per_row_quantize_int8(x: torch.Tensor):
    x = x.contiguous()
    output = torch.empty(*x.shape, device=x.device, dtype=torch.int8)
    output_scale = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    P2 = int(2 ** (math.ceil(math.log2(x.shape[1]))))
    grid = lambda x: (x.shape[0],)
    _quantize_rowwise[grid](x, output, output_scale, BLOCK_SIZE=x.shape[1], P2=P2)
    return output, output_scale