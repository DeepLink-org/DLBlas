# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _min_reduce_kernel(
    x_ptr,
    output_ptr,
    M, N, K,
    stride_m, stride_n, stride_k,
    output_stride_0, output_stride_1,
    reduce_size: tl.constexpr,
    REDUCE_DIM: tl.constexpr,
    BLOCK_R: tl.constexpr
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    
    r = tl.arange(0, BLOCK_R)
    mask = r < reduce_size
    
    if REDUCE_DIM == 0:
        base = pid0 * stride_n + pid1 * stride_k
        offsets = base + r * stride_m
    elif REDUCE_DIM == 1:
        base = pid0 * stride_m + pid1 * stride_k
        offsets = base + r * stride_n
    else:
        base = pid0 * stride_m + pid1 * stride_n
        offsets = base + r * stride_k
        
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=float('inf'))
    min_val = tl.min(x_vals, axis=0)
    
    out_offset = pid0 * output_stride_0 + pid1 * output_stride_1
    tl.store(output_ptr + out_offset, min_val)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim
        if dim < 0:
            dim = dim + x.dim()
        assert x.dim() == 3, "Input must be 3D"
        assert dim in [0,1,2], "dim must be 0,1,2"
        
        shape = list(x.shape)
        del shape[dim]
        output = torch.empty(shape, device=x.device, dtype=x.dtype)
        
        M, N, K = x.shape
        reduce_size = x.shape[dim]
        
        if dim == 0:
            grid = (N, K)
        elif dim == 1:
            grid = (M, K)
        else:
            grid = (M, N)
            
        stride_m, stride_n, stride_k = x.stride()
        
        if output.dim() == 2:
            output_stride_0 = output.stride(0)
            output_stride_1 = output.stride(1)
        else:
            output_stride_0 = output.stride(0)
            output_stride_1 = 0
            
        num_warps = min(32, (reduce_size + 31) // 32)
        _min_reduce_kernel[grid](
            x, output,
            M, N, K,
            stride_m, stride_n, stride_k,
            output_stride_0, output_stride_1,
            reduce_size,
            dim,
            BLOCK_R=reduce_size,
            num_warps=num_warps
        )
        
        return output

batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1]
# =================== EVOLVE-BLOCK-END ===================