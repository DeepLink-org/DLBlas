# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _max_reduction_kernel(
    x_ptr,
    output_ptr,
    D0, D1, D2,
    stride_x0, stride_x1, stride_x2,
    stride_out0, stride_out1,
    d: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    
    if d == 0:
        base_x = pid0 * stride_x1 + pid1 * stride_x2
        step_x = stride_x0
        D_reduction = D0
    elif d == 1:
        base_x = pid0 * stride_x0 + pid1 * stride_x2
        step_x = stride_x1
        D_reduction = D1
    else:  # d == 2
        base_x = pid0 * stride_x0 + pid1 * stride_x1
        step_x = stride_x2
        D_reduction = D2

    r_offsets = tl.arange(0, BLOCK_R)
    x_ptrs = x_ptr + base_x + r_offsets * step_x
    mask = r_offsets < D_reduction
    
    data = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    max_val = tl.max(data, axis=0)
    
    output_index = pid0 * stride_out0 + pid1 * stride_out1
    tl.store(output_ptr + output_index, max_val)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            return torch.max(x, dim=self.dim)[0]
            
        d = self.dim
        D0, D1, D2 = x.shape
        
        if d == 0:
            output_shape = (D1, D2)
        elif d == 1:
            output_shape = (D0, D2)
        elif d == 2:
            output_shape = (D0, D1)
        else:
            raise ValueError("dim must be 0,1,2")
        
        output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
        reduction_dim_size = x.shape[d]
        
        if reduction_dim_size == 0:
            return output
            
        if reduction_dim_size <= 1024 and x.is_cuda:
            stride_x0 = x.stride(0)
            stride_x1 = x.stride(1)
            stride_x2 = x.stride(2)
            
            if output.dim() == 0:
                stride_out0, stride_out1 = 0, 0
            elif output.dim() == 1:
                stride_out0, stride_out1 = output.stride(0), 0
            else:
                stride_out0, stride_out1 = output.stride(0), output.stride(1)
            
            BLOCK_R = triton.next_power_of_2(reduction_dim_size)
            grid = (output_shape[0], output_shape[1])
            _max_reduction_kernel[grid](
                x, output,
                D0, D1, D2,
                stride_x0, stride_x1, stride_x2,
                stride_out0, stride_out1,
                d, BLOCK_R=BLOCK_R
            )
        else:
            output = torch.max(x, dim=d)[0]
            
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