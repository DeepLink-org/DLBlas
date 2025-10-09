# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _reduce_sum_kernel(
    x_ptr,
    output_ptr,
    stride_b, stride_i, stride_j,
    output_stride_b, output_stride_j,
    dim1,
    BLOCK_SIZE_R: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    total = 0.0
    for i in range(0, dim1, BLOCK_SIZE_R):
        r_offsets = i + tl.arange(0, BLOCK_SIZE_R)
        mask = r_offsets < dim1
        base = pid_b * stride_b + r_offsets * stride_i + pid_j * stride_j
        data = tl.load(x_ptr + base, mask=mask, other=0.0)
        total += tl.sum(data)
    
    output_offset = pid_b * output_stride_b + pid_j * output_stride_j
    tl.store(output_ptr + output_offset, total)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim != 1:
            return torch.sum(x, dim=self.dim, keepdim=True)
            
        batch_size, dim1, dim2 = x.shape
        stride_b, stride_i, stride_j = x.stride()
        output = torch.empty((batch_size, dim2), device=x.device, dtype=x.dtype)
        output_stride_b = output.stride(0)
        output_stride_j = output.stride(1)
        
        BLOCK_SIZE_R = 256
        grid = (batch_size, dim2)
        _reduce_sum_kernel[grid](
            x, output,
            stride_b, stride_i, stride_j,
            output_stride_b, output_stride_j,
            dim1,
            BLOCK_SIZE_R=BLOCK_SIZE_R
        )
        return output.unsqueeze(1)

batch_size = 16
dim1 = 256
dim2 = 256
reduce_dim = 1

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [reduce_dim]
# =================== EVOLVE-BLOCK-END ===================