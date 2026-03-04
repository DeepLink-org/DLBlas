# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def argmax_kernel(
    x_ptr,
    output_ptr,
    d0,
    d1,
    d2,
    stride0,
    stride1,
    stride2,
    reduced_dim: tl.constexpr,
):
    pid = tl.program_id(0)
    if reduced_dim == 0:
        idx1 = pid // d2
        idx2 = pid % d2
        base = idx1 * stride1 + idx2 * stride2
        step = stride0
        n = d0
    elif reduced_dim == 1:
        idx0 = pid // d2
        idx2 = pid % d2
        base = idx0 * stride0 + idx2 * stride2
        step = stride1
        n = d1
    else:
        idx0 = pid // d1
        idx1 = pid % d1
        base = idx0 * stride0 + idx1 * stride1
        step = stride2
        n = d2

    max_val = tl.load(x_ptr + base)
    max_index = 0
    for i in range(1, n):
        offset = base + i * step
        val = tl.load(x_ptr + offset)
        if val > max_val:
            max_val = val
            max_index = i

    tl.store(output_ptr + pid, max_index)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or self.dim not in [0, 1, 2]:
            return torch.argmax(x, dim=self.dim)
            
        d0, d1, d2 = x.shape
        strides = x.stride()
        
        if self.dim == 0:
            output_shape = (d1, d2)
        elif self.dim == 1:
            output_shape = (d0, d2)
        else:
            output_shape = (d0, d1)
            
        output = torch.empty(output_shape, device=x.device, dtype=torch.int64)
        grid = (output.numel(),)
        
        argmax_kernel[grid](
            x, output, d0, d1, d2, 
            strides[0], strides[1], strides[2], 
            reduced_dim=self.dim
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