# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def reduce_prod_kernel(
    x_ptr,
    output_ptr,
    stride_x0, stride_x1, stride_x2,
    stride_out0, stride_out1,
    reduction_size,
    BLOCK_SIZE: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // stride_out1
    other_idx = pid % stride_out1

    base_ptr = x_ptr + batch_idx * stride_x0 + other_idx
    offsets = tl.arange(0, BLOCK_SIZE) * stride_x2
    mask = offsets < reduction_size * stride_x2

    accum = 1.0
    for i in range(0, reduction_size, VEC_SIZE):
        vec_offsets = base_ptr + (i + tl.arange(0, VEC_SIZE)) * stride_x1
        vec_mask = mask & ((i + tl.arange(0, VEC_SIZE)) < reduction_size)
        vec = tl.load(vec_offsets, mask=vec_mask, other=1.0)
        
        # Tree reduction within the vector
        for s in range(VEC_SIZE // 2, 0, -1):
            part1 = tl.where(tl.arange(0, s) < s, vec[:s], 1.0)
            part2 = tl.where(tl.arange(0, s) < s, vec[s:2*s], 1.0)
            vec = part1 * part2
        accum = accum * vec[0]

    output_offset = batch_idx * stride_out0 + other_idx * stride_out1
    tl.store(output_ptr + output_offset, accum)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        if self.dim == 1:
            output_shape = (x.shape[0], x.shape[2])
        elif self.dim == 2:
            output_shape = (x.shape[0], x.shape[1])
        else:
            return torch.prod(x, dim=self.dim)

        output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
        
        if self.dim == 1:
            stride_x1 = x.stride(1)
            stride_x2 = x.stride(2)
            reduction_size = x.shape[1]
        else:  # dim == 2
            stride_x1 = x.stride(2)
            stride_x2 = x.stride(1)
            reduction_size = x.shape[2]
            
        total_elements = output_shape[0] * output_shape[1]
        grid = (total_elements,)
        
        BLOCK_SIZE = 256
        VEC_SIZE = 4
        
        reduce_prod_kernel[grid](
            x, output,
            x.stride(0), stride_x1, stride_x2,
            output.stride(0), output.stride(1),
            reduction_size,
            BLOCK_SIZE=BLOCK_SIZE,
            VEC_SIZE=VEC_SIZE
        )
        return output

batch_size = 16
dim1 = 256
dim2 = 256
reduction_dim = 1

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [reduction_dim]
# =================== EVOLVE-BLOCK-END ===================