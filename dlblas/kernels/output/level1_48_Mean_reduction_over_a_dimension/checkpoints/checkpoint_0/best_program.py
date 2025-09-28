# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mean_kernel(
    x_ptr,
    output_ptr,
    reduce_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = 0.0
    count = 0
    for i in range(0, reduce_size, BLOCK_SIZE):
        off = i + tl.arange(0, BLOCK_SIZE)
        mask = off < reduce_size
        vec = tl.load(x_ptr + pid * reduce_size + off, mask=mask, other=0.0)
        total += tl.sum(vec, axis=0)
        count += tl.sum(mask.to(tl.int32), axis=0)
    mean_val = total / tl.cast(count, tl.float32)
    tl.store(output_ptr + pid, mean_val)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        dim = self.dim
        if dim < 0:
            dim = x.dim() + dim
            
        reduction_size = original_shape[dim]
        non_reduction_dims = list(original_shape)
        del non_reduction_dims[dim]
        
        if dim == x.dim() - 1:
            x_flat = x.contiguous().view(-1, reduction_size)
        else:
            perm = list(range(x.dim()))
            perm[dim] = x.dim()-1
            perm[x.dim()-1] = dim
            x_flat = x.contiguous().permute(perm).view(-1, reduction_size)
            
        total_non_reduction = x_flat.shape[0]
        output_flat = torch.empty(total_non_reduction, device=x.device, dtype=x.dtype)
        
        grid = (total_non_reduction,)
        mean_kernel[grid](x_flat, output_flat, reduction_size, BLOCK_SIZE=1024)
        
        output = output_flat.reshape(non_reduction_dims)
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