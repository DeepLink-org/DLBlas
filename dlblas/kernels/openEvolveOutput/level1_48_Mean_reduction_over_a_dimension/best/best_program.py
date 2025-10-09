# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _mean_reduce_kernel(
    x_ptr,
    output_ptr,
    reduction_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * reduction_size
    total = 0.0
    for offset in range(0, reduction_size, BLOCK_SIZE):
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < reduction_size
        vec = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        total += tl.sum(vec, axis=0)
    mean_val = total / reduction_size
    tl.store(output_ptr + pid, mean_val)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.dim if self.dim >= 0 else x.dim() + self.dim
        original_shape = x.shape
        reduction_size = original_shape[d]
        
        other_dims = list(original_shape)
        del other_dims[d]
        num_output_elements = torch.Size(other_dims).numel()
        
        if reduction_size == 0:
            return torch.zeros(other_dims, device=x.device, dtype=x.dtype)
        
        # Permute reduction dimension to innermost position
        if d == x.dim() - 1:
            x_flat = x.contiguous().view(-1, reduction_size)
        else:
            perm = list(range(x.dim()))
            perm.pop(d)
            perm.append(d)
            x_flat = x.permute(perm).contiguous().view(-1, reduction_size)
        
        output_flat = torch.empty(x_flat.shape[0], device=x.device, dtype=x.dtype)
        
        BLOCK_SIZE = min(triton.next_power_of_2(reduction_size), 1024)
        grid = (x_flat.shape[0],)
        _mean_reduce_kernel[grid](
            x_flat, output_flat, reduction_size, BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output_flat.view(other_dims)

batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1]
# =================== EVOLVE-BLOCK-END ===================