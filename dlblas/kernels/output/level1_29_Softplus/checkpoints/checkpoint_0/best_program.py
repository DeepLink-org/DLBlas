# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softplus_kernel(
    output_ptr,
    input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    abs_x = tl.abs(x)
    # Numerically stable softplus implementation
    result = tl.where(
        x > 0,
        x + tl.log(1 + tl.exp(-abs_x)),
        tl.log(1 + tl.exp(abs_x))
    )
    tl.store(output_ptr + offsets, result, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(-1)
        output_flat = torch.empty_like(x_flat)
        n_elements = x_flat.numel()
        
        if n_elements > 0:
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            softplus_kernel[grid](
                output_flat, 
                x_flat, 
                n_elements,
                BLOCK_SIZE=1024
            )
        
        return output_flat.view_as(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================