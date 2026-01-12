# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _softplus_kernel(
    x_ptr,
    output_ptr,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute softplus with threshold optimization
    threshold = 20.0
    result = tl.where(
        x > threshold, 
        x, 
        tl.log(1 + tl.exp(tl.minimum(x, threshold)))
    )
    
    tl.store(output_ptr + offsets, result, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input for efficient kernel processing
        x_flat = x.contiguous().view(-1)
        total_elements = x_flat.numel()
        output_flat = torch.empty_like(x_flat)
        
        # Configure kernel launch parameters
        grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
        _softplus_kernel[grid](
            x_flat, 
            output_flat,
            total_elements,
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