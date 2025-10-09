# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def elu_kernel(
    x_ptr,
    output_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    zero = 0.0
    positive_part = tl.where(x > zero, x, zero)
    negative_part = tl.where(x <= zero, alpha * (tl.exp(x) - 1.0), zero)
    output = positive_part + negative_part
    
    tl.store(output_ptr + offsets, output, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input for 1D kernel processing
        shape = x.shape
        x_flat = x.flatten().contiguous()
        output_flat = torch.empty_like(x_flat)
        n_elements = x_flat.numel()
        
        # Optimized kernel configuration
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        elu_kernel[grid](x_flat, output_flat, self.alpha, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        
        return output_flat.view(shape)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return [1.0]  # Provide alpha value for initialization
# =================== EVOLVE-BLOCK-END ===================