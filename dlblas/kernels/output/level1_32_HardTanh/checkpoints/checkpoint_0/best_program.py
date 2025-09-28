# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hardtanh_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Apply HardTanh: clamp between -1 and 1
    clamped = tl.minimum(tl.maximum(x, -1.0), 1.0)
    
    # Store result
    tl.store(output_ptr + offsets, clamped, mask=mask)

class ModelNew(nn.Module):
    """
    Optimized model that performs HardTanh activation using Triton.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input for efficient kernel processing
        shape = x.shape
        x_flat = x.contiguous().view(-1)
        n_elements = x_flat.numel()
        
        # Skip processing if empty tensor
        if n_elements == 0:
            return torch.empty_like(x)
        
        # Allocate output
        output = torch.empty_like(x_flat)
        
        # Configure kernel launch
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
        
        # Launch kernel
        hardtanh_kernel[grid](x_flat, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        
        return output.view(shape)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
# =================== EVOLVE-BLOCK-END ===================