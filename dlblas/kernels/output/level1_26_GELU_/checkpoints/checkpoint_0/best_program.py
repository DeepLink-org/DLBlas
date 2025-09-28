# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel for GELU activation
@triton.jit
def gelu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x*x*x)))
    a = 0.044715
    b = 0.7978845608028654  # sqrt(2/pi)
    c = 1.0
    x_cubed = x * x * x
    inner = b * (x + a * x_cubed)
    tanh_inner = tl.tanh(inner)
    gelu = 0.5 * x * (c + tanh_inner)
    
    tl.store(output_ptr + offsets, gelu, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process flattened tensor with Triton kernel
        shape = x.shape
        x_flat = x.flatten().contiguous()
        n_elements = x_flat.numel()
        
        # Skip computation for empty tensors
        if n_elements == 0:
            return torch.empty_like(x)
        
        output = torch.empty_like(x_flat)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        gelu_kernel[grid](x_flat, output, n_elements, BLOCK_SIZE=1024)
        
        return output.view(shape)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
# =================== EVOLVE-BLOCK-END ===================