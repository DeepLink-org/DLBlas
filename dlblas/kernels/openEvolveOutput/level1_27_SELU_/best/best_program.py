# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _selu_kernel(
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
    
    # SELU parameters
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    
    # Compute SELU
    zero = tl.zeros(x.shape, x.dtype)
    pos_mask = x > 0
    neg_mask = ~pos_mask
    
    # Positive part: scale * x
    pos_part = tl.where(pos_mask, x, zero) * scale
    
    # Negative part: scale * alpha * (exp(x) - 1)
    exp_x = tl.exp(tl.where(neg_mask, x, zero))
    neg_part = tl.where(neg_mask, (exp_x - 1.0) * alpha * scale, zero)
    
    # Combine results
    result = pos_part + neg_part
    tl.store(output_ptr + offsets, result, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input for efficient processing
        original_shape = x.shape
        x_flat = x.flatten().contiguous()
        output = torch.empty_like(x_flat)
        
        # Kernel parameters
        n_elements = x_flat.numel()
        BLOCK_SIZE = 1024  # Optimized for H100 architecture
        
        # Launch kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _selu_kernel[grid](x_flat, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        
        return output.view(original_shape)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
# =================== EVOLVE-BLOCK-END ===================