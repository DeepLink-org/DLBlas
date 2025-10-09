# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hardsigmoid_kernel(
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
    # HardSigmoid computation: clamp(x/6 + 0.5, 0, 1)
    y = x * 0.16666666666666666 + 0.5  # 1/6 ≈ 0.16666666666666666
    y = tl.minimum(tl.maximum(y, 0.0), 1.0)
    tl.store(output_ptr + offsets, y, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input to 1D tensor for efficient processing
        shape = x.shape
        x_flat = x.contiguous().view(-1)
        output_flat = torch.empty_like(x_flat)
        
        n_elements = x_flat.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        # Launch kernel with optimized block size
        hardsigmoid_kernel[grid](
            x_flat,
            output_flat,
            n_elements,
            BLOCK_SIZE=1024,
        )
        return output_flat.view(shape)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================