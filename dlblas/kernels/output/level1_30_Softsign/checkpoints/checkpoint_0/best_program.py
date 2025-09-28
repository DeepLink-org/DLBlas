# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=1),
    ],
    key=['n_elements']
)
@triton.jit
def softsign_kernel(
    output_ptr,
    input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input elements
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute softsign: x / (1 + |x|)
    abs_x = tl.abs(x)
    one = tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32)
    denominator = one + abs_x
    output = x / denominator
    
    # Store results
    tl.store(output_ptr + offsets, output, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_elements = x.numel()
        output = torch.empty_like(x)
        
        # Use 1D grid where each block processes BLOCK_SIZE elements
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        softsign_kernel[grid](
            output_ptr=output,
            input_ptr=x,
            n_elements=n_elements,
            BLOCK_SIZE=1024,  # Initial value, autotune will override
        )
        return output

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================