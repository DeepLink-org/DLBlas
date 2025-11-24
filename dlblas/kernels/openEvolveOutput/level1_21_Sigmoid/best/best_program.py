# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous memory layout
        x_contig = x.contiguous()
        output = torch.empty_like(x_contig)
        n_elements = x_contig.numel()
        
        # Handle empty tensor case
        if n_elements == 0:
            return output
        
        # Launch optimized kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        self.sigmoid_kernel[grid](output, x_contig, n_elements)
        return output

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 8192}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 16384}, num_warps=4),
        ],
        key=['n_elements'],
    )
    @triton.jit
    def sigmoid_kernel(
        output_ptr,
        input_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load data
        x = tl.load(input_ptr + offsets, mask=mask)
        # Compute sigmoid using optimized hardware instruction
        y = tl.sigmoid(x)
        # Store result
        tl.store(output_ptr + offsets, y, mask=mask)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
# =================== EVOLVE-BLOCK-END ===================