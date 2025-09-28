# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _mul_scalar_kernel(
    A_ptr,
    C_ptr,
    s,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(A_ptr + offsets, mask=mask)
    c = a * s
    tl.store(C_ptr + offsets, c, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        # Flatten to 1D for efficient kernel processing
        A_flat = A.reshape(-1)
        n_elements = A_flat.numel()
        output = torch.empty_like(A_flat)
        
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _mul_scalar_kernel[grid](A_flat, output, s, n_elements)
        
        return output.reshape(A.shape)

M = 16384
N = 4096

def get_inputs():
    A = torch.randn(M, N, device='cuda')
    s = 3.14
    return [A, s]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================