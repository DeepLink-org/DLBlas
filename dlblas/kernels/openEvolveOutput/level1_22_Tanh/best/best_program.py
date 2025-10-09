# ================== EVOLVE-BLOCK-START ==================
import torch
import triton
import triton.language as tl

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        self.tanh_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
        return output

    @staticmethod
    @triton.jit
    def tanh_kernel(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load and compute tanh using exponential formula
        x = tl.load(x_ptr + offsets, mask=mask)
        exp2x = tl.exp(2 * x)
        tanh_val = (exp2x - 1) / (exp2x + 1)
        tl.store(output_ptr + offsets, tanh_val, mask=mask)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
# =================== EVOLVE-BLOCK-END ===================