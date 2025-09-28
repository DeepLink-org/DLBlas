# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x):
        n_elements = x.numel()
        output = torch.empty_like(x)
        
        if n_elements == 0:
            return output
            
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        self._gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
        return output

    @staticmethod
    @triton.jit
    def _gelu_kernel(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        
        # Constants for GELU approximation
        sqrt_2_over_pi = 0.7978845608028654
        k = 0.044715
        
        # Compute GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + k * x^3)))
        x_cubed = x * x * x
        inner = x + k * x_cubed
        tanh_in = sqrt_2_over_pi * inner
        tanh_val = tl.math.tanh(tanh_in)
        gelu_out = 0.5 * x * (1.0 + tanh_val)
        
        tl.store(output_ptr + offsets, gelu_out, mask=mask)

batch_size = 2000
dim = 2000

def get_inputs():
    return [torch.randn(batch_size, dim)]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================