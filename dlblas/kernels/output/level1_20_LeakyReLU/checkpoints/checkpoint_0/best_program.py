# ================== EVOLVE-BLOCK-START ==================
import torch
import triton
import triton.language as tl

class ModelNew(torch.nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
    
    @staticmethod
    @triton.jit
    def leaky_relu_kernel(
        x_ptr,
        output_ptr,
        negative_slope,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask)
        zero = tl.zeros_like(x)
        result = tl.where(x >= 0, x, x * negative_slope)
        tl.store(output_ptr + offsets, result, mask=mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_elements = x.numel()
        output = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        self.leaky_relu_kernel[grid](
            x,
            output,
            self.negative_slope,
            n_elements,
            BLOCK_SIZE=1024
        )
        return output

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================