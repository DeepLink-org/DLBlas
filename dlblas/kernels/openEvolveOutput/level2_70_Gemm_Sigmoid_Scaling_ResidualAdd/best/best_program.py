# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_sigmoid_residual(
    x_ptr,
    output_ptr,
    scaling_factor,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    result = sigmoid_x * scaling_factor + x
    tl.store(output_ptr + offsets, result, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.gemm(x)
        x_contig = x.contiguous()
        output = torch.empty_like(x_contig)
        n_elements = x_contig.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        fused_sigmoid_residual[grid](
            x_contig, 
            output, 
            self.scaling_factor, 
            n_elements,
            BLOCK_SIZE=1024
        )
        return output

batch_size = 128
input_size = 1024
hidden_size = 512
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
# =================== EVOLVE-BLOCK-END ===================