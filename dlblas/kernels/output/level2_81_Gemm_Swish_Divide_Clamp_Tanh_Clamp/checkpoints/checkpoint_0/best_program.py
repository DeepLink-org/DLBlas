# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

# Fused element-wise operations kernel
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _fused_ops_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input element
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Swish: x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    x = x * sigmoid_x
    
    # Divide by 2
    x = x / 2.0
    
    # Clamp between [-1, 1]
    x = tl.minimum(tl.maximum(x, -1.0), 1.0)
    
    # Tanh activation
    x = tl.tanh(x)
    
    # Final clamp between [-1, 1]
    x = tl.minimum(tl.maximum(x, -1.0), 1.0)
    
    # Store result
    tl.store(output_ptr + offsets, x, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        # Compute GEMM
        x = self.gemm(x)
        
        # Fused element-wise operations
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _fused_ops_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
        
        return output

batch_size = 128
in_features = 1024
out_features = 512

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]
# =================== EVOLVE-BLOCK-END ===================