# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

# Fused pointwise operations kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _fused_ops(
    x_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    n_elements_per_batch,
    n_elements_per_channel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # ReLU
    x = tl.where(x >= 0, x, 0.0)
    
    # GELU approximation (faster than exact)
    gelu = x * 0.5 * (1.0 + tl.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    
    # Sigmoid
    sigmoid = 1.0 / (1.0 + tl.exp(-gelu))
    
    # Compute channel indices
    offset_in_batch = offsets % n_elements_per_batch
    channel_idx = offset_in_batch // n_elements_per_channel
    
    # Load bias and add
    bias = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)
    output = sigmoid + bias
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous()
        output = torch.empty_like(x)
        
        n_elements = x.numel()
        if n_elements > 0:
            batch_size, out_channels, d, h, w = x.shape
            n_elements_per_batch = out_channels * d * h * w
            n_elements_per_channel = d * h * w
            
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            _fused_ops[grid](
                x, 
                self.bias, 
                output,
                n_elements,
                n_elements_per_batch,
                n_elements_per_channel
            )
        return output

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
# =================== EVOLVE-BLOCK-END ===================