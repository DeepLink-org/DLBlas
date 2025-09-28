# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fused_pointwise_kernel(
    x_ptr,
    s_ptr,
    output_ptr,
    out_channels,
    block_size,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Load input values
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate channel indices
    channel_idx = (offsets // block_size) % out_channels
    s_vals = tl.load(s_ptr + channel_idx, mask=mask, other=0.0)
    
    # LeakyReLU (slope=0.2)
    leaky = tl.where(x_vals >= 0, x_vals, 0.2 * x_vals)
    
    # Add channel-specific tensor
    added = leaky + s_vals
    
    # Clamp between -1 and 1
    clamped = tl.minimum(tl.maximum(added, -1.0), 1.0)
    
    # Fast GELU approximation
    a = clamped * 0.7978845608028654  # sqrt(2/π)
    b = 0.044715 * (clamped ** 3)
    inner = a * (1 + b)
    tanh_inner = tl.tanh(inner)
    gelu = 0.5 * clamped * (1 + tanh_inner)
    
    tl.store(output_ptr + offsets, gelu, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        
    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous()
        output = torch.empty_like(x)
        
        # Calculate dimensions for kernel
        _, out_channels, depth, height, width = x.shape
        block_size = depth * height * width
        num_elements = x.numel()
        
        # Launch kernel
        grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
        _fused_pointwise_kernel[grid](
            x, 
            self.sum_tensor.view(-1), 
            output,
            out_channels,
            block_size,
            num_elements,
            BLOCK_SIZE=1024
        )
        return output

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
sum_tensor_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, sum_tensor_shape]
# =================== EVOLVE-BLOCK-END ===================