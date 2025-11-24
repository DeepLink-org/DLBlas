# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_pointwise_kernel(
    conv_ptr,
    sum_ptr,
    output_ptr,
    n_elements,
    volume,
    channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load convolution output
    x = tl.load(conv_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate channel indices
    channel_idx = (offsets // volume) % channels
    
    # Load sum tensor values
    s = tl.load(sum_ptr + channel_idx, mask=mask, other=0.0)
    
    # Apply LeakyReLU
    x = tl.where(x >= 0, x, x * 0.2)
    
    # Add sum tensor
    x = x + s
    
    # Clamp values
    x = tl.minimum(tl.maximum(x, -1.0), 1.0)
    
    # Apply GELU using erf approximation
    x = x * 0.5 * (1.0 + tl.erf(x * 0.7071067811865475))
    
    # Store results
    tl.store(output_ptr + offsets, x, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
    
    def forward(self, x):
        conv_output = self.conv(x)
        conv_output = conv_output.contiguous()
        output = torch.empty_like(conv_output)
        n_elements = conv_output.numel()
        volume = conv_output.shape[2] * conv_output.shape[3] * conv_output.shape[4]
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        fused_pointwise_kernel[grid](
            conv_output, 
            self.sum_tensor.view(-1), 
            output, 
            n_elements, 
            volume, 
            conv_output.shape[1]
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