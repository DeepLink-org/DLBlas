# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_activation_kernel(
    x_ptr,
    output_ptr,
    add_value,
    scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute softplus with thresholding for numerical stability
    threshold = 20.0
    exp_x = tl.exp(x)
    softplus = tl.where(x > threshold, x, tl.log(1 + exp_x))
    
    # Compute tanh(softplus) using exponential identity
    two_t = 2.0 * softplus
    exp_2t = tl.exp(two_t)
    tanh_softplus = (exp_2t - 1.0) / (exp_2t + 1.0)
    
    mish = x * tanh_softplus
    
    # Add value and clamp
    y = mish + add_value
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    
    # Scale output
    y = y * scale
    
    tl.store(output_ptr + offsets, y, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        x = self.conv_transpose(x)
        n_elements = x.numel()
        output = torch.empty_like(x)
        
        if n_elements > 0:
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            fused_activation_kernel[grid](
                x, output, self.add_value, self.scale, n_elements
            )
        return output

batch_size = 128
in_channels = 32
out_channels = 64
height, width = 16, 16
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
add_value = 0.5
scale = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]
# =================== EVOLVE-BLOCK-END ===================