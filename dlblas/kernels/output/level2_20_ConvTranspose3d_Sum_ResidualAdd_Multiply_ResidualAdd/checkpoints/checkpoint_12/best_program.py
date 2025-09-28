# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fused_operations_kernel(
    x_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    C,
    D,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the original_x values
    x_vals = tl.load(x_ptr + offsets, mask=mask)

    # Compute channel indices
    c_index = (offsets // (D * H * W)) % C

    # Load bias values using computed channel indices
    bias_vals = tl.load(bias_ptr + c_index, mask=mask)

    # Fused computation: 2*x^2 + (bias+1)*x
    result = 2 * (x_vals * x_vals) + (bias_vals + 1) * x_vals

    # Store results
    tl.store(output_ptr + offsets, result, mask=mask)

class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by fused operations
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x_conv = self.conv_transpose(x)
        B, C, D, H, W = x_conv.shape
        n_elements = B * C * D * H * W
        
        # Prepare output tensor
        output = torch.empty_like(x_conv)
        
        # Reshape bias to 1D tensor
        bias_1d = self.bias.view(C)
        
        # Launch kernel with optimal block size
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _fused_operations_kernel[grid](
            x_conv, bias_1d, output, n_elements, C, D, H, W, BLOCK_SIZE=1024
        )
        
        return output

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]
# =================== EVOLVE-BLOCK-END ===================