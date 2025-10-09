# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_add_hardswish_kernel(
    conv_out_ptr, add_input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load convolution output and add_input
    conv_out = tl.load(conv_out_ptr + offsets, mask=mask)
    add_val = tl.load(add_input_ptr + offsets, mask=mask)
    
    # Fused operations: add + HardSwish multiplication
    z = conv_out + add_val
    shifted = z + 3.0
    relu6 = tl.minimum(tl.maximum(shifted, 0.0), 6.0)
    hardswish_val = z * relu6 / 6.0
    output_val = z * hardswish_val
    
    tl.store(output_ptr + offsets, output_val, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, add_input):
        # Compute transposed convolution
        conv_out = self.conv_transpose(x)
        
        # Prepare for fused kernel
        output = torch.empty_like(conv_out)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        # Launch fused kernel
        fused_add_hardswish_kernel[grid](
            conv_out, add_input, output,
            n_elements,
            BLOCK_SIZE=1024
        )
        return output

batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W), torch.randn(batch_size, out_channels, D*stride, H*stride, W*stride)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]
# =================== EVOLVE-BLOCK-END ===================