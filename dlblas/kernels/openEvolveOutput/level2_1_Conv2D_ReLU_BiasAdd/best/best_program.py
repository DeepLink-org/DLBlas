# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, x):
        conv_output = self.conv(x)
        output = torch.empty_like(conv_output)
        B, C, H, W = conv_output.shape
        
        # Fused ReLU + bias kernel
        @triton.jit
        def _relu_bias_kernel(
            output_ptr, conv_ptr, bias_ptr,
            n_elements, C, H, W,
            BLOCK_SIZE: tl.constexpr
        ):
            pid = tl.program_id(0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            
            # Calculate channel index
            c_idx = (offsets // (H * W)) % C
            conv_vals = tl.load(conv_ptr + offsets, mask=mask)
            bias_vals = tl.load(bias_ptr + c_idx, mask=mask)
            
            # Fused ReLU + bias
            result = tl.where(conv_vals >= 0, conv_vals, 0.0) + bias_vals
            tl.store(output_ptr + offsets, result, mask=mask)
        
        n_elements = B * C * H * W
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _relu_bias_kernel[grid](
            output, conv_output, self.bias.view(-1),
            n_elements, C, H, W,
            BLOCK_SIZE=1024
        )
        return output

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
# =================== EVOLVE-BLOCK-END ===================