# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_operations_kernel(
    input_ptr,
    bias_ptr,
    output_ptr,
    H, W,
    stride_b, stride_h, stride_w,
    out_stride_b, out_stride_c, out_stride_w,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_size = tl.num_programs(0) // W
    b = pid // W
    w = pid % W

    h_idx = tl.arange(0, BLOCK_H)
    mask = h_idx < H

    input_offset = b * stride_b + h_idx * stride_h + w * stride_w
    val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)

    total = tl.sum(val, axis=0)
    base = total * 0.5 * (1.0 + tl.erf(total * 0.70710678118))

    c_idx = tl.arange(0, 16)
    bias_vals = tl.load(bias_ptr + c_idx, mask=c_idx < 16, other=0.0)
    output_vals = base + bias_vals
    output_offsets = b * out_stride_b + c_idx * out_stride_c + w * out_stride_w
    tl.store(output_ptr + output_offsets, output_vals, mask=c_idx < 16)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x_min = torch.min(x, dim=1, keepdim=True)[0].squeeze(1)
        
        batch, H, W = x_min.shape
        output = torch.empty(batch, 16, 1, W, device=x.device, dtype=x.dtype)
        bias_1d = self.bias.view(-1)
        
        block_size = triton.next_power_of_2(H)
        grid = (batch * W,)
        
        fused_operations_kernel[grid](
            x_min, bias_1d, output,
            H, W,
            x_min.stride(0), x_min.stride(1), x_min.stride(2),
            output.stride(0), output.stride(1), output.stride(3),
            block_size,
        )
        
        return output

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]
# =================== EVOLVE-BLOCK-END ===================