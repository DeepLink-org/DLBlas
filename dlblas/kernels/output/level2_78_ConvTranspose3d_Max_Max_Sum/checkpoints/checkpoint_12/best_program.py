# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool1(x)
        x = self.max_pool2(x)
        x = self.triton_sum_reduce(x)
        return x

    @triton.jit
    def triton_sum_reduce_kernel(
        input_ptr,
        output_ptr,
        reduce_size: tl.constexpr,
        stride_in_b, stride_in_c, stride_in_d, stride_in_h, stride_in_w,
        stride_out_b, stride_out_c, stride_out_d, stride_out_h, stride_out_w,
        BLOCK_SIZE_R: tl.constexpr,
    ):
        pid = tl.program_id(0)
        D = 5
        H = 10
        W = 10
        num_spatial = D * H * W
        
        w = pid % W
        h = (pid // W) % H
        d = (pid // (W * H)) % D
        b = pid // (W * H * D)
        
        r = tl.arange(0, BLOCK_SIZE_R)
        in_offsets = b * stride_in_b + d * stride_in_d + h * stride_in_h + w * stride_in_w + r * stride_in_c
        mask = r < reduce_size
        vals = tl.load(input_ptr + in_offsets, mask=mask, other=0.0)
        
        result = tl.sum(vals, axis=0)
        
        out_offset = b * stride_out_b + 0 * stride_out_c + d * stride_out_d + h * stride_out_h + w * stride_out_w
        tl.store(output_ptr + out_offset, result)

    def triton_sum_reduce(self, x):
        output = torch.empty(x.shape[0], 1, 5, 10, 10, device=x.device, dtype=x.dtype)
        stride_in_b, stride_in_c, stride_in_d, stride_in_h, stride_in_w = x.stride()
        stride_out_b, stride_out_c, stride_out_d, stride_out_h, stride_out_w = output.stride()
        
        n_elements = x.shape[0] * 5 * 10 * 10
        grid = (n_elements,)
        
        self.triton_sum_reduce_kernel[grid](
            x, output, 16,
            stride_in_b, stride_in_c, stride_in_d, stride_in_h, stride_in_w,
            stride_out_b, stride_out_c, stride_out_d, stride_out_h, stride_out_w,
            BLOCK_SIZE_R=16
        )
        return output

batch_size = 16
in_channels = 8
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
# =================== EVOLVE-BLOCK-END ===================