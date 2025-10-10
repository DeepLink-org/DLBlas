# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fused_avg_pool_3d_4x_kernel(
    input_ptr,
    output_ptr,
    input_n, input_c, input_d, input_h, input_w,
    output_d, output_h, output_w,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_elements:
        return

    w = pid % output_w
    h = (pid // output_w) % output_h
    d = (pid // (output_w * output_h)) % output_d
    c = (pid // (output_w * output_h * output_d)) % input_c
    n = pid // (output_w * output_h * output_d * input_c)

    base = n * stride_n + c * stride_c
    d_start = d * 4
    h_start = h * 4
    w_start = w * 4

    total = 0.0
    count = 0
    for dd in range(0, 4):
        d_in = d_start + dd
        if d_in < input_d:
            for hh in range(0, 4):
                h_in = h_start + hh
                if h_in < input_h:
                    for ww in range(0, 4):
                        w_in = w_start + ww
                        if w_in < input_w:
                            offset = base + d_in * stride_d + h_in * stride_h + w_in * stride_w
                            total += tl.load(input_ptr + offset)
                            count += 1

    if count > 0:
        average = total / count
    else:
        average = 0.0

    out_offset = n * out_stride_n + c * out_stride_c + d * out_stride_d + h * out_stride_h + w * out_stride_w
    tl.store(output_ptr + out_offset, average)

def fused_avg_pool_3d_4x(x: torch.Tensor) -> torch.Tensor:
    N, C, D, H, W = x.shape
    output = torch.empty((N, C, D//4, H//4, W//4), device=x.device, dtype=x.dtype)
    
    x = x.contiguous()
    output = output.contiguous()
    
    total_elements = output.numel()
    grid = [total_elements]
    
    stride_n, stride_c, stride_d, stride_h, stride_w = x.stride()
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w = output.stride()
    
    _fused_avg_pool_3d_4x_kernel[grid](
        x, output,
        N, C, D, H, W,
        D//4, H//4, W//4,
        stride_n, stride_c, stride_d, stride_h, stride_w,
        out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
        total_elements,
        BLOCK_SIZE=128,
    )
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = fused_avg_pool_3d_4x(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 32, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]
# =================== EVOLVE-BLOCK-END ===================