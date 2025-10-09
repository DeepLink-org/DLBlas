# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    batch_size, channels, height, width,
    out_height, out_width,
    stride, padding,
    stride_b, stride_c, stride_h, stride_w,
    total_elements,
    kernel_size: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= total_elements:
        return

    w_out = pid % out_width
    h_out = (pid // out_width) % out_height
    c_out = (pid // (out_width * out_height)) % channels
    b = pid // (out_width * out_height * channels)

    h_start = h_out * stride - padding
    w_start = w_out * stride - padding

    total = 0.0
    count = 0

    for kh in tl.static_range(kernel_size):
        for kw in tl.static_range(kernel_size):
            h = h_start + kh
            w = w_start + kw
            mask = (h >= 0) & (h < height) & (w >= 0) & (w < width)
            off_b = b * stride_b
            off_c = c_out * stride_c
            off_h = h * stride_h
            off_w = w * stride_w
            off = off_b + off_c + off_h + off_w
            val = tl.load(input_ptr + off, mask=mask, other=0.0)
            total += val
            count += tl.where(mask, 1, 0)

    if count == 0:
        average = 0.0
    else:
        average = total / count

    tl.store(output_ptr + pid, average)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        out_height = (height + 2*self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2*self.padding - self.kernel_size) // self.stride + 1
        output = torch.empty((batch_size, channels, out_height, out_width), 
                             device=x.device, dtype=x.dtype).contiguous()

        stride_b = x.stride(0)
        stride_c = x.stride(1)
        stride_h = x.stride(2)
        stride_w = x.stride(3)
        total_elements = batch_size * channels * out_height * out_width

        grid = lambda meta: (total_elements,)
        _avg_pool2d_kernel[grid](
            x, output,
            batch_size, channels, height, width,
            out_height, out_width,
            self.stride, self.padding,
            stride_b, stride_c, stride_h, stride_w,
            total_elements,
            self.kernel_size,
        )
        return output

batch_size = 16
channels = 64
height = 256
width = 256
kernel_size = 3

def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size]
# =================== EVOLVE-BLOCK-END ===================