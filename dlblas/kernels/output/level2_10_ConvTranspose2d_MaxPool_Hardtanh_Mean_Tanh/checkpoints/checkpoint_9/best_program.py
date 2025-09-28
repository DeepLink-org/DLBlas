# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fused_hardtanh_mean_tanh(
    input_ptr,
    output_ptr,
    input_batch_stride,
    input_channel_stride,
    input_height_stride,
    input_width_stride,
    output_batch_stride,
    output_channel_stride,
    min_val,
    max_val,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    num_elements = H * W

    base = pid_batch * input_batch_stride + pid_channel * input_channel_stride
    accumulator = 0.0
    for i in range(0, num_elements, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < num_elements
        h = idx // W
        w = idx % W
        spatial_offset = h * input_height_stride + w * input_width_stride
        ptr = input_ptr + base + spatial_offset
        val = tl.load(ptr, mask=mask, other=0.0)
        clamped = tl.minimum(tl.maximum(val, min_val), max_val)
        block_sum = tl.sum(clamped, axis=0)
        accumulator += block_sum

    mean_val = accumulator / num_elements
    out_val = tl.tanh(mean_val)
    output_offset = pid_batch * output_batch_stride + pid_channel * output_channel_stride
    tl.store(output_ptr + output_offset, out_val)

def fused_hardtanh_mean_tanh(x, min_val, max_val):
    batch, channels, H, W = x.shape
    if not x.is_contiguous():
        x = x.contiguous()
    output = torch.empty((batch, channels, 1, 1), device=x.device, dtype=x.dtype)
    strides = x.stride()
    grid = (batch, channels)
    BLOCK_SIZE = min(256, H * W)
    _fused_hardtanh_mean_tanh[grid](
        x,
        output,
        strides[0], strides[1], strides[2], strides[3],
        output.stride(0), output.stride(1),
        min_val, max_val,
        H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        x = fused_hardtanh_mean_tanh(x, self.hardtanh_min, self.hardtanh_max)
        return x

batch_size = 128
in_channels = 32
out_channels = 64
height, width = 16, 16
kernel_size = 4
stride = 2
padding = 1
maxpool_kernel_size = 2
maxpool_stride = 2
hardtanh_min = -1
hardtanh_max = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max]
# =================== EVOLVE-BLOCK-END ===================