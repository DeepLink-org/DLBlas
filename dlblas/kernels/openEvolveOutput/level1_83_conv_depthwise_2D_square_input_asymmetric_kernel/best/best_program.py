# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def _depthwise_conv2d_kernel(
    x_ptr, 
    weight_ptr, 
    bias_ptr, 
    out_ptr,
    batch, 
    in_channels, 
    height, 
    width,
    kernel_size, 
    stride, 
    padding, 
    dilation,
    height_out, 
    width_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_rows = batch * in_channels * height_out
    if pid >= num_rows:
        return

    batch_id = pid // (in_channels * height_out)
    residual = pid % (in_channels * height_out)
    channel_id = residual // height_out
    h_out = residual % height_out

    weight_channel_ptr = weight_ptr + channel_id * kernel_size
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + channel_id)
    else:
        bias = 0.0

    for w_out_start in range(0, width_out, BLOCK_SIZE):
        w_offsets = w_out_start + tl.arange(0, BLOCK_SIZE)
        w_mask = w_offsets < width_out

        w_in = w_offsets * stride - padding
        w_in_mask = (w_in >= 0) & (w_in < width)

        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for i in range(kernel_size):
            weight_val = tl.load(weight_channel_ptr + i)
            h_in = h_out * stride - padding + i * dilation
            valid_h = (h_in >= 0) & (h_in < height)
            
            x_base_ptr = x_ptr + batch_id * (in_channels * height * width) + channel_id * (height * width) + h_in * width
            ptrs = x_base_ptr + w_in
            mask = valid_h & w_mask & w_in_mask
            x_vals = tl.load(ptrs, mask=mask, other=0.0)
            acc += x_vals * weight_val

        acc += bias
        out_base_ptr = out_ptr + batch_id * (in_channels * height_out * width_out) + channel_id * (height_out * width_out) + h_out * width_out
        out_ptrs = out_base_ptr + w_offsets
        tl.store(out_ptrs, acc, mask=w_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(torch.empty(in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.kernel_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[2], x.shape[3]
        height_out = (height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        width_out = (width + 2 * self.padding - 1) // self.stride + 1

        output = torch.empty((x.shape[0], self.in_channels, height_out, width_out), device=x.device, dtype=x.dtype)

        if x.numel() == 0:
            return output

        total_rows = x.shape[0] * self.in_channels * height_out
        grid = (total_rows,)

        BLOCK_SIZE = 256

        _depthwise_conv2d_kernel[grid](
            x, self.weight, self.bias, output,
            x.shape[0], self.in_channels, height, width,
            self.kernel_size, self.stride, self.padding, self.dilation,
            height_out, width_out,
            BLOCK_SIZE=BLOCK_SIZE
        )

        return output

# Test code
batch_size = 16
in_channels = 3
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0
dilation = 1

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding, dilation]
# =================== EVOLVE-BLOCK-END ===================