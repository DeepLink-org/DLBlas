# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OW': 64}, num_warps=4),
        triton.Config({'BLOCK_OW': 128}, num_warps=4),
        triton.Config({'BLOCK_OW': 256}, num_warps=8),
        triton.Config({'BLOCK_OW': 64}, num_warps=8),
        triton.Config({'BLOCK_OW': 128}, num_warps=8),
    ],
    key=['OW'],
)
@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,
    w_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    height,
    width,
    out_channels,
    OH,
    OW,
    kernel_size,
    stride,
    padding,
    BLOCK_OW: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_owb = tl.program_id(2)
    
    batch_idx = pid_bc // out_channels
    channel_idx = pid_bc % out_channels
    group_idx = channel_idx // (out_channels // in_channels)
    
    ow_start = pid_owb * BLOCK_OW
    ow_offsets = ow_start + tl.arange(0, BLOCK_OW)
    ow_mask = ow_offsets < OW
    
    acc = tl.zeros((BLOCK_OW,), dtype=tl.float32)
    
    # Preload kernel weights using vectorized load
    w_base = channel_idx * kernel_size * kernel_size
    w_vals = tl.load(w_ptr + w_base + tl.arange(0, kernel_size*kernel_size))
    
    # Convolution computation with unrolled loops
    for kh in tl.static_range(kernel_size):
        for kw in tl.static_range(kernel_size):
            h_in = pid_oh * stride + kh - padding
            w_in = ow_offsets * stride + kw - padding
            
            h_valid = (h_in >= 0) & (h_in < height)
            w_valid = (w_in >= 0) & (w_in < width)
            valid_mask = h_valid & w_valid & ow_mask
            
            x_offsets = batch_idx * in_channels * height * width + \
                        group_idx * height * width + \
                        h_in * width + w_in
            x_vals = tl.load(x_ptr + x_offsets, mask=valid_mask, other=0.0)
            
            w_val = w_vals[kh * kernel_size + kw]
            acc += x_vals * w_val
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + channel_idx)
        acc += bias
    
    output_offsets = batch_idx * out_channels * OH * OW + \
                    channel_idx * OH * OW + \
                    pid_oh * OW + ow_offsets
    tl.store(output_ptr + output_offsets, acc, mask=ow_mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = nn.Parameter(torch.empty(out_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
            
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = in_channels * kernel_size * kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous tensors
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        batch_size, _, height, width = x.shape
        OH = (height + 2*self.padding - self.kernel_size) // self.stride + 1
        OW = (width + 2*self.padding - self.kernel_size) // self.stride + 1
        
        output = torch.empty((batch_size, self.out_channels, OH, OW), 
                            device=x.device, dtype=x.dtype)
        
        grid = lambda meta: (batch_size * self.out_channels, OH, triton.cdiv(OW, meta['BLOCK_OW']))
        bias_ptr = self.bias.data_ptr() if self.bias is not None else None
        
        depthwise_conv2d_kernel[grid](
            x, weight, output, bias_ptr,
            batch_size, self.in_channels, height, width, self.out_channels,
            OH, OW, self.kernel_size, self.stride, self.padding,
        )
        return output

# Test code
batch_size = 16
in_channels = 3
out_channels = 3
kernel_size = 3
width_in = 256
height_in = 128
stride = 1
padding = 0

def get_inputs():
    x = torch.randn(batch_size, in_channels, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
# =================== EVOLVE-BLOCK-END ===================