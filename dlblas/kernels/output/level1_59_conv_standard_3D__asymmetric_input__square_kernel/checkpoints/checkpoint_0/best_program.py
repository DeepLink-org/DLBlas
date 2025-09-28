# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    stride,
    padding,
    dilation,
    height,
    width,
    depth,
    height_out,
    width_out,
    in_channels,
    out_channels,
    kernel_size,
    x_stride_b,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    x_stride_d,
    weight_stride_oc,
    weight_stride_ic,
    weight_stride_h,
    weight_stride_w,
    output_stride_b,
    output_stride_c,
    output_stride_h,
    output_stride_w,
    output_stride_d,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = tl.num_programs(0) * BLOCK_SIZE
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < (height_out * width_out * depth * out_channels * tl.program_id(0).shape[0])
    
    # Compute indices
    d_idx = idx % depth
    idx //= depth
    w_idx = idx % width_out
    idx //= width_out
    h_idx = idx % height_out
    idx //= height_out
    oc_idx = idx % out_channels
    b_idx = idx // out_channels
    
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over input channels and kernel positions
    for c_in in range(in_channels):
        for ky in range(kernel_size):
            for kx in range(kernel_size):
                h_in = h_idx * stride + ky * dilation - padding
                w_in = w_idx * stride + kx * dilation - padding
                
                if h_in >= 0 and h_in < height and w_in >= 0 and w_in < width:
                    x_offset = b_idx * x_stride_b + c_in * x_stride_c + h_in * x_stride_h + w_in * x_stride_w + d_idx * x_stride_d
                    w_offset = oc_idx * weight_stride_oc + c_in * weight_stride_ic + ky * weight_stride_h + kx * weight_stride_w
                    
                    x_val = tl.load(x_ptr + x_offset, mask=None)
                    w_val = tl.load(weight_ptr + w_offset)
                    acc += x_val * w_val
    
    output_offset = b_idx * output_stride_b + oc_idx * output_stride_c + h_idx * output_stride_h + w_idx * output_stride_w + d_idx * output_stride_d
    tl.store(output_ptr + output_offset, acc, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, in_channels, height, width, depth = x.shape
        height_out = (height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        width_out = (width + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        
        output = torch.empty(batch, self.out_channels, height_out, width_out, depth, device=x.device, dtype=x.dtype)
        
        # Prepare weight by removing last dimension
        weight = self.weight.squeeze(-1)
        
        # Get strides
        x_stride = x.stride()
        weight_stride = weight.stride()
        output_stride = output.stride()
        
        total_elements = batch * self.out_channels * depth * height_out * width_out
        grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
        
        conv3d_kernel[grid](
            x, weight, output,
            self.stride, self.padding, self.dilation,
            height, width, depth,
            height_out, width_out,
            in_channels, self.out_channels, self.kernel_size,
            x_stride[0], x_stride[1], x_stride[2], x_stride[3], x_stride[4],
            weight_stride[0], weight_stride[1], weight_stride[2], weight_stride[3],
            output_stride[0], output_stride[1], output_stride[2], output_stride[3], output_stride[4],
            BLOCK_SIZE=1024
        )
        
        if self.bias is not None:
            output += self.bias[None, :, None, None, None]
            
        return output

# Test code
import math
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256
depth = 10

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width, depth, device='cuda')
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
# =================== EVOLVE-BLOCK-END ===================