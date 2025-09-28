# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_channels, out_channels, groups,
    height, width, kernel_h, kernel_w,
    stride_h, stride_w, padding_h, padding_w,
    dilation_h, dilation_w,
    input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
    weight_outc_stride, weight_inc_stride, weight_h_stride, weight_w_stride,
    output_batch_stride, output_channel_stride, output_height_stride, output_width_stride,
    out_height, out_width,
    BLOCK_SIZE: tl.constexpr
):
    # Parallelize over output channels and spatial positions
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(out_channels, BLOCK_SIZE)
    pid_oc = pid % num_pid_n
    pid_spatial = pid // num_pid_n
    
    # Calculate output spatial positions
    oh = pid_spatial // out_width
    ow = pid_spatial % out_width
    
    # Channel block range
    oc_block = tl.arange(0, BLOCK_SIZE)
    oc_idx = oc_block + pid_oc * BLOCK_SIZE
    channel_mask = oc_idx < out_channels
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Kernel position offsets
    kh_start = tl.maximum(0, (padding_h - oh * stride_h + dilation_h - 1) // dilation_h)
    kh_end = tl.minimum(kernel_h, (height + padding_h - oh * stride_h + dilation_h - 1) // dilation_h + 1)
    kw_start = tl.maximum(0, (padding_w - ow * stride_w + dilation_w - 1) // dilation_w)
    kw_end = tl.minimum(kernel_w, (width + padding_w - ow * stride_w + dilation_w - 1) // dilation_w + 1)
    
    # Input position
    ih0 = oh * stride_h - padding_h
    iw0 = ow * stride_w - padding_w
    
    # Loop through input channels
    for ic in range(in_channels):
        # Loop through kernel height
        for kh in range(kh_start, kh_end):
            ih = ih0 + kh * dilation_h
            # Loop through kernel width
            for kw in range(kw_start, kw_end):
                iw = iw0 + kw * dilation_w
                
                # Load input with boundary check
                input_offset = ih * input_height_stride + iw * input_width_stride + ic * input_channel_stride
                input_val = tl.load(input_ptr + input_offset, mask=(ih >= 0) & (ih < height) & (iw >= 0) & (iw < width), other=0.0)
                
                # Load weights
                weight_offset = (oc_idx[:, None] * weight_outc_stride + 
                                ic * weight_inc_stride + 
                                kh * weight_h_stride + 
                                kw * weight_w_stride)
                weight_val = tl.load(weight_ptr + weight_offset, mask=channel_mask[:, None], other=0.0)
                
                # Accumulate
                acc += tl.sum(input_val * weight_val, axis=1)
    
    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + oc_idx, mask=channel_mask, other=0.0)
        acc += bias
    
    # Store output
    output_offset = oc_idx * output_channel_stride + oh * output_height_stride + ow * output_width_stride
    tl.store(output_ptr + output_offset, acc, mask=channel_mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights
        self.weight = nn.Parameter(torch.empty(
            out_channels, 
            in_channels // groups, 
            kernel_size[0], 
            kernel_size[1]
        ))
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
            
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        # Compute output shape
        batch_size, _, height, width = x.shape
        out_height = (height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        out_width = (width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        
        # Allocate output tensor
        output = torch.empty(
            batch_size, 
            self.out_channels, 
            out_height, 
            out_width, 
            device=x.device, 
            dtype=x.dtype
        )
        
        # Prepare bias pointer
        bias_ptr = self.bias.data_ptr() if self.bias is not None else None
        
        # Calculate number of blocks
        grid = lambda meta: (triton.cdiv(self.out_channels, meta['BLOCK_SIZE']) * out_height * out_width,)
        
        # Process each batch element separately
        for b in range(batch_size):
            conv2d_kernel[grid](
                x[b],  # Current batch element
                self.weight,
                bias_ptr,
                output[b],
                self.in_channels // self.groups,
                self.out_channels,
                self.groups,
                height,
                width,
                self.kernel_size[0],
                self.kernel_size[1],
                self.stride[0],
                self.stride[1],
                self.padding[0],
                self.padding[1],
                self.dilation[0],
                self.dilation[1],
                x.stride(1),
                x.stride(2),
                x.stride(3),
                self.weight.stride(0),
                self.weight.stride(1),
                self.weight.stride(2),
                self.weight.stride(3),
                output.stride(1),
                output.stride(2),
                output.stride(3),
                out_height,
                out_width,
                BLOCK_SIZE=64
            )
        
        return output

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5)  # Asymmetric kernel
height = 256
width = 128  # Asymmetric input dimensions

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
# =================== EVOLVE-BLOCK-END ===================