# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def _conv2d_forward(
    # Tensors
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Input tensor dimensions
    in_channels, height, width,
    # Weight tensor dimensions
    out_channels, kernel_h, kernel_w,
    # Stride, padding, dilation
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    # Input tensor strides
    input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
    # Weight tensor strides
    weight_out_channel_stride, weight_in_channel_stride, weight_kernel_h_stride, weight_kernel_w_stride,
    # Output tensor strides
    output_batch_stride, output_channel_stride, output_height_stride, output_width_stride,
    # Output dimensions
    height_out, width_out,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Compute program ID
    pid_boc = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_x = tl.program_id(2)
    
    # Calculate batch index and output channel
    batch_idx = pid_boc // out_channels
    oc = pid_boc % out_channels
    
    # Check spatial boundaries
    if pid_y >= height_out or pid_x >= width_out:
        return
    
    # Initialize accumulator
    accum = 0.0
    
    # Precompute kernel volume and input strides
    kernel_vol = kernel_h * kernel_w
    
    # Loop over kernel positions
    for kh in range(kernel_h):
        h_in = pid_y * stride_h + kh * dilation_h - padding_h
        for kw in range(kernel_w):
            w_in = pid_x * stride_w + kw * dilation_w - padding_w
            # Check input boundaries
            if h_in >= 0 and h_in < height and w_in >=0 and w_in < width:
                # Loop over input channels
                for ic in range(in_channels):
                    # Calculate input pointer
                    input_offset = (batch_idx * input_batch_stride + 
                                   ic * input_channel_stride + 
                                   h_in * input_height_stride + 
                                   w_in * input_width_stride)
                    input_val = tl.load(input_ptr + input_offset)
                    
                    # Calculate weight pointer
                    weight_offset = (oc * weight_out_channel_stride + 
                                    ic * weight_in_channel_stride + 
                                    kh * weight_kernel_h_stride + 
                                    kw * weight_kernel_w_stride)
                    weight_val = tl.load(weight_ptr + weight_offset)
                    
                    # Accumulate
                    accum += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + oc)
        accum += bias_val
    
    # Calculate output pointer
    output_offset = (batch_idx * output_batch_stride + 
                     oc * output_channel_stride + 
                     pid_y * output_height_stride + 
                     pid_x * output_width_stride)
    tl.store(output_ptr + output_offset, accum)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        assert groups == 1, "Only groups=1 supported in Triton kernel"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute output dimensions
        h_in, w_in = x.shape[2], x.shape[3]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation
        
        height_out = (h_in + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        width_out = (w_in + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
        
        # Create output tensor
        output = torch.empty((x.shape[0], self.out_channels, height_out, width_out), 
                             device=x.device, dtype=x.dtype)
        
        # Launch kernel
        grid = (x.shape[0] * self.out_channels, height_out, width_out)
        _conv2d_forward[grid](
            x, self.weight, self.bias, output,
            self.in_channels, h_in, w_in,
            self.out_channels, kernel_h, kernel_w,
            stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.weight.stride(0), self.weight.stride(1), self.weight.stride(2), self.weight.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            height_out, width_out,
            BLOCK_SIZE=1,
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