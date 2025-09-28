# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_operations_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    scale,
    clamp_min,
    clamp_max,
    input_batch_stride, input_channel_stride, input_d_stride, input_h_stride, input_w_stride,
    output_batch_stride, output_channel_stride,
    in_channels, out_channels, 
    depth, height, width,
    kernel_size, stride, padding,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = pid // out_channels
    channel_id = pid % out_channels
    
    # Initialize accumulator
    acc = 0.0
    
    # Loop over input depth dimension
    for d in range(depth):
        # Loop over kernel depth dimension
        for kd in range(kernel_size):
            out_d = d * stride - padding + kd
            if out_d < 0 or out_d >= depth * stride:
                continue
                
            # Loop over input height dimension
            for h in range(height):
                # Loop over kernel height dimension
                for kh in range(kernel_size):
                    out_h = h * stride - padding + kh
                    if out_h < 0 or out_h >= height * stride:
                        continue
                    
                    # Loop over input width dimension
                    for w in range(width):
                        # Loop over kernel width dimension
                        for kw in range(kernel_size):
                            out_w = w * stride - padding + kw
                            if out_w < 0 or out_w >= width * stride:
                                continue
                            
                            # Calculate input and weight indices
                            input_idx = (
                                batch_id * input_batch_stride +
                                d * input_d_stride +
                                h * input_h_stride +
                                w * input_w_stride
                            )
                            
                            weight_idx = (
                                channel_id * (kernel_size**3 * in_channels) +
                                kd * (kernel_size**2 * in_channels) +
                                kh * (kernel_size * in_channels) +
                                kw * in_channels
                            )
                            
                            # Loop over input channels
                            for c in range(in_channels):
                                input_val = tl.load(input_ptr + input_idx + c * input_channel_stride)
                                weight_val = tl.load(weight_ptr + weight_idx + c)
                                acc += input_val * weight_val
    
    # Apply scale and clamp
    acc = acc * scale
    acc = tl.minimum(tl.maximum(acc, clamp_min), clamp_max)
    
    # Store result
    output_idx = batch_id * output_batch_stride + channel_id * output_channel_stride
    tl.store(output_ptr + output_idx, acc)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.scale = scale
        self.clamp_min = 0
        self.clamp_max = 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        )

    def forward(self, x):
        # Get tensor shape information
        batch_size, in_channels, depth, height, width = x.shape
        out_channels = self.weight.shape[0]
        
        # Calculate output dimensions
        out_depth = (depth * self.stride)
        out_height = (height * self.stride)
        out_width = (width * self.stride)
        
        # Create output tensor
        output = torch.empty(
            batch_size, 
            out_channels, 
            1, 1, 1,  # Final output after pooling is 1x1x1
            device=x.device, 
            dtype=x.dtype
        )
        
        # Get strides for input tensor
        strides = x.stride()
        
        # Launch kernel
        grid = (batch_size * out_channels,)
        fused_operations_kernel[grid](
            x,
            self.weight,
            output,
            self.scale,
            self.clamp_min,
            self.clamp_max,
            strides[0], strides[1], strides[2], strides[3], strides[4],
            output.stride(0), output.stride(1),
            in_channels, out_channels,
            depth, height, width,
            self.kernel_size, self.stride, self.padding,
            BLOCK_SIZE=32
        )
        
        return output

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale = 0.5
maxpool_kernel_size = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size]
# =================== EVOLVE-BLOCK-END ===================