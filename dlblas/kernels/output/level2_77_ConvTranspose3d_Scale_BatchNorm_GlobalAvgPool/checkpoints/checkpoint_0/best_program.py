# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def global_avg_pool_kernel(
    input_ptr,
    output_ptr,
    input_batch_stride,
    input_channel_stride,
    input_depth_stride,
    input_height_stride,
    input_width_stride,
    depth: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    
    # Initialize accumulator
    accum = tl.zeros((1,), dtype=tl.float32)
    count = depth * height * width
    
    # Loop over depth dimension
    for d in range(0, depth, BLOCK_D):
        d_idx = d + tl.arange(0, BLOCK_D)
        d_mask = d_idx < depth
        
        # Loop over height dimension
        for h in range(0, height, BLOCK_H):
            h_idx = h + tl.arange(0, BLOCK_H)
            h_mask = h_idx < height
            
            # Loop over width dimension
            for w in range(0, width, BLOCK_W):
                w_idx = w + tl.arange(0, BLOCK_W)
                w_mask = w_idx < width
                
                # Create combined mask
                mask = d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]
                
                # Compute pointer offsets
                offsets = (
                    pid_batch * input_batch_stride +
                    pid_channel * input_channel_stride +
                    d_idx[:, None, None] * input_depth_stride +
                    h_idx[None, :, None] * input_height_stride +
                    w_idx[None, None, :] * input_width_stride
                )
                
                # Load data with mask
                data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
                accum += tl.sum(data)
    
    # Compute average and store result
    avg = accum / count
    tl.store(output_ptr + pid_batch * output_batch_stride + pid_channel * output_channel_stride, avg)

def global_avg_pool_3d_triton(x):
    batch_size, channels = x.shape[0], x.shape[1]
    depth, height, width = x.shape[2], x.shape[3], x.shape[4]
    
    # Create output tensor
    output = torch.empty((batch_size, channels, 1, 1, 1), device=x.device, dtype=torch.float32)
    
    # Compute strides
    input_batch_stride = x.stride(0)
    input_channel_stride = x.stride(1)
    input_depth_stride = x.stride(2)
    input_height_stride = x.stride(3)
    input_width_stride = x.stride(4)
    output_batch_stride = output.stride(0)
    output_channel_stride = output.stride(1)
    
    # Kernel configuration
    BLOCK_D, BLOCK_H, BLOCK_W = 8, 16, 16
    grid = (batch_size, channels)
    
    # Launch kernel
    global_avg_pool_kernel[grid](
        x,
        output,
        input_batch_stride,
        input_channel_stride,
        input_depth_stride,
        input_height_stride,
        input_width_stride,
        depth,
        height,
        width,
        BLOCK_D,
        BLOCK_H,
        BLOCK_W,
        output_batch_stride=output_batch_stride,
        output_channel_stride=output_channel_stride
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        
    def forward(self, x):
        x = self.conv_transpose(x)
        x = x * self.scale_factor
        x = self.batch_norm(x)
        x = global_avg_pool_3d_triton(x)
        return x

batch_size = 16
in_channels = 64
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]
# =================== EVOLVE-BLOCK-END ===================