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
    output_batch_stride,
    output_channel_stride,
    depth: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    
    total = 0.0
    num_elements = depth * height * width
    
    for d in range(0, depth, BLOCK_D):
        d_offsets = d + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < depth
        
        for h in range(0, height, BLOCK_H):
            h_offsets = h + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < height
            
            for w in range(0, width, BLOCK_W):
                w_offsets = w + tl.arange(0, BLOCK_W)
                w_mask = w_offsets < width
                
                mask = d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]
                base_offset = (
                    pid_batch * input_batch_stride +
                    pid_channel * input_channel_stride +
                    d_offsets[:, None, None] * input_depth_stride +
                    h_offsets[None, :, None] * input_height_stride +
                    w_offsets[None, None, :] * input_width_stride
                )
                
                data = tl.load(input_ptr + base_offset, mask=mask, other=0.0)
                total += tl.sum(data)
    
    avg = total / num_elements
    output_offset = pid_batch * output_batch_stride + pid_channel * output_channel_stride
    tl.store(output_ptr + output_offset, avg)

def global_avg_pool_3d_triton(x):
    batch_size, channels = x.shape[0], x.shape[1]
    depth, height, width = x.shape[2], x.shape[3], x.shape[4]
    
    output = torch.empty((batch_size, channels, 1, 1, 1), device=x.device, dtype=x.dtype)
    
    BLOCK_D, BLOCK_H, BLOCK_W = 8, 16, 16
    grid = (batch_size, channels)
    
    global_avg_pool_kernel[grid](
        x,
        output,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        output.stride(0), output.stride(1),
        depth, height, width,
        BLOCK_D, BLOCK_H, BLOCK_W
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