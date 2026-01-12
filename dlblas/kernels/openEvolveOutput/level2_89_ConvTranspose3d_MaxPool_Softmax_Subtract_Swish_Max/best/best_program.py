# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_ops_kernel(
    x_ptr,
    subtract_ptr,
    output_ptr,
    channels,
    batch_size,
    D, H, W,
    x_stride_batch, x_stride_channel, x_stride_d, x_stride_h, x_stride_w,
    out_stride_batch, out_stride_d, out_stride_h, out_stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_positions = batch_size * D * H * W
    if pid >= total_positions:
        return
    
    # Compute spatial indices
    w = pid % W
    h = (pid // W) % H
    d = (pid // (W * H)) % D
    batch_id = pid // (W * H * D)
    
    # Base pointer for input
    base = batch_id * x_stride_batch + d * x_stride_d + h * x_stride_h + w * x_stride_w
    
    # Load channel vector
    channel_offsets = tl.arange(0, BLOCK_SIZE)
    mask = channel_offsets < channels
    vec = tl.load(x_ptr + base + channel_offsets * x_stride_channel, mask=mask, other=float('-inf'))
    
    # Compute softmax
    max_val = tl.max(vec, axis=0)
    vec = vec - max_val
    exp_vec = tl.exp(vec)
    sum_exp = tl.sum(exp_vec, axis=0)
    softmax_vec = exp_vec / sum_exp
    
    # Load subtract vector and apply
    sub_vec = tl.load(subtract_ptr + channel_offsets, mask=mask, other=0.0)
    softmax_vec = softmax_vec - sub_vec
    
    # Swish activation
    swish_vec = softmax_vec * tl.sigmoid(softmax_vec)
    
    # Max reduction
    result = tl.max(swish_vec, axis=0)
    
    # Store result
    out_base = batch_id * out_stride_batch + d * out_stride_d + h * out_stride_h + w * out_stride_w
    tl.store(output_ptr + out_base, result)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        
        # Extract tensor dimensions
        batch, channels, D, H, W = x.shape
        total_positions = batch * D * H * W
        
        # Prepare output tensor
        output = torch.empty(batch, D, H, W, device=x.device, dtype=x.dtype)
        
        # Get strides
        x_strides = x.stride()
        out_strides = output.stride()
        
        # Launch kernel
        BLOCK_SIZE = triton.next_power_of_2(channels)
        grid = (total_positions,)
        fused_ops_kernel[grid](
            x, self.subtract, output, channels, batch, D, H, W,
            x_strides[0], x_strides[1], x_strides[2], x_strides[3], x_strides[4],
            out_strides[0], out_strides[1], out_strides[2], out_strides[3],
            BLOCK_SIZE
        )
        
        return output

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
pool_stride = 2
pool_padding = 0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding]
# =================== EVOLVE-BLOCK-END ===================