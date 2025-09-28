# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_operations_kernel(
    input_ptr,
    bias_ptr,
    output_ptr,
    in_channels,
    in_height,
    in_width,
    stride_batch,
    stride_channel,
    stride_height,
    stride_width,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr
):
    pid = tl.program_id(0)
    spatial_size = in_height * in_width
    
    # Initialize log-sum-exp state
    max_val = -float('inf')
    exp_sum = 0.0
    
    # Process each channel
    for c in range(in_channels):
        spatial_sum = 0.0
        
        # Process spatial dimension in blocks
        for block_idx in range(0, spatial_size, BLOCK_SIZE):
            spatial_offsets = block_idx + tl.arange(0, BLOCK_SIZE)
            mask = spatial_offsets < spatial_size
            
            # Compute pointer and load data
            ptr = input_ptr + pid * stride_batch + c * stride_channel + spatial_offsets * stride_height
            data = tl.load(ptr, mask=mask, other=0.0)
            spatial_sum += tl.sum(data, axis=0)
        
        # Compute channel average and add bias
        channel_avg = spatial_sum / spatial_size
        bias_val = tl.load(bias_ptr + c)
        val = channel_avg + bias_val
        
        # Update log-sum-exp state
        if max_val == -float('inf'):
            max_val = val
            exp_sum = 1.0
        elif val > max_val:
            exp_sum = exp_sum * tl.exp(max_val - val) + 1.0
            max_val = val
        else:
            exp_sum += tl.exp(val - max_val)
    
    # Compute final result
    result = (tl.log(exp_sum) + max_val) * 10.0
    tl.store(output_ptr + pid, result)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape).squeeze())
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        batch_size, _, h, w = x.shape
        
        # Prepare inputs for Triton kernel
        x_flat = x.contiguous().view(batch_size, self.out_channels, -1)
        output = torch.empty(batch_size, device=x.device, dtype=torch.float32)
        
        # Launch kernel with optimized block size
        grid = (batch_size,)
        fused_operations_kernel[grid](
            x_flat,
            self.bias,
            output,
            self.out_channels,
            h,
            w,
            x_flat.stride(0),
            x_flat.stride(1),
            x_flat.stride(2),
            1,
            BLOCK_SIZE=128,  # Optimized spatial block size
            P2=0
        )
        
        return output.unsqueeze(1)

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
# =================== EVOLVE-BLOCK-END ===================