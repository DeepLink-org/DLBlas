# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_min_softmax_kernel(
    input_ptr,
    output_ptr,
    batch_size, n_channels, depth, height, width,
    stride_batch, stride_channel, stride_depth, stride_height, stride_width,
    stride_batch_out, stride_channel_out, stride_height_out, stride_width_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = batch_size * height * width
    if pid >= num_pids:
        return
    
    w = pid % width
    h = (pid // width) % height
    b = pid // (height * width)
    
    min_vals = tl.full((n_channels,), float('inf'), dtype=tl.float32)
    
    for d in range(depth):
        for c in range(n_channels):
            offset = b * stride_batch + c * stride_channel + d * stride_depth + h * stride_height + w * stride_width
            val = tl.load(input_ptr + offset)
            if val < min_vals[c]:
                min_vals = tl.store(min_vals, min_vals[c] - (min_vals[c] - val) * (val < min_vals[c]), c)
    
    max_val = min_vals[0]
    for i in range(1, n_channels):
        if min_vals[i] > max_val:
            max_val = min_vals[i]
    
    exp_vals = tl.zeros((n_channels,), dtype=tl.float32)
    exp_sum = 0.0
    for i in range(n_channels):
        exp_val = tl.exp(min_vals[i] - max_val)
        exp_vals = tl.store(exp_vals, exp_val, i)
        exp_sum += exp_val
    
    for c in range(n_channels):
        output_val = exp_vals[c] / exp_sum
        output_offset = b * stride_batch_out + c * stride_channel_out + h * stride_height_out + w * stride_width_out
        tl.store(output_ptr + output_offset, output_val)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous()
        
        batch_size, out_channels, D, H, W = x.shape
        output = torch.empty((batch_size, out_channels, H, W), device=x.device, dtype=x.dtype)
        
        s_b, s_c, s_d, s_h, s_w = x.stride()
        s_b_out, s_c_out, s_h_out, s_w_out = output.stride()
        
        grid = (batch_size * H * W,)
        fused_min_softmax_kernel[grid](
            x, output,
            batch_size, out_channels, D, H, W,
            s_b, s_c, s_d, s_h, s_w,
            s_b_out, s_c_out, s_h_out, s_w_out,
            BLOCK_SIZE=1
        )
        
        return output

batch_size = 128
in_channels = 3
out_channels = 16
D, H, W = 16, 32, 32
kernel_size = 3
dim = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]
# =================== EVOLVE-BLOCK-END ===================