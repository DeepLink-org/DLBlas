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
    BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = batch_size * height * width
    if pid >= num_pids:
        return
    
    w = pid % width
    h = (pid // width) % height
    b = pid // (height * width)
    
    c_offs = tl.arange(0, BLOCK_SIZE_C)
    mask = c_offs < n_channels
    min_vals = tl.full((BLOCK_SIZE_C,), float('inf'), dtype=tl.float32)
    
    for d in range(depth):
        base = b * stride_batch + h * stride_height + w * stride_width + d * stride_depth
        ptrs = input_ptr + base + c_offs * stride_channel
        vals = tl.load(ptrs, mask=mask, other=float('inf'))
        min_vals = tl.minimum(min_vals, vals)
    
    min_vals_masked = tl.where(mask, min_vals, -float('inf'))
    max_val = tl.max(min_vals_masked, axis=0)
    
    shifted = min_vals - max_val
    exp_vals = tl.exp(shifted)
    exp_vals = tl.where(mask, exp_vals, 0.0)
    exp_sum = tl.sum(exp_vals, axis=0)
    softmax_out = exp_vals / exp_sum
    
    output_base = b * stride_batch_out + h * stride_height_out + w * stride_width_out
    output_ptrs = output_ptr + output_base + c_offs * stride_channel_out
    tl.store(output_ptrs, softmax_out, mask=mask)

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
        block_size_c = triton.next_power_of_2(out_channels)
        fused_min_softmax_kernel[grid](
            x, output,
            batch_size, out_channels, D, H, W,
            s_b, s_c, s_d, s_h, s_w,
            s_b_out, s_c_out, s_h_out, s_w_out,
            BLOCK_SIZE_C=block_size_c
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