# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_ops_pool_kernel(
    input_ptr,
    multiplier_ptr,
    output_ptr,
    batch_size, out_channels, D_in, H_in, W_in,
    stride_b, stride_c, stride_d, stride_h, stride_w,
    multiplier_stride_c,
    D_out, H_out, W_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    elements_per_program = tl.cdiv(batch_size * out_channels * D_out * H_out * W_out, num_pid)
    start_idx = pid * elements_per_program
    end_idx = min(start_idx + elements_per_program, batch_size * out_channels * D_out * H_out * W_out)
    
    for idx in range(start_idx, end_idx, BLOCK_SIZE):
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        
        # Compute 5D indices
        w_out = offsets % W_out
        h_out = (offsets // W_out) % H_out
        d_out = (offsets // (W_out * H_out)) % D_out
        c = (offsets // (W_out * H_out * D_out)) % out_channels
        b = offsets // (W_out * H_out * D_out * out_channels)
        
        # Input spatial indices
        d_in_start = d_out * 2
        h_in_start = h_out * 2
        w_in_start = w_out * 2
        
        # Process 2x2x2 window
        max_vals = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    d_in = d_in_start + di
                    h_in = h_in_start + dj
                    w_in = w_in_start + dk
                    
                    # Check boundaries
                    in_bounds = (d_in < D_in) & (h_in < H_in) & (w_in < W_in)
                    in_offsets = b*stride_b + c*stride_c + d_in*stride_d + h_in*stride_h + w_in*stride_w
                    
                    # Load input with boundary check
                    x_val = tl.load(input_ptr + in_offsets, mask=mask & in_bounds, other=0.0)
                    
                    # First LeakyReLU
                    x_val = tl.where(x_val >= 0, x_val, x_val * 0.2)
                    
                    # Multiply by per-channel scalar
                    scale = tl.load(multiplier_ptr + c * multiplier_stride_c)
                    x_val = x_val * scale
                    
                    # Second LeakyReLU
                    x_val = tl.where(x_val >= 0, x_val, x_val * 0.2)
                    
                    # Update max value
                    max_vals = tl.maximum(max_vals, x_val)
        
        # Store results
        out_offsets = b * (out_channels * D_out * H_out * W_out) + \
                      c * (D_out * H_out * W_out) + \
                      d_out * (H_out * W_out) + \
                      h_out * W_out + w_out
        tl.store(output_ptr + out_offsets, max_vals, mask=mask)

def fused_ops_pool(x, multiplier):
    batch_size, out_channels, D_in, H_in, W_in = x.shape
    D_out, H_out, W_out = D_in // 2, H_in // 2, W_in // 2
    output = torch.empty((batch_size, out_channels, D_out, H_out, W_out), 
                         device=x.device, dtype=x.dtype)
    
    # Prepare multiplier
    multiplier = multiplier.squeeze().contiguous()
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(batch_size * out_channels * D_out * H_out * W_out, meta['BLOCK_SIZE']),)
    fused_ops_pool_kernel[grid](
        x, multiplier, output,
        batch_size, out_channels, D_in, H_in, W_in,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        multiplier.stride(0),
        D_out, H_out, W_out,
        BLOCK_SIZE=1024
    )
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, 
                                                 stride=stride, padding=padding, 
                                                 output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
    
    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_ops_pool(x, self.multiplier)
        return x

batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape]
# =================== EVOLVE-BLOCK-END ===================