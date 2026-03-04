# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_ops_kernel(
    x_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    add_scalar,
    eps,
    stride_b, stride_c, stride_d, stride_h, stride_w,
    out_stride_b, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    n_batch, n_channels, in_depth, in_height, in_width,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    out_d = pid_d
    out_h = pid_h
    out_w = pid_w
    
    c_offsets = tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < n_channels
    
    pool_sum = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    for d_offset in range(0, 2):
        for h_offset in range(0, 2):
            for w_offset in range(0, 2):
                d = pid_d * 2 + d_offset
                h = pid_h * 2 + h_offset
                w = pid_w * 2 + w_offset
                
                if d < in_depth and h < in_height and w < in_width:
                    x_ptr_offset = pid_b * stride_b + d * stride_d + h * stride_h + w * stride_w
                    x_ptrs = x_ptr + x_ptr_offset + c_offsets * stride_c
                    x = tl.load(x_ptrs, mask=c_mask, other=0.0)
                    x = x + add_scalar
                    
                    # Online mean/variance calculation
                    mean = tl.sum(x, axis=0) / n_channels
                    centered = x - mean
                    var = tl.sum(centered * centered, axis=0) / n_channels
                    rstd = 1.0 / tl.sqrt(var + eps)
                    
                    # LayerNorm
                    normalized = centered * rstd
                    weight = tl.load(weight_ptr + c_offsets, mask=c_mask)
                    bias = tl.load(bias_ptr + c_offsets, mask=c_mask)
                    result = normalized * weight + bias
                    
                    pool_sum += result
                else:
                    pass
    
    # Average pooling (divide by 8)
    pool_avg = pool_sum / 8.0
    
    # GELU activation
    gelu = 0.5 * pool_avg * (1.0 + tl.erf(pool_avg / tl.sqrt(2.0)))
    
    # Store results
    out_offset = pid_b * out_stride_b + out_d * out_stride_d + out_h * out_stride_h + out_w * out_stride_w
    out_ptrs = output_ptr + out_offset + c_offsets * out_stride_c
    tl.store(out_ptrs, gelu, mask=c_mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.gelu = nn.GELU()
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        batch_size, _, depth, height, width = x.shape
        out_depth, out_height, out_width = depth // 2, height // 2, width // 2
        
        # Create output tensor
        output = torch.empty((batch_size, self.out_channels, out_depth, out_height, out_width), 
                            device=x.device, dtype=x.dtype)
        
        # Launch Triton kernel
        grid = (batch_size, out_depth, out_height, out_width)
        fused_ops_kernel[grid](
            x,
            output,
            self.norm.weight.data,
            self.norm.bias.data,
            self.sum_weight,
            self.norm.eps,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
            batch_size, self.out_channels, depth, height, width,
            BLOCK_SIZE_C=triton.next_power_of_2(self.out_channels),
            BLOCK_SIZE_D=1,
            BLOCK_SIZE_H=1,
            BLOCK_SIZE_W=1
        )
        return output

batch_size = 128
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = (1, 1, 1)
output_padding = (1, 1, 1)
sum_weight = 1.0
norm_shape = (out_channels,)
pool_kernel_size = (2, 2, 2)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size]
# =================== EVOLVE-BLOCK-END ===================