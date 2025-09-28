# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def clamp_softmax_multiply_kernel(
    x_ptr,
    output_ptr,
    clamp_min,
    clamp_max,
    C,
    n_spatial,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch_id = pid // n_spatial
    spatial_id = pid % n_spatial
    c_offsets = tl.arange(0, BLOCK_SIZE)
    mask = c_offsets < C
    
    # Calculate base pointer position
    base_ptr = batch_id * C * n_spatial + spatial_id
    ptrs = x_ptr + base_ptr + c_offsets * n_spatial
    vec = tl.load(ptrs, mask=mask, other=-float('inf'))
    
    # Clamp values
    clamped = tl.minimum(tl.maximum(vec, clamp_min), clamp_max)
    
    # Compute softmax
    max_val = tl.max(clamped, axis=0)
    exp_vals = tl.exp(clamped - max_val)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_out = exp_vals / sum_exp
    
    # Multiply by 2 and store
    result = softmax_out * 2.0
    tl.store(output_ptr + base_ptr + c_offsets * n_spatial, result, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.avg_pool(x)
        
        # Get tensor dimensions
        batch, C, D, H, W = x.shape
        n_spatial = D * H * W
        
        # Prepare for Triton kernel
        x_contig = x.contiguous()
        output = torch.empty_like(x_contig)
        
        # Launch kernel only if there are elements to process
        if batch * n_spatial > 0 and C > 0:
            grid = (batch * n_spatial,)
            BLOCK_SIZE = triton.next_power_of_2(C)
            clamp_softmax_multiply_kernel[grid](
                x_contig, output, self.clamp_min, self.clamp_max, C, n_spatial,
                BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            output = x_contig
            
        return output

batch_size = 16
in_channels = 8
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max]
# =================== EVOLVE-BLOCK-END ===================