# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sigmoid_sum_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    start_idx = pid * n_elements
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (start_idx + n_elements)
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sig_x = 1.0 / (1.0 + tl.exp(-x))
    sig_x = tl.where(mask, sig_x, 0.0)
    row_sum = tl.sum(sig_x, axis=0)
    tl.store(output_ptr + pid, row_sum)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.avg_pool(x)
        
        # Flatten spatial and channel dimensions
        x_flat = x.flatten(1)
        n_elements = x_flat.shape[1]
        output = torch.zeros(x_flat.shape[0], device=x.device, dtype=torch.float32)
        
        # Launch Triton kernel for fused sigmoid+sum
        grid = (x_flat.shape[0],)
        sigmoid_sum_kernel[grid](
            x_flat, output, n_elements, 
            BLOCK_SIZE=triton.next_power_of_2(n_elements)
        )
        return output

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
# =================== EVOLVE-BLOCK-END ===================