# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_activation_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Optimized fused ReLU + HardSwish
    x = tl.maximum(x, 0.0)  # ReLU
    # Efficient HardSwish using intermediate registers
    t = (x + 3.0) * 0.16666667  # 1/6 constant optimized
    t = tl.minimum(t, 1.0)
    x = x * t
    tl.store(output_ptr + offsets, x, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
    def forward(self, x):
        x = self.conv(x)
        n_elements = x.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        fused_activation_kernel[grid](x, x, n_elements, BLOCK_SIZE=1024)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
# =================== EVOLVE-BLOCK-END ===================