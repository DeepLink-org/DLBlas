# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_pointwise(
    x_ptr,
    output_ptr,
    bias_ptr,
    constant_value,
    scaling_factor,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    
    total_elements = N * C * H * W
    elements_per_program = (total_elements + num_pid - 1) // num_pid
    start_idx = pid * elements_per_program
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    n = offsets // (C * H * W)
    c = (offsets % (C * H * W)) // (H * W)
    hw = offsets % (H * W)
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias_val = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x = tl.minimum(x, constant_value)
    x = x + bias_val
    x = x * scaling_factor
    
    tl.store(output_ptr + offsets, x, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv(x)
        N, C, H, W = x.shape
        
        output = torch.empty_like(x)
        total_elements = N * C * H * W
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
        
        fused_pointwise[grid](
            x, output, self.bias.view(C),
            self.constant_value, self.scaling_factor,
            N, C, H, W,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return output

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]
# =================== EVOLVE-BLOCK-END ===================