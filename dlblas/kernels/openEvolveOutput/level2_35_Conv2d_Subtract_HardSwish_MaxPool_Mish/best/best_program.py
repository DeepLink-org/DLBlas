# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

# Define fused subtract + HardSwish Triton kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_subtract_hardswish_kernel(
    input_ptr,
    output_ptr,
    subtract_value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Fused operations: subtract + HardSwish
    x = x - subtract_value
    # HardSwish: x * relu6(x + 3) / 6
    y = x * tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) / 6.0
    
    # Store result
    tl.store(output_ptr + offsets, y, mask=mask)

def fused_subtract_hardswish(x: torch.Tensor, value: float):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_subtract_hardswish_kernel[grid](x, output, value, n_elements)
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)
        
    def forward(self, x):
        x = self.conv(x)
        x = fused_subtract_hardswish(x, self.subtract_value)  # Fused Triton kernel
        x = self.pool(x)
        x = torch.nn.functional.mish(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]
# =================== EVOLVE-BLOCK-END ===================