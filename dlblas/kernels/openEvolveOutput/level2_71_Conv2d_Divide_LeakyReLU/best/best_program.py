# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_div_leaky_relu(
    x_ptr,
    output_ptr,
    divisor,
    negative_slope,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Fused operations: division + leaky ReLU
    x = x / divisor
    leaky_val = x * negative_slope
    x = tl.where(x >= 0, x, leaky_val)
    tl.store(output_ptr + offsets, x, mask=mask)

class ModelNew(nn.Module):
    """
    Optimized model using Triton for fused division + LeakyReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor

    def forward(self, x):
        x = self.conv(x)
        # Ensure tensor is contiguous for Triton kernel
        x_contig = x.contiguous()
        output = torch.empty_like(x_contig)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        # Fuse division and LeakyReLU in single kernel
        fused_div_leaky_relu[grid](
            x_contig, output, self.divisor, 0.01, n_elements, 
            BLOCK_SIZE=1024
        )
        return output

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
divisor = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]
# =================== EVOLVE-BLOCK-END ===================