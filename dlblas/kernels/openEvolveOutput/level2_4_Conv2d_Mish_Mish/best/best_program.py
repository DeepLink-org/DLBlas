# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def double_mish_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < n_elements

    x_val = tl.load(x_ptr + off, mask=mask)

    # First Mish activation using exponential and log
    exp_x = tl.exp(x_val)
    softplus_x = tl.log(1 + exp_x)
    mish1 = x_val * (1 - 2 / (tl.exp(2 * softplus_x) + 1))
    
    # Second Mish activation
    exp_mish1 = tl.exp(mish1)
    softplus_mish1 = tl.log(1 + exp_mish1)
    mish2 = mish1 * (1 - 2 / (tl.exp(2 * softplus_mish1) + 1))

    tl.store(output_ptr + off, mish2, mask=mask)

def double_mish(x: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    double_mish_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = double_mish(x)  # Fused double Mish activation
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