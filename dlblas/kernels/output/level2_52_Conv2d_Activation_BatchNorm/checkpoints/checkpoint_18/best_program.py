# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_activation_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute fused activation: x * tanh(softplus(x))
    # Stable softplus computation using basic operations
    abs_x = tl.abs(x)
    softplus = tl.where(
        abs_x > 20.0,
        tl.maximum(x, 0.0),
        tl.log(1 + tl.exp(-abs_x)) + tl.maximum(x, 0.0)
    )
    
    # Compute tanh(softplus) using basic exponentials
    exp_neg_2y = tl.exp(-2 * softplus)
    tanh_sp = (1 - exp_neg_2y) / (1 + exp_neg_2y)
    
    result = x * tanh_sp
    tl.store(output_ptr + offsets, result, mask=mask)

def fused_activation(x):
    n_elements = x.numel()
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_activation_kernel[grid](
        x, output, n_elements,
        BLOCK_SIZE=1024
    )
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.conv(x)
        x = fused_activation(x)
        x = self.bn(x)
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