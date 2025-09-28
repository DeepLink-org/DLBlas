# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fused_mish_tanh_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute stable softplus: log(1 + exp(-|x|)) + max(x, 0)
    abs_x = tl.abs(x)
    max_x = tl.maximum(x, 0.0)
    log_exp = tl.log(1.0 + tl.exp(-abs_x))
    softplus = log_exp + max_x
    
    # Compute Mish: x * tanh(softplus)
    tanh_softplus = tl.tanh(softplus)
    mish = x * tanh_softplus
    
    # Compute final Tanh
    out = tl.tanh(mish)
    tl.store(output_ptr + offsets, out, mask=mask)

def fused_mish_tanh(x):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _fused_mish_tanh_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies fused Mish-Tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        x = self.conv(x)
        x = fused_mish_tanh(x)
        return x

batch_size = 16
in_channels = 3
out_channels = 16
D, H, W = 16, 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
# =================== EVOLVE-BLOCK-END ===================