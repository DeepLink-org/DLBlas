# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _logsumexp_relu_kernel(
    x_ptr,
    output_ptr,
    n_channels,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_elements:
        return
    row_start = pid * n_channels
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < n_channels
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)
    if m == -float('inf'):
        lse = -float('inf')
    else:
        x_exp = tl.exp(x - m)
        s = tl.sum(x_exp, axis=0)
        lse = m + tl.log(s)
    out = tl.maximum(lse, 0.0)
    tl.store(output_ptr + pid, out)

def logsumexp_relu(x, dim, keepdim=False):
    original_shape = x.shape
    n_channels = x.shape[dim]
    x_flat = x.reshape(-1, n_channels).contiguous()
    n_rows = x_flat.shape[0]
    output_flat = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = triton.next_power_of_2(n_channels)
    grid = (n_rows,)
    _logsumexp_relu_kernel[grid](x_flat, output_flat, n_channels, n_rows, BLOCK_SIZE=BLOCK_SIZE)
    output = output_flat.view(original_shape[0], *original_shape[2:])
    if keepdim:
        output = output.unsqueeze(1)
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = logsumexp_relu(x, dim=1, keepdim=True)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 1
padding = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
# =================== EVOLVE-BLOCK-END ===================