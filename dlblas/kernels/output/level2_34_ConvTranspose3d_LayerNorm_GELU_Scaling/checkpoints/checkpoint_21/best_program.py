# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Triton kernel for fused LayerNorm + GELU + scaling
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _fused_layernorm_gelu_scale_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    eps,
    scaling_factor,
    n_elements,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * row_stride
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < n_elements

    # Load input row
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load weights
    w = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    b = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)

    # Compute mean and variance
    mean = tl.sum(x, axis=0) / n_elements
    x_centered = x - mean
    x_var = tl.sum(x_centered * x_centered, axis=0) / n_elements
    inv_std = 1.0 / tl.sqrt(x_var + eps)

    # Normalize and affine transform
    x_norm = x_centered * inv_std
    y = x_norm * w + b

    # GELU activation (0.5 * x * (1 + erf(x / sqrt(2)))
    gelu = 0.5 * y * (1.0 + libdevice.erf(y * 0.7071067811865475))
    
    # Apply scaling
    y_out = gelu * scaling_factor

    # Store result
    tl.store(output_ptr + offsets, y_out, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.eps = eps
        self.scaling_factor = scaling_factor
        
        # Compute output spatial dimensions (fixed for given input)
        self.w = (32 - 1) * stride + kernel_size - 2 * padding  # W' = 64
        self.weight = nn.Parameter(torch.ones(self.w))
        self.bias = nn.Parameter(torch.zeros(self.w))

    def forward(self, x):
        x = self.conv_transpose(x)
        n, c, d, h, w = x.shape
        
        # Flatten spatial and channel dimensions
        x_flat = x.contiguous().view(-1, w)
        output = torch.empty_like(x_flat)
        
        # Kernel parameters
        grid = (x_flat.shape[0],)
        block_size = triton.next_power_of_2(w)
        _fused_layernorm_gelu_scale_kernel[grid](
            x_flat,
            self.weight,
            self.bias,
            output,
            self.eps,
            self.scaling_factor,
            n_elements=w,
            row_stride=w,
            BLOCK_SIZE=block_size,
        )
        
        return output.view(n, c, d, h, w)

batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 4
stride = 2
padding = 1
bias = True
eps = 1e-5
scaling_factor = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias, eps, scaling_factor]
# =================== EVOLVE-BLOCK-END ===================