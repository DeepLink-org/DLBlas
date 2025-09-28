# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies Swish activation, 
    group normalization, and then HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        
    @triton.jit
    def swish_kernel(
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
        sig_x = tl.sigmoid(x)
        output = x * sig_x
        tl.store(output_ptr + offsets, output, mask=mask)
        
    @triton.jit
    def hardswish_kernel(
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
        clamped = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
        output = x * clamped / 6.0
        tl.store(output_ptr + offsets, output, mask=mask)

    def forward(self, x):
        # Convolution operation remains in PyTorch
        x = self.conv_transpose(x)
        
        # Optimized Swish activation using Triton
        n_elements = x.numel()
        swish_out = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        self.swish_kernel[grid](
            x, swish_out, n_elements, 
            BLOCK_SIZE=1024
        )
        
        # Group normalization remains in PyTorch
        x = self.group_norm(swish_out)
        
        # Optimized HardSwish activation using Triton
        hardswish_out = torch.empty_like(x)
        self.hardswish_kernel[grid](
            x, hardswish_out, n_elements, 
            BLOCK_SIZE=1024
        )
        return hardswish_out

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
groups = 4
eps = 1e-5

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, eps]
# =================== EVOLVE-BLOCK-END ===================