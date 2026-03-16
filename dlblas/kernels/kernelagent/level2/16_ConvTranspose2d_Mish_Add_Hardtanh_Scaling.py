import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _fused_mish_add_hardtanh_scale_kernel(
    x_ptr, y_ptr,
    n_elements,
    add_value, scale_value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Hints for better vectorization/coalescing
    tl.max_contiguous(offs, BLOCK_SIZE)
    tl.multiple_of(offs, 16)
    mask = offs < n_elements

    # Load; values are single-use so prefer evict_last
    x = tl.load(x_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_last")

    # Mish: x * tanh(softplus(x))
    # Stable softplus: max(x, 0) + log1p(exp(-abs(x)))
    ax = tl.abs(x)
    exp_term = tl.exp(-ax)
    sp = tl.maximum(x, 0.0) + libdevice.log1p(exp_term)

    # Efficient tanh via sigmoid: tanh(z) = 2 / (1 + exp(-2z)) - 1
    # Use exp2 for faster exponent: exp(-2z) = exp2(-2z / ln(2))
    LOG2E = 1.4426950408889634  # 1 / ln(2)
    s2 = 2.0 * sp
    tanh_sp = 2.0 / (1.0 + tl.exp2(-s2 * LOG2E)) - 1.0
    mish = x * tanh_sp

    # Add, clamp to [-1, 1] (hardtanh), then scale
    out = mish + add_value
    out = tl.minimum(tl.maximum(out, -1.0), 1.0)
    out = out * scale_value

    tl.store(y_ptr + offs, out, mask=mask)


def _fused_mish_add_hardtanh_scale(x: torch.Tensor, add_value: float, scale: float) -> torch.Tensor:
    # Fallback to PyTorch on CPU or empty tensor
    if (not x.is_cuda) or x.numel() == 0:
        y = torch.nn.functional.mish(x)
        y = y + add_value
        y = torch.nn.functional.hardtanh(y, min_val=-1, max_val=1)
        y = y * scale
        return y

    # Work in-place to reduce memory traffic and allocations
    x_contig = x.contiguous()
    n_elements = x_contig.numel()
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Alias output to input for in-place epilogue
    _fused_mish_add_hardtanh_scale_kernel[grid](
        x_contig, x_contig,
        n_elements,
        float(add_value), float(scale),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=3,
    )
    return x_contig


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, applies Mish activation, adds a value, 
    applies Hardtanh activation, and scales the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused Triton kernel: mish -> add -> hardtanh -> scale
        x = _fused_mish_add_hardtanh_scale(x, self.add_value, self.scale)
        return x

batch_size = 128
in_channels = 32
out_channels = 64
height, width = 16, 16
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
add_value = 0.5
scale = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]