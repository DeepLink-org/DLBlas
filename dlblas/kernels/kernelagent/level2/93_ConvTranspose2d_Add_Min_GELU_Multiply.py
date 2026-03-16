import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
    ],
    key=["n_elements"],
)
@triton.jit
def _fused_min_gelu_mul_kernel(
    x_ptr, y_ptr,
    add_value, multiply_value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    # Alignment/contiguity hints for better codegen
    tl.multiple_of(offs, 256)
    tl.max_contiguous(offs, BLOCK_SIZE)

    # Load and upcast to fp32 for numerics
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    in_dtype = x.dtype
    x32 = x.to(tl.float32)

    # x = x + add_value; x = min(x, 0.0)
    x32 = x32 + add_value
    x32 = tl.minimum(x32, 0.0)

    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    t = x32 * inv_sqrt2
    e = libdevice.erf(t)
    scale = 0.5 * multiply_value
    y32 = x32 * (1.0 + e) * scale

    # Store back in original dtype
    tl.store(y_ptr + offs, y32.to(in_dtype), mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = float(add_value)
        self.multiply_value = float(multiply_value)

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused add -> min(.,0) -> GELU(exact) -> mul in a single Triton kernel for performance
        if x.is_cuda and x.numel() > 0:
            x_contig = x.contiguous()
            y = torch.empty_like(x_contig)
            n_elements = x_contig.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            _fused_min_gelu_mul_kernel[grid](
                x_contig, y, self.add_value, self.multiply_value, n_elements
            )
            return y
        else:
            # CPU / non-CUDA fallback: preserve exact PyTorch semantics
            x = x + self.add_value
            x = torch.min(x, torch.tensor(0.0))
            x = torch.nn.functional.gelu(x)
            x = x * self.multiply_value
            return x


batch_size = 128
in_channels = 32
out_channels = 16
height, width = 32, 32
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]