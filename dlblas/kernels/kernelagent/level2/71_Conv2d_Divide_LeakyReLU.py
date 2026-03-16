import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=8, num_stages=3),
    ],
    key=["n_elements"],
)
@triton.jit
def _div_leakyrelu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    inv_div,
    neg_slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, 16)

    # Stream from global via L2
    x = tl.load(x_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg")

    # Do math in input dtype
    inv = tl.full((), inv_div, x.dtype)
    slope = tl.full((), neg_slope, x.dtype)
    one = tl.full((), 1.0, x.dtype)
    zero = tl.full((), 0.0, x.dtype)

    y = x * inv
    # Branchless LeakyReLU: y + (slope - 1) * min(y, 0)
    k = slope - one
    y_neg = tl.minimum(y, zero)
    out = y + y_neg * k

    tl.store(y_ptr + offs, out, mask=mask, eviction_policy="evict_first")


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, divides by a constant, and applies LeakyReLU.
    Fuses division and LeakyReLU into a single Triton kernel on CUDA for better performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor

    def forward(self, x):
        x = self.conv(x)

        # Autograd or CPU fallback preserves original semantics
        if (not x.is_cuda) or (torch.is_grad_enabled() and x.requires_grad):
            x = x / self.divisor
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
            return x

        x_contig = x.contiguous()
        out = torch.empty_like(x_contig)
        n_elements = out.numel()
        if n_elements == 0:
            return out

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        inv_div = float(1.0 / float(self.divisor))
        neg_slope = 0.01

        _div_leakyrelu_kernel[grid](
            x_contig, out, n_elements, inv_div, neg_slope
        )
        return out


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