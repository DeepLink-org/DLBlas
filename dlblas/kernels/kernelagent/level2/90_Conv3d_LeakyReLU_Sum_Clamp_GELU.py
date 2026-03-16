import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def _fused_post_conv_kernel(
    x_ptr,          # *float32, input from conv: [N, C, D, H, W] flattened
    sum_ptr,        # *float32, per-channel bias: [C]
    y_ptr,          # *float32, output buffer (same shape as x)
    inner,          # int32, D*H*W
    C,              # int32, number of channels
    n_elements,     # int32, total number of elements in x
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # LeakyReLU with negative_slope=0.2
    neg = 0.2
    y = tl.where(x >= 0.0, x, x * neg)

    # Add per-channel bias (sum_tensor) with fast-path when block doesn't cross INNER boundary
    group0 = block_start // inner
    group1 = (block_start + (BLOCK_SIZE - 1)) // inner
    if group0 == group1:
        c_block = group0 % C
        sbias = tl.load(sum_ptr + c_block)  # scalar bias for the whole block
        y = y + sbias
    else:
        c_idx = (offsets // inner) % C
        vbias = tl.load(sum_ptr + c_idx, mask=mask, other=0.0)
        y = y + vbias

    # Clamp to [-1.0, 1.0]
    y = tl.maximum(tl.minimum(y, 1.0), -1.0)

    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    t = y * inv_sqrt2
    erf_t = libdevice.erf(t)
    gelu = 0.5 * y * (1.0 + erf_t)

    tl.store(y_ptr + offsets, gelu, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies LeakyReLU, sums with a tensor, clamps, and applies GELU activation.
    Fuses the elementwise ops after convolution into a single Triton kernel for better performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        # Convolution via PyTorch (cuDNN/cutlass optimized)
        x = self.conv(x)

        # Fused LeakyReLU -> Add (per-channel) -> Clamp -> GELU via Triton
        # Shapes
        N, C, D, H, W = x.shape
        n_elements = x.numel()
        inner = D * H * W

        # Ensure contiguous tensors
        x_contig = x.contiguous()
        bias = self.sum_tensor.view(C).contiguous()
        out = torch.empty_like(x_contig)

        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
        _fused_post_conv_kernel[grid](
            x_contig, bias, out,
            inner, C, n_elements,
        )
        return out


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
sum_tensor_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, sum_tensor_shape]