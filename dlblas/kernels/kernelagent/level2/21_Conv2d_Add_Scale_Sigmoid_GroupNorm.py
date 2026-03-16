import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _bias_scale_sigmoid_kernel(
    x_ptr,            # *f32 [N, C, H, W] contiguous
    bias_ptr,         # *f32 [C, 1, 1] contiguous
    scale_ptr,        # *f32 [C, 1, 1] contiguous
    y_ptr,            # *f32 [N, C, H, W] contiguous
    HW: tl.constexpr, # H * W
    C: tl.constexpr,  # channels
    n_elements,       # total elements N*C*H*W
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load input
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Compute channel index for each element: c = (idx // HW) % C
    c_idx = (offs // HW) % C

    # Load bias and scale by channel
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    s = tl.load(scale_ptr + c_idx, mask=mask, other=0.0)

    # Fused: y = sigmoid((x + b) * s)
    z = (x + b) * s
    y = 1.0 / (1.0 + tl.exp(-z))

    # Store output
    tl.store(y_ptr + offs, y, mask=mask)


def fused_bias_scale_sigmoid(x: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor):
    # Fallback for non-CUDA tensors
    if not x.is_cuda:
        return torch.sigmoid((x + bias) * scale)

    # Ensure contiguity for linear indexing and consistent strides
    x_contig = x.contiguous()
    bias_contig = bias.contiguous()
    scale_contig = scale.contiguous()

    N, C, H, W = x_contig.shape
    HW = H * W
    n_elements = x_contig.numel()

    y = torch.empty_like(x_contig)

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    _bias_scale_sigmoid_kernel[grid](
        x_contig, bias_contig, scale_contig, y,
        HW, C, n_elements,
    )
    return y


class ModelNew(nn.Module):
    """
    Model that performs a convolution, adds a bias term, scales, applies sigmoid, and performs group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)
        # Fused bias + scale + sigmoid via Triton on CUDA, fallback on CPU
        x = fused_bias_scale_sigmoid(x, self.bias, self.scale)
        x = self.group_norm(x)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]