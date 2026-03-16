import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _bias_sub_tanh_kernel(
    x_ptr,         # *T: input/output tensor (N, C, H, W) flattened
    b_ptr,         # *T: bias tensor (C)
    y_ptr,         # *T: output tensor (same as x)
    HW: tl.constexpr,   # H * W
    C: tl.constexpr,    # number of channels
    NCHW: tl.constexpr, # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < NCHW

    # Compute channel index for each element in flattened NCHW layout
    c_idx = (offs // HW) % C

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + c_idx, mask=mask, other=0.0)

    x32 = x.to(tl.float32)
    b32 = b.to(tl.float32)
    z = x32 - b32
    y = libdevice.tanh(z)

    # Store back; Triton will cast to the destination pointer dtype if needed
    tl.store(y_ptr + offs, y, mask=mask)


def _bias_sub_tanh_fused(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # Fused kernel: y = tanh(x - bias) with bias broadcast over (N, H, W)
    if not x.is_cuda:
        # CPU fallback
        return torch.tanh(x - bias)

    # Ensure bias is shape (C,)
    b = bias.reshape(-1)
    if b.dtype != x.dtype:
        b = b.to(dtype=x.dtype)
    if b.device != x.device:
        b = b.to(device=x.device)
    # In-place to reduce memory traffic
    y = x

    N, C, H, W = x.shape
    n_elements = x.numel()
    HW = H * W

    # Launch configuration
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _bias_sub_tanh_kernel[grid](
        x, b, y,
        HW, C, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused bias subtraction + tanh via Triton on CUDA, falls back to PyTorch otherwise
        if x.is_cuda:
            # Ensure bias broadcasting semantics (C,1,1)
            return _bias_sub_tanh_fused(x, self.bias)
        else:
            x = x - self.bias
            x = torch.tanh(x)
            return x


batch_size = 128
in_channels = 32
out_channels = 16
height, width = 16, 16
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]