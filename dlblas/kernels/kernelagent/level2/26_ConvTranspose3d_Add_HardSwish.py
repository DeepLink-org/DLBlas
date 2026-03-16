import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 8192}, num_warps=8, num_stages=2),
    ],
    key=["N"],
)
@triton.jit
def _fused_add_hswish_mul_kernel(x_ptr, add_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    add = tl.load(add_ptr + offs, mask=mask, other=0.0)
    z = x + add

    # Compute HardSwish(z) = z * clip(z + 3, 0, 6) / 6
    three = 3.0
    six = 6.0
    z_p3 = z + three
    clipped = tl.minimum(tl.maximum(z_p3, 0.0), six)
    hswish = z * (clipped / six)

    # Final: z * hardswish(z)
    out = z * hswish
    tl.store(out_ptr + offs, out, mask=mask)


def _fused_add_hswish_mul(x: torch.Tensor, add_input: torch.Tensor) -> torch.Tensor:
    # Fallback to PyTorch path if not CUDA or non-contiguous tensors
    if (not x.is_cuda) or (not add_input.is_cuda):
        z = x + add_input
        return z * torch.nn.functional.hardswish(z)

    # Ensure contiguous for flat indexing
    x_c = x.contiguous()
    add_c = add_input.contiguous()
    out = torch.empty_like(x_c)

    N = x_c.numel()
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    _fused_add_hswish_mul_kernel[grid](x_c, add_c, out, N)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation,
    with a fused Triton kernel for the post-convolution elementwise computation:
      out = (convT(x) + add_input) * hardswish(convT(x) + add_input)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        # Keep parameter to match original API/semantics (not used in forward)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, add_input):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
            add_input (torch.Tensor): Tensor added after transposed convolution, shape (batch_size, out_channels, D, H, W).
        Returns:
            torch.Tensor: Output tensor after fused addition and HardSwish-based multiplication.
        """
        x = self.conv_transpose(x)
        # Fused: (x + add_input) * hardswish(x + add_input)
        x = _fused_add_hswish_mul(x, add_input)
        return x


batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)

def get_inputs():
    return [
        torch.randn(batch_size, in_channels, D, H, W),
        torch.randn(batch_size, out_channels, D * stride, H * stride, W * stride),
    ]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]