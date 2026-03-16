import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _act_softplus_tanh_mul_kernel(x_ptr, y_ptr, n_elements, THRESHOLD: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Compute y = x * tanh(softplus(x)) elementwise with PyTorch's softplus default (beta=1.0, threshold=20.0).
    For x <= THRESHOLD:
        tanh(softplus(x)) = 1 - 2 / (e^{2x} + 2 e^{x} + 2)
    For x > THRESHOLD (softplus(x) = x in PyTorch): use tanh(softplus(x)) ~ 1.0 in fp32.
    This uses a single exp per element and avoids tl.tanh/log to reduce compute.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x_in = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x = x_in.to(tl.float32)

    use_large = x > THRESHOLD
    # Only compute exp for the "small" branch; set x_small=0 for large branch to keep exp well-conditioned.
    x_small = tl.where(use_large, 0.0, x)
    t = tl.exp(x_small)
    den = t * t + 2.0 * t + 2.0
    tanh_small = 1.0 - 2.0 / den

    tval = tl.where(use_large, 1.0, tanh_small)
    y = (tval * x).to(x_in.dtype)

    tl.store(y_ptr + offs, y, mask=mask)


def fused_softplus_tanh_mul(x: torch.Tensor) -> torch.Tensor:
    # Fallback for non-CUDA tensors or empty inputs
    if (not x.is_cuda) or x.numel() == 0:
        return torch.multiply(torch.tanh(F.softplus(x)), x)
    xi = x.contiguous()
    y = torch.empty_like(xi)
    n = xi.numel()
    BLOCK_SIZE = 4096
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _act_softplus_tanh_mul_kernel[grid](xi, y, n, THRESHOLD=20.0, BLOCK_SIZE=BLOCK_SIZE, num_warps=4, num_stages=2)
    return y


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies activation, and then applies Batch Normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.conv(x)
        # Fused activation: x * tanh(softplus(x))
        x = fused_softplus_tanh_mul(x)
        x = self.bn(x)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]