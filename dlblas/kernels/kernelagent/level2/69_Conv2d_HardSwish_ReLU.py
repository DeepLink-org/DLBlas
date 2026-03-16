import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch.nn.functional as F


@triton.jit
def _hswish_relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 1D launch over the flattened tensor
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Provide vectorization hints for better memory coalescing
    tl.multiple_of(offsets, 16)
    tl.max_contiguous(offsets, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute ReLU(HardSwish(x)) with minimal ops:
    # rx = max(x, 0)
    # r  = min(rx/6 + 0.5, 1)
    # y  = rx * r
    rx = tl.maximum(x, 0.0)
    y = rx * tl.minimum(rx * (1.0 / 6.0) + 0.5, 1.0)

    tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, applies HardSwish, and then ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def _fused_hardswish_relu_triton(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton only when on CUDA and autograd not required (ensure safety)
        if (not x.is_cuda) or x.requires_grad:
            return F.relu(F.hardswish(x))

        x_in = x.contiguous()
        n_elements = x_in.numel()
        if n_elements == 0:
            return x_in

        # Heuristic tuning for H200 and this elementwise workload
        if n_elements >= (1 << 20):
            BLOCK_SIZE = 8192
            num_warps = 8
        elif n_elements >= (1 << 18):
            BLOCK_SIZE = 4096
            num_warps = 8
        else:
            BLOCK_SIZE = 2048
            num_warps = 4

        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
        # In-place to reduce memory traffic and allocation
        _hswish_relu_kernel[grid](x_in, x_in, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=1)
        return x_in

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height - kernel_size + 1, width - kernel_size + 1).
        """
        x = self.conv(x)
        # Fused HardSwish + ReLU using Triton for speed; safe CPU/grad fallback
        x = self._fused_hardswish_relu_triton(x)
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