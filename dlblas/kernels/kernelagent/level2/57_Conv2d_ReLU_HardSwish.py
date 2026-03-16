import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _relu_hswish_inplace_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Hint for better vectorization on contiguous ranges
    tl.max_contiguous(offsets, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Fused ReLU + HardSwish:
    # r = max(x, 0)
    # y = r * clamp((r + 3)/6, 0, 1)
    # For r >= 0, clamp reduces to min((r + 3)/6, 1) = min(r + 3, 6) * (1/6)
    r = tl.maximum(x, 0.0)
    inv6 = 1.0 / 6.0
    y = r * tl.minimum(r + 3.0, 6.0) * inv6

    tl.store(x_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies ReLU, and applies HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        # Fused ReLU + HardSwish on CUDA using Triton for speed
        if x.is_cuda:
            x = x.contiguous()
            n_elements = x.numel()
            # Heuristic tuning for H200: smaller tiles with fewer warps for mid-size tensors,
            # larger tiles for very large tensors to reduce launch overhead.
            if n_elements >= (1 << 22):
                BLOCK_SIZE = 8192
                num_warps = 8
                num_stages = 2
            else:
                BLOCK_SIZE = 4096
                num_warps = 4
                num_stages = 1
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            _relu_hswish_inplace_kernel[grid](x, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=num_stages)
            return x
        else:
            # CPU fallback preserving exact semantics
            x = torch.relu(x)
            x = x * torch.clamp((x + 3) / 6, 0, 1)
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