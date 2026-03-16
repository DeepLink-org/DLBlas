import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fill_const_kernel(out_ptr, value, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Hints for better vectorization/coalescing
    tl.multiple_of(offsets, 16)
    tl.max_contiguous(offsets, 16)
    # Store the scalar constant; Triton will broadcast and cast to dst dtype
    tl.store(out_ptr + offsets, value, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized model:
    After mean over channels with keepdim=True, the channel dimension is 1.
    Softmax along a singleton channel dimension is identically 1, so the final
    output is a constant tensor tanh(1) * scaling_factor with the spatial shape
    produced by ConvTranspose3d. We compute that shape analytically and fill it
    using a fast Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

        # Cache conv attributes as 3-tuples to avoid per-forward branching
        def _to3(x):
            return (x, x, x) if isinstance(x, int) else x
        self._k = _to3(self.conv_transpose.kernel_size)
        self._s = _to3(self.conv_transpose.stride)
        self._p = _to3(self.conv_transpose.padding)
        self._d = _to3(self.conv_transpose.dilation)
        self._op = _to3(self.conv_transpose.output_padding)

        # Precompute constant result
        self._const_val = math.tanh(1.0) * float(self.scaling_factor)

    def forward(self, x):
        # Compute ConvTranspose3d output spatial size
        N, _, Di, Hi, Wi = x.shape
        k, s, p, d, op = self._k, self._s, self._p, self._d, self._op
        Do = (Di - 1) * s[0] - 2 * p[0] + d[0] * (k[0] - 1) + op[0] + 1
        Ho = (Hi - 1) * s[1] - 2 * p[1] + d[1] * (k[1] - 1) + op[1] + 1
        Wo = (Wi - 1) * s[2] - 2 * p[2] + d[2] * (k[2] - 1) + op[2] + 1

        out = torch.empty((N, 1, Do, Ho, Wo), device=x.device, dtype=x.dtype)

        const_val = self._const_val
        n_elements = out.numel()

        # Choose larger blocks to reduce CTA count and launch overhead on H200
        if n_elements >= (1 << 22):      # >= 4,194,304
            BLOCK = 131072
            num_warps = 8
        elif n_elements >= (1 << 20):    # >= 1,048,576
            BLOCK = 65536
            num_warps = 8
        elif n_elements >= (1 << 18):    # >= 262,144
            BLOCK = 32768
            num_warps = 8
        elif n_elements >= (1 << 16):    # >= 65,536
            BLOCK = 16384
            num_warps = 4
        else:
            BLOCK = 4096
            num_warps = 4

        grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
        _fill_const_kernel[grid](out, const_val, n_elements, BLOCK_SIZE=BLOCK, num_warps=num_warps, num_stages=1)
        return out


batch_size = 16
in_channels = 8
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (1, 1, 1, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape, scaling_factor]