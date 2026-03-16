import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_bias_scale_clamp_inplace(
    in_out_ptr,     # *float*, tensor after conv_transpose, will be updated in-place
    bias_ptr,       # *float*, bias of shape [C]
    s,              # *float*, scaling factor
    n_elements,     # total number of elements = N * C * H * W
    C,              # number of channels
    HW,             # H * W
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + arange
    mask = offsets < n_elements

    # Stream the large tensor via L2 to keep L1 available for the tiny bias array
    x = tl.load(in_out_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")

    # With BLOCK_SIZE == HW (set by the launcher), each program handles one (n, c) plane.
    plane_idx = block_start // HW
    ch = plane_idx % C
    b_scalar = tl.load(bias_ptr + ch)
    y = x + b_scalar

    # clamp to [0, 1], scale, clamp again, then divide by scale
    y = tl.maximum(y, 0.0)
    y = tl.minimum(y, 1.0)
    y = y * s
    y = tl.maximum(y, 0.0)
    y = tl.minimum(y, 1.0)
    y = y / s

    tl.store(in_out_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    The post-conv elementwise ops are fused into a single Triton kernel for improved performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        y = self.conv_transpose(x)
        s = float(self.scaling_factor)

        # Use Triton for the fused elementwise ops if on CUDA; otherwise, fall back to PyTorch ops.
        if y.is_cuda:
            y = y.contiguous()
            # Ensure bias dtype matches output dtype; flatten to [C]
            bias = self.bias.to(dtype=y.dtype).contiguous().view(-1)

            N, C, H, W = y.shape
            n_elements = y.numel()
            HW = H * W

            # Launch one program per (N, C) plane; BLOCK_SIZE == HW ensures single bias load per block.
            BLOCK_SIZE = HW
            grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
            _fused_bias_scale_clamp_inplace[grid](
                y, bias, s, n_elements, C, HW,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=8,
                num_stages=3,
            )
            return y
        else:
            y = y + self.bias
            y = torch.clamp(y, min=0.0, max=1.0)
            y = y * s
            y = torch.clamp(y, min=0.0, max=1.0)
            y = y / s
            return y


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]