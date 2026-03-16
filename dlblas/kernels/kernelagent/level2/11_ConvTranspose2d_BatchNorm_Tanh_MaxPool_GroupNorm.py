import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _tanh_maxpool2x2_nchw_kernel(
    x_ptr,                   # *const T
    y_ptr,                   # *mut T
    NC: tl.constexpr,        # int, N*C combined
    H: tl.constexpr,         # int, input height
    W: tl.constexpr,         # int, input width
    H_OUT: tl.constexpr,     # int, output height (H//2)
    W_OUT: tl.constexpr,     # int, output width  (W//2)
    BLOCK_W: tl.constexpr,   # tile size along output width
):
    pid0 = tl.program_id(0)  # over (nc, oh)
    pid1 = tl.program_id(1)  # over tiles of ow

    nc = pid0 // H_OUT
    oh = pid0 % H_OUT

    ow = pid1 * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_ow = ow < W_OUT

    # Input coordinates for 2x2 pooling window
    iy0 = oh * 2
    ix0 = ow * 2

    # Base offset in flattened [NC, H, W] layout
    base = nc * H * W + iy0 * W + ix0

    # Compute offsets for the 4 elements in the pooling window
    off00 = base
    off01 = base + 1
    off10 = base + W
    off11 = base + W + 1

    # Load inputs with proper masking
    v00 = tl.load(x_ptr + off00, mask=mask_ow, other=0.0)
    v01 = tl.load(x_ptr + off01, mask=mask_ow, other=0.0)
    v10 = tl.load(x_ptr + off10, mask=mask_ow, other=0.0)
    v11 = tl.load(x_ptr + off11, mask=mask_ow, other=0.0)

    # MaxPool first (tanh is monotonic => tanh(max) == max(tanh))
    m0 = tl.maximum(v00, v01)
    m1 = tl.maximum(v10, v11)
    mp_in = tl.maximum(m0, m1)

    # Apply tanh once on the pooled value using fast libdevice implementation
    mp = libdevice.tanh(mp_in)

    # Output offsets in flattened [NC, H_OUT, W_OUT]
    y_off = nc * H_OUT * W_OUT + oh * W_OUT + ow
    tl.store(y_ptr + y_off, mp, mask=mask_ow)


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, batch normalization, tanh activation, max pooling, and group normalization.
    Tanh + MaxPool2d are fused into a single Triton kernel for improved performance on GPU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        # Keep modules for API compatibility; forward uses fused kernel for tanh+maxpool when on CUDA
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)

        # Fused tanh + maxpool using Triton when on CUDA; fallback to PyTorch otherwise
        if x.is_cuda:
            x = x.contiguous()
            N, C, H, W = x.shape
            # MaxPool2d(kernel=2, stride=2, padding=0, ceil_mode=False)
            H_OUT = H // 2
            W_OUT = W // 2
            y = torch.empty((N, C, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

            # Launch grid: one program per (nc, oh) row, tiled across W_OUT
            NC = N * C

            # Choose tile size and warps based on width to limit masked compute
            if W_OUT >= 128:
                BLOCK_W = 128
                num_warps = 4
            elif W_OUT >= 64:
                BLOCK_W = 64
                num_warps = 2
            else:
                BLOCK_W = 32
                num_warps = 1

            grid = (NC * H_OUT, triton.cdiv(W_OUT, BLOCK_W))
            _tanh_maxpool2x2_nchw_kernel[grid](
                x, y,
                NC=NC, H=H, W=W,
                H_OUT=H_OUT, W_OUT=W_OUT,
                BLOCK_W=BLOCK_W,
                num_warps=num_warps, num_stages=1
            )
            x = y
        else:
            # CPU or non-CUDA fallback ensures identical semantics
            x = self.tanh(x)
            x = self.max_pool(x)

        x = self.group_norm(x)
        return x


batch_size = 128
in_channels = 32
out_channels = 64
kernel_size = 4
stride = 2
padding = 1
groups = 8
num_groups = 4
height, width = 32, 32

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, num_groups]