import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _softmax_pool2_fused_kernel(
    x_ptr, y_ptr,
    x_stride_n, x_stride_c, x_stride_d, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_d, y_stride_h, y_stride_w,
    N, C, D, H, W, OD, OH, OW,
    K: tl.constexpr,                 # fused kernel size (K1 * K1)
    BLOCK_C: tl.constexpr,           # channel tile (>= C, padded to pow2)
    BLOCK_OW: tl.constexpr,          # width tile
):
    # Grid:
    #  axis 0 => over (N * OD * OH)
    #  axis 1 => tiles along OW
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    oh = pid0 % OH
    t = pid0 // OH
    od = t % OD
    n = t // OD

    ow_start = pid1 * BLOCK_OW
    offs_ow = tl.arange(0, BLOCK_OW)
    ow = ow_start + offs_ow
    mask_ow = ow < OW

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # Accumulator for the pooled result over the fused KxKxK window
    acc = tl.full([BLOCK_C, BLOCK_OW], -float("inf"), dtype=tl.float32)

    # Starting coordinates in the input for this output tile
    in_d0 = od * K
    in_h0 = oh * K
    in_w0_tile = ow_start * K

    # Base pointer for the (n, in_d0, in_h0) plane
    plane_base = n * x_stride_n + in_d0 * x_stride_d + in_h0 * x_stride_h

    # Iterate over fused K x K x K window
    for kd in range(0, K):
        for kh in range(0, K):
            # Base for this kd, kh slice
            slice_base = plane_base + kd * x_stride_d + kh * x_stride_h + in_w0_tile * x_stride_w
            # For width, each output ow pulls from input w = ow*K + kw
            ow_offsets = offs_ow * K  # [BLOCK_OW]
            for kw in range(0, K):
                # Build 2D pointers [BLOCK_C, BLOCK_OW]
                ptrs = (x_ptr
                        + slice_base
                        + ow_offsets[None, :] * x_stride_w
                        + kw * x_stride_w
                        + offs_c[:, None] * x_stride_c)
                m2d = mask_c[:, None] & mask_ow[None, :]
                x = tl.load(ptrs, mask=m2d, other=-float("inf")).to(tl.float32)
                # Channel-wise softmax for each spatial position in the tile
                x_max = tl.max(x, axis=0)                          # [BLOCK_OW]
                x = tl.exp(x - x_max[None, :])                     # [BLOCK_C, BLOCK_OW]
                x_sum = tl.sum(x, axis=0)                          # [BLOCK_OW]
                y_tile = x / x_sum[None, :]                        # [BLOCK_C, BLOCK_OW]
                # Max-pool over the fused window
                acc = tl.maximum(acc, y_tile)

    # Store results
    out_ptrs = (y_ptr
                + n * y_stride_n
                + od * y_stride_d
                + oh * y_stride_h
                + ow[None, :] * y_stride_w
                + offs_c[:, None] * y_stride_c)
    tl.store(out_ptrs, acc, mask=(mask_c[:, None] & mask_ow[None, :]))


def _softmax_then_two_pools_fused_triton(x: torch.Tensor, pool_kernel_size: int) -> torch.Tensor:
    # x: [N, C, D, H, W], compute softmax along C then two MaxPool3d(K) (stride=K) fused into one with Kf = K*K (stride=K*K).
    assert x.is_cuda and x.ndim == 5
    x = x.contiguous()
    N, C, D, H, W = x.shape
    K1 = int(pool_kernel_size)
    Kf = K1 * K1

    # Output dims for single pool with kernel=Kf, stride=Kf, padding=0 (ceil_mode=False)
    def odim(L, K):
        if L < K:
            return 0
        return (L - K) // K + 1

    OD = odim(D, Kf)
    OH = odim(H, Kf)
    OW = odim(W, Kf)

    y = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)
    if OD == 0 or OH == 0 or OW == 0:
        return y

    xs = x.stride()
    ys = y.stride()

    # Tile configuration
    BLOCK_OW = min(128, 1 << (OW - 1).bit_length())
    BLOCK_C = min(128, 1 << (C - 1).bit_length())

    grid = (N * OD * OH, triton.cdiv(OW, BLOCK_OW))
    _softmax_pool2_fused_kernel[grid](
        x, y,
        xs[0], xs[1], xs[2], xs[3], xs[4],
        ys[0], ys[1], ys[2], ys[3], ys[4],
        N, C, D, H, W, OD, OH, OW,
        K=Kf,
        BLOCK_C=BLOCK_C,
        BLOCK_OW=BLOCK_OW,
        num_warps=4,
        num_stages=4,
    )
    return y


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Softmax, and performs two max pooling operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # Keep original modules for CPU path
        self.pool1 = nn.MaxPool3d(pool_kernel_size)
        self.pool2 = nn.MaxPool3d(pool_kernel_size)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width') where depth', height', width' are the dimensions after pooling.
        """
        x = self.conv(x)
        if x.is_cuda:
            # Fully fuse softmax (along channel) with two successive maxpools into one Triton kernel
            x = _softmax_then_two_pools_fused_triton(x, self.pool_kernel_size)
        else:
            x = torch.softmax(x, dim=1)
            x = self.pool1(x)
            x = self.pool2(x)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]