import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


@triton.jit
def _min_reduce_dim2_kernel(
    x_ptr,                # *f32 [B, C, D, H, W]
    y_ptr,                # *f32 [B, C, H, W]
    B, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_W: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Grid maps over (n, c, h, w_tile)
    pid = tl.program_id(axis=0)
    num_w_tiles = tl.cdiv(W, BLOCK_W)

    w_tile = pid % num_w_tiles
    pid = pid // num_w_tiles
    h_idx = pid % H
    pid = pid // H
    c_idx = pid % C
    n_idx = pid // C

    w_offsets = w_tile * BLOCK_W + tl.arange(0, BLOCK_W)
    d_offsets = tl.arange(0, BLOCK_D)

    offs_w = w_offsets[None, :]
    offs_d = d_offsets[:, None]

    in_base = n_idx * stride_n + c_idx * stride_c + h_idx * stride_h
    ptrs = x_ptr + in_base + offs_d * stride_d + offs_w * stride_w

    mask = (w_offsets[None, :] < W) & (d_offsets[:, None] < D)
    vals = tl.load(ptrs, mask=mask, other=float("inf"))
    # Compute min across D dimension (axis=0)
    min_vals = tl.min(vals, axis=0)

    out_ptrs = y_ptr + n_idx * out_stride_n + c_idx * out_stride_c + h_idx * out_stride_h + w_offsets * out_stride_w
    tl.store(out_ptrs, min_vals, mask=w_offsets < W)


@triton.jit
def _softmax_dim1_kernel(
    x_ptr,                # *f32 [B, C, H, W]
    y_ptr,                # *f32 [B, C, H, W]
    B, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_C: tl.constexpr,
):
    # Grid maps over (n, h, w)
    pid = tl.program_id(axis=0)
    w_idx = pid % W
    pid = pid // W
    h_idx = pid % H
    n_idx = pid // H

    c_offsets = tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    base = n_idx * stride_n + h_idx * stride_h + w_idx * stride_w
    x_ptrs = x_ptr + base + c_offsets * stride_c

    x_vals = tl.load(x_ptrs, mask=c_mask, other=-float("inf"))
    x_vals = x_vals.to(tl.float32)
    x_max = tl.max(x_vals, axis=0)
    x_vals = x_vals - x_max
    x_exp = tl.exp(x_vals)
    x_sum = tl.sum(x_exp, axis=0)
    y_vals = x_exp / x_sum

    y_ptrs = y_ptr + base + c_offsets * out_stride_c
    tl.store(y_ptrs, y_vals, mask=c_mask)


@triton.jit
def _fused_minD_softmaxC_wtile_singleCTile(
    x_ptr,                # *f32 [B, C, D, H, W]
    y_ptr,                # *f32 [B, C, H, W]
    B, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    TOT_D_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Grid maps over (n, h, w_tile)
    pid = tl.program_id(axis=0)
    num_w_tiles = tl.cdiv(W, BLOCK_W)

    w_tile = pid % num_w_tiles
    pid = pid // num_w_tiles
    h_idx = pid % H
    n_idx = pid // H

    # Offsets
    c_offsets = tl.arange(0, BLOCK_C)
    w_offsets = w_tile * BLOCK_W + tl.arange(0, BLOCK_W)

    c_mask = c_offsets < C
    w_mask = w_offsets < W

    # Prepare run_min over D for each (c, w) in the tile
    pos_inf = float("inf")
    neg_inf = -float("inf")
    run_min = tl.full([BLOCK_C, BLOCK_W], pos_inf, dtype=tl.float32)

    in_base = n_idx * stride_n + h_idx * stride_h

    # Iterate over D tiles
    for dt in tl.static_range(0, TOT_D_TILES):
        d_offsets = dt * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D

        ptrs = (
            x_ptr
            + in_base
            + c_offsets[:, None, None] * stride_c
            + d_offsets[None, :, None] * stride_d
            + w_offsets[None, None, :] * stride_w
        )
        mask3d = c_mask[:, None, None] & d_mask[None, :, None] & w_mask[None, None, :]
        vals = tl.load(ptrs, mask=mask3d, other=pos_inf)
        tile_min = tl.min(vals, axis=1)  # reduce along D
        run_min = tl.minimum(run_min, tile_min)

    # Softmax along channel dimension per (w) with numerical stability
    gmax = tl.max(tl.where(c_mask[:, None], run_min, neg_inf), axis=0)
    exps = tl.exp(run_min - gmax[None, :])
    exps = tl.where(c_mask[:, None] & w_mask[None, :], exps, 0.0)
    gsum = tl.sum(exps, axis=0)
    out_vals = exps / gsum[None, :]

    # Store results to [B, C, H, W]
    out_ptrs = (
        y_ptr
        + n_idx * out_stride_n
        + c_offsets[:, None] * out_stride_c
        + h_idx * out_stride_h
        + w_offsets[None, :] * out_stride_w
    )
    tl.store(out_ptrs, out_vals, mask=c_mask[:, None] & w_mask[None, :])


def _reduce_min_dim2_triton(x: torch.Tensor) -> torch.Tensor:
    # x: [B, C, D, H, W]
    B, C, D, H, W = x.shape
    y = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)

    stride_n, stride_c, stride_d, stride_h, stride_w = x.stride()
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = y.stride()

    BLOCK_W = 32 if W >= 32 else _next_pow2(W)
    BLOCK_D = _next_pow2(D)
    grid = (triton.cdiv(W, BLOCK_W) * H * C * B,)

    _min_reduce_dim2_kernel[grid](
        x, y,
        B, C, D, H, W,
        stride_n, stride_c, stride_d, stride_h, stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        BLOCK_W=BLOCK_W, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2
    )
    return y


def _softmax_dim1_triton(x: torch.Tensor) -> torch.Tensor:
    # x: [B, C, H, W]
    B, C, H, W = x.shape
    y = torch.empty_like(x)

    stride_n, stride_c, stride_h, stride_w = x.stride()
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = y.stride()

    BLOCK_C = _next_pow2(C)
    grid = (B * H * W,)

    _softmax_dim1_kernel[grid](
        x, y,
        B, C, H, W,
        stride_n, stride_c, stride_h, stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        BLOCK_C=BLOCK_C,
        num_warps=4, num_stages=2
    )
    return y


def _fused_single_ctile_triton(x: torch.Tensor) -> torch.Tensor:
    # x: [B, C, D, H, W] contiguous NCDHW
    B, C, D, H, W = x.shape
    y = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)

    sN, sC, sD, sH, sW = x.stride()
    oN, oC, oH, oW = y.stride()

    BLOCK_W = 32 if W >= 32 else _next_pow2(W)
    BLOCK_D = _next_pow2(D)
    BLOCK_C = _next_pow2(C)
    tot_d_tiles = (D + BLOCK_D - 1) // BLOCK_D

    grid = (triton.cdiv(W, BLOCK_W) * H * B,)

    _fused_minD_softmaxC_wtile_singleCTile[grid](
        x, y,
        B, C, D, H, W,
        sN, sC, sD, sH, sW,
        oN, oC, oH, oW,
        TOT_D_TILES=tot_d_tiles,
        BLOCK_C=BLOCK_C,
        BLOCK_D=BLOCK_D,
        BLOCK_W=BLOCK_W,
        num_warps=8,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    """
    Simple model that performs a 3D convolution, applies minimum operation along a specific dimension, 
    and then applies softmax.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W)
        """
        x = self.conv(x)
        # Fast Triton path for common case: reduce along depth (dim==2) then softmax over channels (dim==1)
        if x.is_cuda and self.dim == 2:
            x = x.contiguous()
            B, C, D, H, W = x.shape
            # Use fused single-C-tile kernel when C fits in one tile for maximum efficiency
            if _next_pow2(C) <= 64:
                return _fused_single_ctile_triton(x)
            else:
                # General fallback: two-step Triton (reduce-min over D, then softmax over C)
                y_min = _reduce_min_dim2_triton(x)
                y = _softmax_dim1_triton(y_min)
                return y
        else:
            x = torch.min(x, dim=self.dim)[0]  # Apply minimum along the specified dimension
            x = torch.softmax(x, dim=1)  # Apply softmax along the channel dimension
            return x


batch_size = 128
in_channels = 3
out_channels = 16
D, H, W = 16, 32, 32
kernel_size = 3
dim = 2  # Dimension along which to apply minimum operation (e.g., depth)

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]