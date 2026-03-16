import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _swish_reduce_3d(
    x_ptr,                 # *f32 [N, C, D, H, W]
    sum_ptr,               # *f32 [N * G * D]
    sumsq_ptr,             # *f32 [N * G * D]
    N: tl.constexpr,       # int
    C: tl.constexpr,       # int
    D, H, W,               # int (runtime)
    strideN, strideC, strideD, strideH, strideW,  # int strides
    group_size,            # int
    num_groups,            # int
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Program ids
    pid0 = tl.program_id(0)  # over N*C*D
    pid1 = tl.program_id(1)  # over H tiles
    pid2 = tl.program_id(2)  # over W tiles

    # Decode (n, c, d)
    CD = C * D
    n = pid0 // CD
    tmp = pid0 % CD
    c = tmp // D
    d = tmp % D

    # Tile origins
    h_start = pid1 * BLOCK_H
    w_start = pid2 * BLOCK_W

    # Indices within tile
    h_idx = h_start + tl.arange(0, BLOCK_H)[:, None]
    w_idx = w_start + tl.arange(0, BLOCK_W)[None, :]
    mask = (h_idx < H) & (w_idx < W)

    # Offsets for the tile
    offs = (
        n * strideN
        + c * strideC
        + d * strideD
        + h_idx * strideH
        + w_idx * strideW
    )

    # Load and compute Swish
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.sigmoid(x) * x  # Swish

    # Partial reductions over tile
    tile_sum_h = tl.sum(s, axis=1)
    tile_sum = tl.sum(tile_sum_h, axis=0)
    ssq = s * s
    tile_sumsq_h = tl.sum(ssq, axis=1)
    tile_sumsq = tl.sum(tile_sumsq_h, axis=0)

    # Accumulate per-(n, g, d) to reduce atomic contention
    g = c // group_size
    idx = (n * num_groups + g) * D + d
    tl.atomic_add(sum_ptr + idx, tile_sum)
    tl.atomic_add(sumsq_ptr + idx, tile_sumsq)


@triton.jit
def _apply_gn_hswish_3d(
    x_ptr,                 # *f32 [N, C, D, H, W]
    mean_ptr,              # *f32 [N * G]
    invstd_ptr,            # *f32 [N * G]
    weight_ptr,            # *f32 [C]
    bias_ptr,              # *f32 [C]
    y_ptr,                 # *f32 [N, C, D, H, W]
    N: tl.constexpr,       # int
    C: tl.constexpr,       # int
    D, H, W,               # int
    strideN, strideC, strideD, strideH, strideW,  # int strides
    group_size,            # int
    num_groups,            # int
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid0 = tl.program_id(0)  # over N*C*D
    pid1 = tl.program_id(1)  # over H tiles
    pid2 = tl.program_id(2)  # over W tiles

    CD = C * D
    n = pid0 // CD
    tmp = pid0 % CD
    c = tmp // D
    d = tmp % D

    h_start = pid1 * BLOCK_H
    w_start = pid2 * BLOCK_W

    h_idx = h_start + tl.arange(0, BLOCK_H)[:, None]
    w_idx = w_start + tl.arange(0, BLOCK_W)[None, :]
    mask = (h_idx < H) & (w_idx < W)

    offs = (
        n * strideN
        + c * strideC
        + d * strideD
        + h_idx * strideH
        + w_idx * strideW
    )

    # Load input and compute Swish again (no large intermediate buffer)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.sigmoid(x) * x  # Swish

    g = c // group_size
    stat_idx = n * num_groups + g
    mu = tl.load(mean_ptr + stat_idx)
    invstd = tl.load(invstd_ptr + stat_idx)
    gamma = tl.load(weight_ptr + c)
    beta = tl.load(bias_ptr + c)

    # GroupNorm affine: ((s - mu) * invstd) * gamma + beta
    v = ((s - mu) * invstd) * gamma + beta

    # HardSwish: v * clamp(v + 3, 0, 6) / 6
    vp3 = v + 3.0
    clamp6 = tl.minimum(tl.maximum(vp3, 0.0), 6.0)
    hs = v * clamp6 * (1.0 / 6.0)

    tl.store(y_ptr + offs, hs, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies Swish activation, 
    group normalization, and then HardSwish activation.
    Uses Triton kernels to fuse Swish + GroupNorm + HardSwish for speed.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)

    def forward(self, x):
        y = self.conv_transpose(x)
        # Use Triton fused path on CUDA; fallback to PyTorch otherwise
        if y.is_cuda:
            N, C, D, H, W = y.shape
            device = y.device
            dtype = y.dtype

            num_groups = self.group_norm.num_groups
            group_size = C // num_groups
            eps = self.group_norm.eps

            # Strides (in elements)
            sN, sC, sD, sH, sW = y.stride()

            # Accumulators for per-(N, group, D) statistics to reduce atomics contention
            sums = torch.zeros(N * num_groups * D, device=device, dtype=torch.float32)
            sumsq = torch.zeros(N * num_groups * D, device=device, dtype=torch.float32)

            # Launch kernel to compute Swish partial reductions
            # Larger tiles reduce number of atomics per (n,c,d)
            BLOCK_H, BLOCK_W = 16, 64
            grid = (N * C * D, triton.cdiv(H, BLOCK_H), triton.cdiv(W, BLOCK_W))
            _swish_reduce_3d[grid](
                y, sums, sumsq,
                N, C, D, H, W,
                sN, sC, sD, sH, sW,
                group_size, num_groups,
                BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
                num_warps=4, num_stages=2,
            )

            # Reduce across depth on device to get per-(N, group) stats
            sums = sums.view(N, num_groups, D).sum(dim=2).contiguous()
            sumsq = sumsq.view(N, num_groups, D).sum(dim=2).contiguous()

            # Compute mean/var and invstd on device
            M = float(group_size * D * H * W)
            means = (sums / M).contiguous()
            vars_ = (sumsq / M - means * means).clamp_min(0.0)
            invstd = torch.rsqrt(vars_ + eps).contiguous()

            # Prepare affine parameters
            weight = self.group_norm.weight.to(device=device, dtype=torch.float32, non_blocking=True)
            bias = self.group_norm.bias.to(device=device, dtype=torch.float32, non_blocking=True)

            out = torch.empty_like(y)

            _apply_gn_hswish_3d[grid](
                y, means.view(-1), invstd.view(-1), weight, bias, out,
                N, C, D, H, W,
                sN, sC, sD, sH, sW,
                group_size, num_groups,
                BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
                num_warps=4, num_stages=2,
            )
            return out.to(dtype)
        else:
            y = torch.sigmoid(y) * y  # Swish activation
            y = self.group_norm(y)
            y = torch.nn.functional.hardswish(y)  # HardSwish activation
            return y


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
groups = 4
eps = 1e-5

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, eps]