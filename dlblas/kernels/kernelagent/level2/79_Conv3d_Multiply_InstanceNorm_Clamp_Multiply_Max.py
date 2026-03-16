import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _compute_mu_rstd_kernel(
    x_ptr,         # *f32, input tensor after conv, shape [N*C*S]
    m_ptr,         # *f32, multiplier, shape [C]
    mu_ptr,        # *f32, output mean, shape [N*C]
    rstd_ptr,      # *f32, output rstd, shape [N*C]
    S,             # int32, number of spatial elements per (N, C)
    C,             # int32, number of channels
    eps,           # f32, epsilon for numerical stability
    BLOCK_S: tl.constexpr,  # tile over spatial dimension
):
    pid = tl.program_id(axis=0)  # range: [0, N*C)
    n = pid // C
    c = pid % C

    base_nc = n * C + c
    x_base = x_ptr + base_nc * S

    # Load multiplier for this channel
    m = tl.load(m_ptr + c)

    offs = tl.arange(0, BLOCK_S)
    acc_sum = tl.zeros((), dtype=tl.float32)
    acc_sq = tl.zeros((), dtype=tl.float32)

    s = 0
    while s < S:
        idx = s + offs
        mask = idx < S
        v = tl.load(x_base + idx, mask=mask, other=0.0).to(tl.float32)
        v = v * m
        # accumulate scalars to reduce register pressure
        vsum = tl.sum(tl.where(mask, v, 0.0), axis=0)
        vsqsum = tl.sum(tl.where(mask, v * v, 0.0), axis=0)
        acc_sum += vsum
        acc_sq += vsqsum
        s += BLOCK_S

    S_f = tl.full((), S, dtype=tl.float32)
    mean = acc_sum / S_f
    var = tl.maximum(0.0, acc_sq / S_f - mean * mean)
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mu_ptr + base_nc, mean)
    tl.store(rstd_ptr + base_nc, rstd)


@triton.jit
def _postprocess_and_reduce_max_kernel(
    x_ptr,         # *f32, input after conv, shape [N*C*S]
    m_ptr,         # *f32, multiplier, shape [C]
    mu_ptr,        # *f32, mean per (N,C), shape [N*C]
    rstd_ptr,      # *f32, rstd per (N,C), shape [N*C]
    out_ptr,       # *f32, output max over C, shape [N*S]
    S,             # int32
    C,             # int32
    clamp_min,     # f32
    clamp_max,     # f32
    BLOCK_S: tl.constexpr,  # tile over spatial
    BLOCK_C: tl.constexpr,  # tile over channels
):
    pid_n = tl.program_id(axis=0)  # [0, N)
    pid_sb = tl.program_id(axis=1)  # [0, ceil_div(S, BLOCK_S))

    s_offs = pid_sb * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    # Initialize maxima for the spatial tile
    neg_inf = tl.full([BLOCK_S], -1e30, dtype=tl.float32)
    max_vals = neg_inf

    base_nC = pid_n * C
    base_nS = pid_n * S

    c_start = 0
    while c_start < C:
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C

        # Load per-channel stats and multiplier
        m_vec = tl.load(m_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        mu_vec = tl.load(mu_ptr + base_nC + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        rstd_vec = tl.load(rstd_ptr + base_nC + c_offs, mask=c_mask, other=0.0).to(tl.float32)

        # Build 2D pointers for [Ctile, Stile]
        nc_idx = (base_nC + c_offs)[:, None]  # [Ctile, 1]
        ptrs = x_ptr + nc_idx * S + s_offs[None, :]  # [Ctile, Stile]
        mask2d = c_mask[:, None] & s_mask[None, :]

        x_tile = tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32)
        # First multiplication
        y1 = x_tile * m_vec[:, None]
        # Instance norm: (y1 - mu) * rstd
        normed = (y1 - mu_vec[:, None]) * rstd_vec[:, None]
        # Clamp
        normed = tl.maximum(normed, clamp_min)
        normed = tl.minimum(normed, clamp_max)
        # Second multiplication
        y2 = normed * m_vec[:, None]

        # Ensure masked lanes don't affect reduction
        y2 = tl.where(mask2d, y2, -1e30)

        # Reduce over channel tile
        cmax = tl.max(y2, axis=0)  # [Stile]
        max_vals = tl.maximum(max_vals, cmax)

        c_start += BLOCK_C

    # Store results
    tl.store(out_ptr + base_nS + s_offs, max_vals, mask=s_mask)


class ModelNew(nn.Module):
    """
    A 3D convolutional layer followed by multiplication, instance normalization, clamping, multiplication, and a max operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        # Conv3d
        x = self.conv(x)

        # Fallback to reference PyTorch ops if not CUDA
        if not x.is_cuda:
            x = x * self.multiplier
            x = self.instance_norm(x)
            x = torch.clamp(x, self.clamp_min, self.clamp_max)
            x = x * self.multiplier
            x = torch.max(x, dim=1)[0]
            return x

        # Triton fused path:
        # Shapes
        N, C, D, H, W = x.shape
        S = D * H * W

        # Ensure contiguous for predictable indexing
        x = x.contiguous()

        # Flatten multiplier to [C]
        m = self.multiplier.view(C).contiguous()

        # Allocate stats buffers
        mu = torch.empty((N, C), device=x.device, dtype=x.dtype)
        rstd = torch.empty((N, C), device=x.device, dtype=x.dtype)

        # Kernel 1: compute mean and rstd over spatial dims for (x * multiplier)
        grid_mu = (N * C,)
        BLOCK_S1 = 2048
        _compute_mu_rstd_kernel[grid_mu](
            x, m, mu, rstd,
            S, C, self.instance_norm.eps,
            BLOCK_S=BLOCK_S1,
            num_warps=8,
            num_stages=4,
        )

        # Kernel 2: normalize, clamp, second multiply, and reduce max over channels
        out = torch.empty((N, S), device=x.device, dtype=x.dtype)
        BLOCK_S2 = 512
        BLOCK_C2 = 64
        grid_reduce = (N, triton.cdiv(S, BLOCK_S2))
        _postprocess_and_reduce_max_kernel[grid_reduce](
            x, m, mu, rstd, out,
            S, C, float(self.clamp_min), float(self.clamp_max),
            BLOCK_S=BLOCK_S2, BLOCK_C=BLOCK_C2,
            num_warps=8,
            num_stages=4,
        )

        # Reshape to [N, D, H, W]
        out = out.view(N, D, H, W)
        return out


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
multiplier_shape = (out_channels, 1, 1, 1)
clamp_min = -1.0
clamp_max = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max]