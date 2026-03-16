import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _relu_groupnorm_kernel(
    x_ptr,           # *T
    y_ptr,           # *T
    w_ptr,           # *fp32
    b_ptr,           # *fp32
    N, C, D, H, W,   # int32
    G,               # int32
    eps,             # fp32
    sC,              # int32 = D*H*W
    sN,              # int32 = C*sC
    GROUP_ELEMS,     # int32 = (C//G) * sC
    NUM_TILES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_channels = C // G
    c_start = g * group_channels
    base_group = n * sN + c_start * sC
    group_elems = GROUP_ELEMS

    offs = tl.arange(0, BLOCK_SIZE)

    # First pass: compute mean and var of ReLU(x) over the group (double-buffered prefetch)
    sum1 = tl.zeros([], dtype=tl.float32)
    sum2 = tl.zeros([], dtype=tl.float32)
    if NUM_TILES == 1:
        idx0 = offs
        mask0 = idx0 < group_elems
        x0 = tl.load(x_ptr + base_group + idx0, mask=mask0, other=0.0)
        z0 = tl.maximum(x0.to(tl.float32), 0.0)
        sum1 += tl.sum(z0, axis=0)
        sum2 += tl.sum(z0 * z0, axis=0)
    else:
        idx0 = offs
        mask0 = idx0 < group_elems
        x0 = tl.load(x_ptr + base_group + idx0, mask=mask0, other=0.0)
        for t in tl.static_range(1, NUM_TILES):
            idx1 = t * BLOCK_SIZE + offs
            mask1 = idx1 < group_elems
            x1 = tl.load(x_ptr + base_group + idx1, mask=mask1, other=0.0)

            z0 = tl.maximum(x0.to(tl.float32), 0.0)
            sum1 += tl.sum(z0, axis=0)
            sum2 += tl.sum(z0 * z0, axis=0)

            x0 = x1
            idx0 = idx1
            mask0 = mask1

        # tail
        z0 = tl.maximum(x0.to(tl.float32), 0.0)
        sum1 += tl.sum(z0, axis=0)
        sum2 += tl.sum(z0 * z0, axis=0)

    denom = tl.full([], group_elems, dtype=tl.float32)
    mean = sum1 / denom
    var = sum2 / denom - mean * mean
    inv_std = tl.rsqrt(var + eps)

    # Second pass: iterate per-channel to minimize gamma/beta global loads
    ci = 0
    while ci < group_channels:
        ch = c_start + ci
        gamma = tl.load(w_ptr + ch).to(tl.float32)
        beta = tl.load(b_ptr + ch).to(tl.float32)
        base_ch = n * sN + ch * sC

        off_c = 0
        while off_c < sC:
            idx = off_c + offs
            mask = idx < sC
            xr = tl.load(x_ptr + base_ch + idx, mask=mask, other=0.0)
            x_dtype = xr.dtype
            z = tl.maximum(xr.to(tl.float32), 0.0)
            z = (z - mean) * inv_std
            y = z * gamma + beta
            tl.store(y_ptr + base_ch + idx, y.to(x_dtype), mask=mask)
            off_c += BLOCK_SIZE
        ci += 1


def _fused_relu_groupnorm_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, groups: int, eps: float):
    # x: (N,C,D,H,W), weight/bias: (C,)
    assert x.is_cuda, "Triton kernel requires CUDA tensor"
    assert x.is_contiguous(), "Input must be contiguous NCDHW"
    N, C, D, H, W = x.shape
    sC = D * H * W
    sN = C * sC
    group_channels = C // groups
    group_elems = group_channels * sC

    y = torch.empty_like(x)

    def pick_block_size(n):
        # favor large tiles for fewer passes while keeping good occupancy
        for bs in (8192, 4096, 2048, 1024, 512, 256, 128, 64):
            if n >= bs:
                return bs
        return 64

    BLOCK_SIZE = pick_block_size(group_elems)
    NUM_TILES = (group_elems + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_warps = 8 if BLOCK_SIZE >= 2048 else 4

    # Ensure params are fp32 contiguous
    w = weight.to(torch.float32).contiguous()
    b = bias.to(torch.float32).contiguous()

    grid = (N * groups,)

    _relu_groupnorm_kernel[grid](
        x, y, w, b,
        N, C, D, H, W,
        groups,
        float(eps),
        sC,
        sN,
        group_elems,
        NUM_TILES=NUM_TILES,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=4,
    )
    return y


class ModelNew(nn.Module):
    """
    Model that performs a transposed 3D convolution, applies ReLU, and then applies group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        x = self.conv_transpose(x)
        # Use fused Triton kernel on CUDA for ReLU + GroupNorm to reduce memory traffic
        if x.is_cuda and x.is_contiguous() and (x.dtype in (torch.float16, torch.float32, torch.bfloat16)):
            weight = self.group_norm.weight
            bias = self.group_norm.bias
            x = _fused_relu_groupnorm_triton(x, weight, bias, self.group_norm.num_groups, self.group_norm.eps)
        else:
            # Fallback path preserving exact semantics
            x = self.relu(x)
            x = self.group_norm(x)
        return x


batch_size = 16
in_channels = 64
out_channels = 128
D, H, W = 8, 16, 16
kernel_size = 3
groups = 8
bias = False

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W, device="cuda")]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, bias]