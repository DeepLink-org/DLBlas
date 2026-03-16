import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _gelu_groupnorm_kernel(
    x_ptr,       # *f32
    w_ptr,       # *f32
    b_ptr,       # *f32
    y_ptr,       # *f32
    N, C, H, W,  # i32
    G,           # i32
    eps,         # f32
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # each program handles one (n, g) group
    n = pid // G
    g = pid % G

    HW = H * W
    cpg = C // G  # channels per group
    group_elems = cpg * HW  # total elements in the group

    # Base offsets in flattened NCHW memory (contiguous)
    c_start = g * cpg
    n_base = n * C * H * W
    group_base = n_base + c_start * H * W

    offs = tl.arange(0, BLOCK)
    inv_sqrt2 = 0.7071067811865476

    # First pass: compute mean and variance over GELU(x) for this (n, g)
    x_group_ptr = x_ptr + group_base

    # Accumulate per-lane across the whole group, then reduce once to scalars
    acc1 = tl.zeros([BLOCK], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK], dtype=tl.float32)

    idx = 0
    while idx < group_elems:
        i = idx + offs
        mask = i < group_elems

        # Load a contiguous slice of the group's data
        x = tl.load(x_group_ptr + i, mask=mask, other=0.0).to(tl.float32)

        # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        z = 0.5 * x * (1.0 + libdevice.erf(x * inv_sqrt2))

        # Accumulate per-lane
        z = tl.where(mask, z, 0.0)
        acc1 += z
        acc2 += z * z

        idx += BLOCK

    s1 = tl.sum(acc1, axis=0)
    s2 = tl.sum(acc2, axis=0)

    ge_f = tl.full((), group_elems, dtype=tl.float32)
    mean = s1 / ge_f
    var = s2 / ge_f - mean * mean
    rstd = tl.rsqrt(var + eps)

    # Second pass: normalize and apply affine
    y_group_ptr = y_ptr + group_base

    ch = 0
    while ch < cpg:
        ch_base = ch * HW

        # Load affine params once per channel to avoid redundant gathers
        gamma = tl.load(w_ptr + c_start + ch).to(tl.float32)
        beta = tl.load(b_ptr + c_start + ch).to(tl.float32)

        off = 0
        while off < HW:
            idx_hw = off + offs
            mask_hw = idx_hw < HW

            x = tl.load(x_group_ptr + ch_base + idx_hw, mask=mask_hw, other=0.0).to(tl.float32)
            z = 0.5 * x * (1.0 + libdevice.erf(x * inv_sqrt2))

            y = (z - mean) * rstd
            y = y * gamma + beta

            tl.store(y_group_ptr + ch_base + idx_hw, y, mask=mask_hw)
            off += BLOCK
        ch += 1


def gelu_groupnorm_fused(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, num_groups: int, eps: float):
    """
    Fused GELU followed by GroupNorm using a Triton kernel.
    Args:
        x: [N, C, H, W] float32 tensor (contiguous)
        weight: [C] float32 tensor
        bias: [C] float32 tensor
        num_groups: int
        eps: float
    Returns:
        y: [N, C, H, W] float32 tensor
    """
    assert x.is_cuda, "Triton kernel requires CUDA tensor"
    assert x.dtype == torch.float32, "This fused kernel expects float32 input"
    N, C, H, W = x.shape
    assert C % num_groups == 0, "num_groups must divide C"
    y = torch.empty_like(x)

    # Ensure parameters are on the same device and contiguous
    w = weight.contiguous()
    b = bias.contiguous()

    # Each program handles one (n, g)
    grid = (N * num_groups,)

    # Choose a tile size; 1024 is generally a good default
    BLOCK = 1024
    _gelu_groupnorm_kernel[grid](
        x, w, b, y,
        N, C, H, W,
        num_groups,
        eps,
        BLOCK=BLOCK,
        num_warps=8,
        num_stages=5,
    )
    return y


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, applies GELU, and normalizes with GroupNorm.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        if x.is_cuda:
            # Fused GELU + GroupNorm for performance on GPU
            x = gelu_groupnorm_fused(
                x,
                self.group_norm.weight,
                self.group_norm.bias,
                self.group_norm.num_groups,
                self.group_norm.eps,
            )
        else:
            # CPU fallback: preserve exact semantics
            x = torch.nn.functional.gelu(x)
            x = self.group_norm(x)
        return x


batch_size = 128
in_channels = 32
out_channels = 64
height, width = 32, 32
kernel_size = 4
stride = 2
groups = 8
num_groups = 8

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]