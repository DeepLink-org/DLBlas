import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=8, num_stages=4),
    ],
    key=["group_elems"],
)
@triton.jit
def _groupnorm_fwd_kernel(
    x_ptr,         # * (N*C*HW) flattened as (N, C, HW)
    y_ptr,         # * (N*C*HW) float32 output
    gamma_ptr,     # * (C)
    beta_ptr,      # * (C)
    N,             # batch size
    C,             # channels
    HW,            # product of spatial dims
    groups,        # number of groups
    channels_per_group,    # C / groups
    group_elems,   # channels_per_group * HW
    eps,           # epsilon
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = pid // groups
    g = pid % groups

    c_start = g * channels_per_group
    base = (n * C + c_start) * HW

    offsets = tl.arange(0, BLOCK_SIZE)
    tl.multiple_of(offsets, 16)
    tl.max_contiguous(offsets, BLOCK_SIZE)

    # First pass: compute mean and variance using double-buffered prefetch
    sum1 = tl.zeros((), dtype=tl.float32)
    sum2 = tl.zeros((), dtype=tl.float32)

    idx0 = offsets
    mask0 = idx0 < group_elems
    x0 = tl.load(x_ptr + base + idx0, mask=mask0, other=0.0, eviction_policy='evict_last').to(tl.float32)

    cur = BLOCK_SIZE
    while cur < group_elems:
        idx1 = cur + offsets
        mask1 = idx1 < group_elems
        x1 = tl.load(x_ptr + base + idx1, mask=mask1, other=0.0, eviction_policy='evict_last').to(tl.float32)
        # accumulate current tile
        sum1 += tl.sum(x0, axis=0)
        sum2 += tl.sum(x0 * x0, axis=0)
        # rotate buffer
        x0 = x1
        cur += BLOCK_SIZE

    # accumulate last buffered tile
    sum1 += tl.sum(x0, axis=0)
    sum2 += tl.sum(x0 * x0, axis=0)

    ge = group_elems.to(tl.float32)
    mean = sum1 / ge
    var = sum2 / ge - mean * mean
    rstd = tl.rsqrt(var + eps)

    # Second pass: normalize and apply affine; reuse per-channel scalars
    ch = 0
    while ch < channels_per_group:
        ch_idx = c_start + ch
        gamma_k = tl.load(gamma_ptr + ch_idx).to(tl.float32)
        beta_k = tl.load(beta_ptr + ch_idx).to(tl.float32)
        # fuse affine with normalization to reduce ops
        scale = gamma_k * rstd
        bias_k = beta_k - mean * scale

        base_ch = base + ch * HW
        inner = 0
        while inner < HW:
            idx_hw = inner + offsets
            mask_hw = idx_hw < HW
            x = tl.load(x_ptr + base_ch + idx_hw, mask=mask_hw, other=0.0, eviction_policy='evict_last').to(tl.float32)
            y = x * scale + bias_k
            tl.store(y_ptr + base_ch + idx_hw, y, mask=mask_hw)
            inner += BLOCK_SIZE
        ch += 1


def group_norm_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, num_groups: int, eps: float):
    # Ensure contiguous layout and flatten spatial dims
    x_in = x
    x = x.contiguous()
    N, C = x.shape[:2]
    HW = x.numel() // (N * C)
    x_flat = x.view(N, C, HW)

    # Output buffer in fp32 for stable compute; cast back later
    y_flat = torch.empty_like(x_flat, dtype=torch.float32)

    assert C % num_groups == 0, "num_groups must divide number of channels"
    channels_per_group = C // num_groups
    group_elems = channels_per_group * HW

    # Prepare affine params in fp32
    if (weight is None) or (bias is None):
        w = torch.ones(C, device=x.device, dtype=torch.float32)
        b = torch.zeros(C, device=x.device, dtype=torch.float32)
    else:
        w = weight.to(torch.float32)
        b = bias.to(torch.float32)

    grid = (N * num_groups,)
    _groupnorm_fwd_kernel[grid](
        x_flat, y_flat, w, b,
        N, C, HW, num_groups, channels_per_group, group_elems, float(eps),
    )
    y = y_flat.view_as(x).to(x_in.dtype)
    return y


class ModelNew(nn.Module):
    """
    Simple model that performs Group Normalization.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Use Triton kernel on CUDA tensors without autograd; fallback otherwise for exact semantics
        if x.is_cuda and (not x.requires_grad):
            return group_norm_triton(x, self.gn.weight, self.gn.bias, self.gn.num_groups, self.gn.eps)
        else:
            return self.gn(x)

batch_size = 16
features = 64
num_groups = 8
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features, num_groups] # num_features