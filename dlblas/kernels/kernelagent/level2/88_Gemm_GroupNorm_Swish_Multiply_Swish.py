import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_gn_swish_mul_swish_kernel(
    x_ptr,            # (N, C)
    gamma_ptr,        # (C,)
    beta_ptr,         # (C,)
    mulw_ptr,         # (C,)
    y_ptr,            # (N, C)
    N,                # batch size
    C,                # out_features / channels
    G,                # num_groups
    GROUP_SIZE,       # C // G
    EPS,              # eps for GroupNorm
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b_idx = pid // G
    g_idx = pid % G

    # Channel indices for this group
    offs = tl.arange(0, BLOCK_SIZE)
    c_start = g_idx * GROUP_SIZE
    c_idx = c_start + offs
    mask = offs < GROUP_SIZE

    # Base offset for this sample
    base = b_idx * C

    # Load group slice once
    x = tl.load(x_ptr + base + c_idx, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / GROUP_SIZE

    # Compute variance using centered values
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / GROUP_SIZE
    inv_std = 1.0 / tl.sqrt(var + EPS)

    # Normalize + affine
    gamma = tl.load(gamma_ptr + c_idx, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + c_idx, mask=mask, other=0.0)
    gn = (x - mean) * inv_std
    y = gn * gamma + beta

    # Swish: y * sigmoid(y)
    y = y * tl.sigmoid(y)

    # Multiply with external weight
    mw = tl.load(mulw_ptr + c_idx, mask=mask, other=0.0)
    y = y * mw

    # Second Swish
    out = y * tl.sigmoid(y)

    tl.store(y_ptr + base + c_idx, out, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, GroupNorm, Swish, Multiply, and Swish operations.
    Fused Triton kernel implements: GroupNorm + Swish + Multiply + Swish.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape)) 

    def forward(self, x):
        # GEMM (cuBLAS / PyTorch)
        x = self.gemm(x)

        # Fallback to PyTorch reference if not CUDA or shape unsupported
        if (not x.is_cuda) or (x.dim() != 2):
            x = self.group_norm(x)
            x = x * torch.sigmoid(x)
            x = x * self.multiply_weight
            x = x * torch.sigmoid(x)
            return x

        N, C = x.shape
        G = self.group_norm.num_groups
        assert C % G == 0, "out_features must be divisible by num_groups for GroupNorm"
        GROUP_SIZE = C // G
        EPS = self.group_norm.eps

        # Ensure contiguous memory
        x = x.contiguous()
        y = torch.empty_like(x)

        gamma = self.group_norm.weight
        beta = self.group_norm.bias
        mulw = self.multiply_weight

        # Launch Triton kernel: one program per (batch, group)
        total_groups = N * G
        # Tile: one CTA handles one group; choose BLOCK_SIZE >= GROUP_SIZE with mask
        BLOCK_SIZE = 1 << (GROUP_SIZE - 1).bit_length()  # next power of two
        # Cap BLOCK_SIZE to avoid excessive threads (sane upper bound)
        BLOCK_SIZE = min(BLOCK_SIZE, 1024)

        # Favor fewer warps and stages for small tiles to reduce overhead on Hopper
        num_warps = 2 if BLOCK_SIZE <= 64 else 4
        num_stages = 2

        grid = (total_groups,)
        _fused_gn_swish_mul_swish_kernel[grid](
            x, gamma, beta, mulw, y,
            N, C, G, GROUP_SIZE, EPS,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return y


batch_size = 128
in_features = 512
out_features = 1024
num_groups = 16
multiply_weight_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]