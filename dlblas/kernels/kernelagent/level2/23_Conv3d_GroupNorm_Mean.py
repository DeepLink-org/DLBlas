import torch
import torch.nn as nn
import triton
import triton.language as tl

# Help cuDNN pick optimal conv kernels for fixed shapes
torch.backends.cudnn.benchmark = True


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 4096}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 8192}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 16384}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 16384}, num_warps=16, num_stages=4),
    ],
    key=["M", "GROUP_SIZE"],
)
@triton.jit
def _group_mean_contrib_kernel(
    x_ptr,              # *f32 [N, C, D, H, W], contiguous in last 3 dims (per-channel)
    gamma_ptr,          # *f32 [C]
    sum_gamma_ptr,      # *f32 [G]
    out_ptr,            # *f32 [N] partial numerator contributions per sample
    N, C, G, M,         # ints
    stride_n,           # x.stride(0)
    stride_c,           # x.stride(1) == M for NCDHW contiguous
    eps,                # float32 epsilon
    GROUP_SIZE: tl.constexpr,  # channels per group
    BLOCK_M: tl.constexpr,     # tile over flattened spatial M = D*H*W
):
    pid = tl.program_id(axis=0)
    n = pid // G
    g = pid % G

    base_n = n * stride_n
    c_start = g * GROUP_SIZE

    # Accumulators in fp32
    acc_A = tl.zeros((), dtype=tl.float32)  # sum over group of x
    acc_B = tl.zeros((), dtype=tl.float32)  # sum over group of x^2
    acc_T = tl.zeros((), dtype=tl.float32)  # sum over group of gamma[c] * sum_spatial(x_{n,c})

    offs = tl.arange(0, BLOCK_M)
    tl.max_contiguous(offs, BLOCK_M)

    # Loop over channels in the group; unrolled at compile-time
    for c_rel in tl.static_range(0, GROUP_SIZE):
        c = c_start + c_rel
        base = x_ptr + base_n + c * stride_c

        s_chan = tl.zeros((), dtype=tl.float32)
        ss_chan = tl.zeros((), dtype=tl.float32)

        m = 0
        while m < M:
            idx = m + offs
            mask = idx < M
            vals = tl.load(base + idx, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            s_chan += tl.sum(vals, axis=0)
            ss_chan += tl.sum(vals * vals, axis=0)
            m += BLOCK_M

        gamma_c = tl.load(gamma_ptr + c)
        acc_A += s_chan
        acc_B += ss_chan
        acc_T += gamma_c * s_chan

    # Group statistics
    Mg = GROUP_SIZE * M
    Mg_f32 = tl.full((), Mg, dtype=tl.float32)
    M_f32 = tl.full((), M, dtype=tl.float32)
    mu = acc_A / Mg_f32
    var = acc_B / Mg_f32 - mu * mu
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Contribution of this (n,g) to the numerator
    sum_gamma_g = tl.load(sum_gamma_ptr + g)
    term_g = (acc_T - mu * (M_f32 * sum_gamma_g)) * inv_std

    # Accumulate into out[n]
    tl.atomic_add(out_ptr + n, term_g)


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, computes the mean
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        # Feed Conv3d with channels_last_3d for better throughput on Hopper/H200
        if x.is_cuda:
            x = x.to(memory_format=torch.channels_last_3d)

        # Convolution
        y = self.conv(x)

        if not y.is_cuda:
            # CPU fallback: exact semantics
            y = self.group_norm(y)
            return y.mean(dim=[1, 2, 3, 4])

        # Ensure per-(n,c) spatial contiguity for the Triton kernel
        y = y.contiguous()  # [N, C, D, H, W], NCDHW contiguous

        N, C, D, H, W = y.shape
        M = D * H * W
        G = self.group_norm.num_groups
        assert C % G == 0, "Channels must be divisible by num_groups for GroupNorm."
        GROUP_SIZE = C // G

        # GroupNorm parameters
        gamma = self.group_norm.weight.to(device=y.device, dtype=torch.float32)  # [C]
        beta = self.group_norm.bias.to(device=y.device, dtype=torch.float32)     # [C]
        eps = float(self.group_norm.eps)

        # Precompute per-group sum of gamma and global sum of beta
        sum_gamma = gamma.view(G, GROUP_SIZE).sum(dim=1).contiguous()  # [G]
        sum_beta = beta.sum()  # scalar

        # Accumulate numerator contributions per-sample across groups in-kernel
        partial = torch.zeros((N,), device=y.device, dtype=torch.float32)

        # Launch kernel: one program per (n, g)
        grid = lambda meta: (N * G,)
        _group_mean_contrib_kernel[grid](
            y,
            gamma,
            sum_gamma,
            partial,
            N,
            C,
            G,
            M,
            y.stride(0),
            y.stride(1),
            eps,
            GROUP_SIZE=GROUP_SIZE,
        )

        # Final mean across [C, D, H, W] after GroupNorm
        numerator = partial + M * sum_beta
        out = numerator / (C * M)

        return out


batch_size = 128
in_channels = 3
out_channels = 16
D, H, W = 16, 32, 32
kernel_size = 3
num_groups = 8

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups]