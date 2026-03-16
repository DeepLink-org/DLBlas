import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _groupnorm_hardtanh_kernel(
    x_ptr,           # [N, C] input (post-GEMM), contiguous
    gamma_ptr,       # [C] groupnorm affine weight
    beta_ptr,        # [C] groupnorm affine bias
    out_ptr,         # [N, C] output
    N,               # number of rows (batch size)
    C,               # number of channels (out_features)
    G,               # number of groups
    Cg,              # channels per group = C // G
    eps,             # eps for numerical stability (float)
    minv,            # hardtanh min
    maxv,            # hardtanh max
    BLOCK_SIZE: tl.constexpr,  # power-of-two >= Cg
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    offs = tl.arange(0, BLOCK_SIZE)
    tl.max_contiguous(offs, BLOCK_SIZE)
    tl.multiple_of(offs, 16)

    offs_c = g * Cg + offs
    row_start = n * C
    base = row_start + offs_c

    ch_mask = offs < Cg
    mask = ch_mask & (n < N)

    # Load input and affine params; compute in fp32 for stability
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + offs_c, mask=ch_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_c, mask=ch_mask, other=0.0).to(tl.float32)

    # Compute mean and variance via E[x] and E[x^2]
    inv_cg = 1.0 / tl.full((), Cg, tl.float32)
    sum_x = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)
    mean = sum_x * inv_cg
    var = sum_x2 * inv_cg - mean * mean

    # Normalize with fused affine: y = x * (gamma * inv_std) + (beta - mean * gamma * inv_std)
    inv_std = tl.rsqrt(var + eps)
    scale = gamma * inv_std
    shift = beta - mean * scale
    y = x * scale + shift

    # HardTanh clamp
    y = tl.maximum(tl.minimum(y, maxv), minv)

    tl.store(out_ptr + base, y, mask=mask)


class ModelNew(nn.Module):
    """
    Simple model that performs a GEMM, applies Group Normalization, and then HardTanh.
    The GEMM uses PyTorch's highly-optimized cuBLAS backend.
    GroupNorm + HardTanh are fused into a single Triton kernel for improved performance.
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # GEMM via cuBLAS
        y = self.gemm(x)

        # CPU / non-CUDA fallback
        if not y.is_cuda:
            y = self.group_norm(y)
            y = self.hardtanh(y)
            return y

        # Triton fused GroupNorm + HardTanh
        y = y.contiguous()
        N, C = y.shape
        G = self.group_norm.num_groups
        assert C % G == 0, "out_features must be divisible by num_groups"
        Cg = C // G

        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()
        eps = float(self.group_norm.eps)
        minv = float(self.hardtanh.min_val)
        maxv = float(self.hardtanh.max_val)

        out = torch.empty_like(y)

        # Tile size: next power-of-two of Cg for efficient reduction
        def next_power_of_two(v: int) -> int:
            return 1 if v <= 1 else 1 << ((v - 1).bit_length())
        BLOCK_SIZE = next_power_of_two(Cg)

        # Heuristic tuning for Hopper: small pipeline depth helps latency hiding
        if BLOCK_SIZE >= 256:
            num_warps = 8
            num_stages = 2
        elif BLOCK_SIZE >= 128:
            num_warps = 4
            num_stages = 2
        elif BLOCK_SIZE >= 64:
            num_warps = 2
            num_stages = 2
        else:
            num_warps = 1
            num_stages = 1

        grid = (N * G,)
        _groupnorm_hardtanh_kernel[grid](
            y, gamma, beta, out,
            N, C, G, Cg,
            eps, minv, maxv,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out


batch_size = 128
in_features = 1024
out_features = 512
num_groups = 8
hardtanh_min = -2.0
hardtanh_max = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, hardtanh_min, hardtanh_max]