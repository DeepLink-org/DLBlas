import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=4, num_stages=1),
    ],
    key=["M", "N"],
)
@triton.jit
def _affine_per_col_kernel(
    y_ptr,          # [M, N] input/output (row-major)
    alpha_ptr,      # [N] per-column scale
    beta_ptr,       # [N] per-column bias
    M, N,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Load tile of y
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y = tl.load(y_ptrs, mask=mask, other=0.0, cache_modifier=".cg")

    # Load per-column params once per tile
    alpha = tl.load(alpha_ptr + offs_n, mask=mask_n, other=1.0)
    beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0)

    # Apply: y = y * alpha + beta
    y = y * alpha[None, :] + beta[None, :]

    # Store back
    tl.store(y_ptrs, y, mask=mask, cache_modifier=".cg")


def _linear_then_scale_bn_infer(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, s: torch.Tensor, bn: nn.BatchNorm1d) -> torch.Tensor:
    # 1) GEMM via cuBLAS: y = x @ W^T + b
    y = torch.nn.functional.linear(x, w, b).contiguous()
    M, N = y.shape

    # 2) Precompute fused per-channel affine params for eval BN with prior scaling:
    # out = (y * s - rm) / sqrt(rv + eps) * gamma + beta
    #     = y * (s * gamma / sqrt(rv + eps)) + (beta - rm * gamma / sqrt(rv + eps))
    s_ = s.contiguous()
    rm = bn.running_mean
    rv = bn.running_var
    eps = bn.eps
    if bn.weight is None:
        gamma = torch.ones_like(s_, dtype=torch.float32)
    else:
        gamma = bn.weight
    if bn.bias is None:
        beta = torch.zeros_like(s_, dtype=torch.float32)
    else:
        beta = bn.bias

    inv_std = torch.rsqrt(rv + eps)
    alpha = (s_.to(torch.float32) * gamma.to(torch.float32)) * inv_std
    beta2 = beta.to(torch.float32) - (rm.to(torch.float32) * gamma.to(torch.float32)) * inv_std

    alpha = alpha.contiguous()
    beta2 = beta2.contiguous()

    # 3) In-place fused epilogue: y = y * alpha + beta2
    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    _affine_per_col_kernel[grid](
        y, alpha, beta2,
        M, N,
        y.stride(0), y.stride(1),
    )
    return y


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, scales the result, and applies batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        # Use Triton path only in inference without autograd to preserve exact training/autograd semantics.
        use_triton = (
            x.is_cuda
            and x.dtype == torch.float32
            and (not self.training)
            and (not x.requires_grad)
        )
        if use_triton:
            return _linear_then_scale_bn_infer(x, self.gemm.weight, self.gemm.bias, self.scale, self.bn)
        # Fallback to PyTorch path to preserve full autograd/training behavior
        x = self.gemm(x)
        x = x * self.scale
        x = self.bn(x)
        return x


batch_size = 128
in_features = 1024
out_features = 512
scale_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scale_shape]