import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=8),
        # Enable larger K tiles for better TF32 tensor-core utilization on Hopper/H200
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=5, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=5, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _linear_mul_leaky_kernel(
    A_ptr,  # [M, K]
    B_ptr,  # [N, K]  (weight)
    Bias_ptr,  # [N]
    C_ptr,  # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    multiplier, negative_slope,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_remaining = K
    while k_remaining > 0:
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < k_remaining)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Enable TF32 to leverage tensor cores while keeping fp32 accumulation
        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_remaining -= BLOCK_K

    # Add bias [N] broadcast across M
    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Multiply by scalar
    acc = acc * multiplier

    # LeakyReLU
    out = tl.where(acc >= 0, acc, acc * negative_slope)

    # Store
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, out, mask=c_mask)


def _fused_linear_mul_leaky(x: torch.Tensor,
                            weight: torch.Tensor,
                            bias: torch.Tensor,
                            multiplier: float,
                            negative_slope: float) -> torch.Tensor:
    """
    Prefer highly-optimized cuBLAS for GEMM, fusing multiplier via alpha/beta of addmm.
    For much larger problems, fall back to a single fused Triton GEMM + epilogue.
    """
    M, K = x.shape
    N = weight.shape[0]
    x_c = x if x.is_contiguous() else x.contiguous()

    # Heuristic: Triton only for very large problems
    use_triton = (M >= 256 and N >= 1024) or (K >= 2048 and (M >= 256 or N >= 512))

    if use_triton:
        y = torch.empty((M, N), device=x.device, dtype=x.dtype)
        bias_buf = bias if bias is not None else torch.zeros(N, device=x.device, dtype=x.dtype)
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
        _linear_mul_leaky_kernel[grid](
            x_c,                             # A_ptr
            weight,                          # B_ptr expects [N, K]
            bias_buf,                        # Bias_ptr
            y,                               # C_ptr
            M, N, K,
            x_c.stride(0), x_c.stride(1),
            weight.stride(0), weight.stride(1),
            y.stride(0), y.stride(1),
            float(multiplier), float(negative_slope),
        )
        return y
    else:
        # Fuse both GEMM scaling and bias scaling via alpha/beta to avoid extra ops
        if bias is not None:
            y = torch.addmm(bias, x_c, weight.t(), beta=float(multiplier), alpha=float(multiplier))
        else:
            # No bias: use beta=0 to skip reading 'input'
            zeros = torch.zeros(N, device=x.device, dtype=x.dtype)
            y = torch.addmm(zeros, x_c, weight.t(), beta=0.0, alpha=float(multiplier))
        return torch.nn.functional.leaky_relu_(y, negative_slope)


class ModelNew(nn.Module):
    """
    Simple model that performs a Gemm, multiplies the result, and applies LeakyReLU.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        # Fast path on CUDA: prefer cuBLAS GEMM with multiplier fused via alpha/beta,
        # and use a fully-fused Triton kernel for very large sizes.
        if x.is_cuda:
            y = _fused_linear_mul_leaky(
                x, self.gemm.weight, self.gemm.bias,
                float(self.multiplier), float(self.leaky_relu.negative_slope)
            )
            return y
        else:
            # CPU fallback: exact reference semantics
            x = self.gemm(x)
            x = x * self.multiplier
            x = self.leaky_relu(x)
            return x


batch_size = 128
in_features = 1024
out_features = 512
multiplier = 2.0
negative_slope = 0.1

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]