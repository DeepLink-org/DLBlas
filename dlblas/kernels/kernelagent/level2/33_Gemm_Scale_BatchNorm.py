import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=5),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _linear_scale_kernel(
    A_ptr,  # [M, K]
    B_ptr,  # stored as [N, K], accessed with strides to emulate [K, N]
    Bias_ptr,  # [N]
    Scale_ptr,  # [N]
    C_ptr,  # [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
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
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_iter = 0
    while k_iter < K:
        k_mask_a = (offs_m[:, None] < M) & (k_iter + offs_k[None, :] < K)
        k_mask_b = (k_iter + offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=k_mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask_b, other=0.0)
        acc += tl.dot(a, b)
        k_iter += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias and apply scale
    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    scale = tl.load(Scale_ptr + offs_n, mask=offs_n < N, other=1.0)
    acc = acc + bias[None, :]
    acc = acc * scale[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(nn.Module):
    """
    Simple model that performs a GEMM (general matrix multiplication), applies scaling,
    and then batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def _fused_linear_scale(self, x: torch.Tensor) -> torch.Tensor:
        # x: [M, K]; weight: [N, K]
        M, K = x.shape
        N = self.gemm.weight.shape[0]
        y = torch.empty((M, N), device=x.device, dtype=torch.float32)

        A = x
        B = self.gemm.weight  # shape [N, K]
        Bias = self.gemm.bias if self.gemm.bias is not None else torch.zeros(N, device=x.device, dtype=torch.float32)
        Scale = self.scale

        # Ensure dtypes and contiguous layout
        A = A.contiguous()
        B = B.contiguous()
        Bias = Bias.contiguous()
        Scale = Scale.contiguous()

        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
        _linear_scale_kernel[grid](
            A, B, Bias, Scale, y,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(1), B.stride(0),  # access as [K, N]
            y.stride(0), y.stride(1),
        )
        return y

    def forward(self, x):
        # Use fused Triton kernel only in inference (no grads) and on CUDA
        use_triton = x.is_cuda and (not torch.is_grad_enabled())
        if use_triton:
            y = self._fused_linear_scale(x)
            y = self.bn(y)
            return y
        else:
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