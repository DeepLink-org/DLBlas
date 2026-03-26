import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _matmul_tf32_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    k = 0
    while k < K:
        k_mask = (k + offs_k) < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_mask[None, :]), other=0.0)
        b = tl.load(b_ptrs, mask=(k_mask[:, None]) & (offs_n[None, :] < N), other=0.0)

        # Emulate TF32 operand rounding (round mantissa to 10 bits) to match torch.matmul on Ampere/Hopper
        ai = tl.view(a, tl.int32)
        bi = tl.view(b, tl.int32)
        sign_a = ai & 0x80000000
        sign_b = bi & 0x80000000
        mag_a = ai & 0x7FFFFFFF
        mag_b = bi & 0x7FFFFFFF
        mag_a = (mag_a + 0x00001000) & 0xFFFFE000
        mag_b = (mag_b + 0x00001000) & 0xFFFFE000
        a = tl.view(sign_a | mag_a, tl.float32)
        b = tl.view(sign_b | mag_b, tl.float32)

        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        # Use Triton kernel on CUDA float32 tensors when TF32 is enabled to match torch.matmul semantics on Hopper/Ampere
        if (
            A.is_cuda and B.is_cuda and
            A.dtype == torch.float32 and B.dtype == torch.float32 and
            torch.backends.cuda.matmul.allow_tf32
        ):
            M, K = A.shape
            Kb, N = B.shape
            assert Kb == K, "Inner dimensions must match for matmul"

            # Allocate output
            C = torch.empty((M, N), device=A.device, dtype=A.dtype)

            # Compute strides in elements
            stride_am, stride_ak = A.stride()
            stride_bk, stride_bn = B.stride()
            stride_cm, stride_cn = C.stride()

            # Tile sizes tuned for large GEMM on Hopper
            BLOCK_M = 128
            BLOCK_N = 128
            BLOCK_K = 64

            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

            _matmul_tf32_kernel[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=8, num_stages=3,
            )
            return C
        else:
            # Fallback to PyTorch matmul for other cases or when TF32 is disabled
            return torch.matmul(A, B)


M = 1024
K = 4096
N = 2048

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed