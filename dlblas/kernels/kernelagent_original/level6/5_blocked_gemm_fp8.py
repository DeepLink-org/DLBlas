import torch
import torch.nn as nn
from torch import Tensor


def _expand_scale_A(scale_A: Tensor, group_size: int) -> Tensor:
    # scale_A: (M, K//G) -> (M, K)
    return scale_A.repeat_interleave(group_size, dim=1)


def _expand_scale_B(scale_B: Tensor, group_size: int, K: int, N: int) -> Tensor:
    # scale_B: (K//G, N//G) -> (K, N)
    return scale_B.repeat_interleave(group_size, dim=0).repeat_interleave(group_size, dim=1)[:K, :N]


def blocked_gemm_fp8(A: Tensor,
                     A_scale: Tensor,
                     B: Tensor,
                     B_scale: Tensor,
                     out_dtype: torch.dtype = torch.float16) -> Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K2 == K
    # infer group size from scale shapes
    group_size_k = K // A_scale.shape[1]
    scale_A_full = _expand_scale_A(A_scale, group_size_k)
    # scale_B expected (K//G, N//G)
    group_size_n = N // B_scale.shape[1]
    assert group_size_k == K // B_scale.shape[0]
    scale_B_full = _expand_scale_B(B_scale, group_size_k, K, N)

    A = A.to(torch.float32) * scale_A_full
    B = B.to(torch.float32) * scale_B_full
    C = A @ B
    return C.to(out_dtype)


class Model(nn.Module):
    def __init__(self, out_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.out_dtype = out_dtype

    def forward(self, A: Tensor, A_scale: Tensor, B: Tensor, B_scale: Tensor):
        return blocked_gemm_fp8(A, A_scale, B, B_scale, out_dtype=self.out_dtype)


# Hyperparameters mirroring test setup
M = 256
K = 512
N = 1088
group_size = 128
quant_dtype = torch.float8_e4m3fn
out_dtype = torch.float16


def _aligned_size(a: int, b: int) -> int:
    return (a + b - 1) // b * b


def _make_A(M, K, group_size, out_dtype):
    quant_A = torch.rand(M, K // group_size, group_size, dtype=torch.float32)
    # -1 ~ 1
    quant_A = quant_A * 2 - 1
    # scaling abs max to fmax
    finfo = torch.finfo(out_dtype)
    fmax = finfo.max
    scaling = fmax / quant_A.abs().amax(-1, keepdim=True)
    quant_A *= scaling
    quant_A = quant_A.to(out_dtype).to(torch.float32)

    # create scale and A
    scale = torch.rand(M, K // group_size, dtype=torch.float32)
    scale /= fmax
    A = quant_A * scale[..., None]

    A = A.reshape(M, K)
    quant_A = quant_A.reshape(M, K).to(out_dtype)
    scale = scale.T.contiguous().T
    return A, quant_A, scale


def _make_B(K, N, group_size, out_dtype):
    K_aligned = _aligned_size(K, group_size)
    N_aligned = _aligned_size(N, group_size)

    quant_B = torch.rand(K_aligned // group_size,
                         group_size,
                         N_aligned // group_size,
                         group_size,
                         dtype=torch.float32)
    quant_B = quant_B * 2 - 1

    # scaling abs max to fmax
    finfo = torch.finfo(out_dtype)
    fmax = finfo.max
    scaling = fmax / quant_B.abs().amax((1, 3), keepdim=True)
    quant_B *= scaling
    quant_B = quant_B.to(out_dtype).to(torch.float32)

    scale = torch.rand(K_aligned // group_size, 1, N_aligned // group_size, 1, dtype=torch.float32)
    scale /= fmax

    B = quant_B * scale

    B = B.reshape(K_aligned, N_aligned)[:K, :N]
    quant_B = quant_B.reshape(K_aligned, N_aligned).to(out_dtype)[:K, :N]
    scale = scale.reshape(K_aligned // group_size, N_aligned // group_size)
    quant_B = quant_B.transpose(0, 1).contiguous().transpose(0, 1)
    return B, quant_B, scale


def get_inputs():
    # Use the same functions as the test
    _, quant_A, scale_A = _make_A(M, K, group_size, quant_dtype)
    _, quant_B, scale_B = _make_B(K, N, group_size, quant_dtype)
    return [quant_A, scale_A, quant_B, scale_B]


def get_init_inputs():
    return []

