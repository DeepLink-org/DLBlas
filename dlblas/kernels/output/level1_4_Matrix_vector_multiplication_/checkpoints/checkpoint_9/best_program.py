# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _matvec_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K,
    stride_am, stride_ak,
    stride_bk, stride_b1,
    stride_cm, stride_c1,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    offs_m = pid
    c_ptr = C_ptr + offs_m * stride_cm

    accumulator = 0.0
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask = offs_k < K

        a_ptr = A_ptr + offs_m * stride_am + offs_k * stride_ak
        a = tl.load(a_ptr, mask=mask, other=0.0)

        b_ptr = B_ptr + offs_k * stride_bk
        b = tl.load(b_ptr, mask=mask, other=0.0)

        product = a * b
        accumulator += tl.sum(product, axis=0)

    tl.store(c_ptr, accumulator)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, K = A.shape
        assert B.shape == (K, 1), f"B must be of shape ({K}, 1)"
        C = torch.empty((M, 1), device=A.device, dtype=A.dtype)
        
        if M == 0:
            return C

        BLOCK_K = 1024
        grid = (M,)
        stride_am, stride_ak = A.stride()
        stride_bk, stride_b1 = B.stride()
        stride_cm, stride_c1 = C.stride()

        _matvec_kernel[grid](
            A, B, C,
            M, K,
            stride_am, stride_ak,
            stride_bk, stride_b1,
            stride_cm, stride_c1,
            BLOCK_K=BLOCK_K
        )
        return C

M = 256
K = 131072

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, 1)
    return [A, B]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================