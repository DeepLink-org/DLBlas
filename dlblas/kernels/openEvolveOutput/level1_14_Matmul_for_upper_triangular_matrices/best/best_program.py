# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        N = A.shape[0]
        # Create output tensor initialized to zeros
        C = torch.zeros_like(A)
        A = A.contiguous()
        B = B.contiguous()
        BLOCK_SIZE = 128
        grid = (N,)
        self.kernel[grid](A, B, C, 
                          A.stride(0), A.stride(1),
                          B.stride(0), B.stride(1),
                          C.stride(0), C.stride(1),
                          N, BLOCK_SIZE=BLOCK_SIZE)
        return C

    @staticmethod
    @triton.jit
    def kernel(
        A_ptr, B_ptr, C_ptr,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        i = pid

        # Skip if beyond matrix dimension
        if i >= N:
            return

        # Process columns in chunks for coalesced memory access
        for j_base in range(i, N, BLOCK_SIZE):
            col_offsets = tl.arange(0, BLOCK_SIZE)
            j = j_base + col_offsets
            mask = j < N
            c = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

            # Vectorized accumulation loop
            for k in range(i, N):
                a = tl.load(A_ptr + i * stride_am + k * stride_ak)
                b = tl.load(B_ptr + k * stride_bk + j * stride_bn, mask=mask, other=0)
                update_mask = (k <= j) & mask
                c = tl.where(update_mask, c + a * b, c)

            # Store computed values with mask
            tl.store(C_ptr + i * stride_cm + j * stride_cn, c, mask=mask)

N = 4096

def get_inputs():
    A = torch.triu(torch.randn(N, N))
    B = torch.triu(torch.randn(N, N))
    return [A, B]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================