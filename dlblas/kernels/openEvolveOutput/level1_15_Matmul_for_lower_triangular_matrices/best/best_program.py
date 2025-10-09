# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triangular_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    stride_am, stride_ak,
    stride_bm, stride_bk,
    stride_cm, stride_ck,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_start = tl.program_id(1) * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    col_mask = (col_offsets <= row) & (col_offsets < n)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Compute dot product for valid columns
    for k in range(0, row + 1):
        a_val = tl.load(A_ptr + row * stride_am + k * stride_ak)
        b_vals = tl.load(
            B_ptr + k * stride_bm + col_offsets * stride_bk,
            mask=col_mask,
            other=0.0
        )
        acc += a_val * b_vals
    
    # Store results
    tl.store(
        C_ptr + row * stride_cm + col_offsets * stride_ck,
        acc.to(tl.float32),
        mask=col_mask
    )

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        n = A.size(0)
        C = torch.zeros_like(A)
        BLOCK_SIZE = 64
        
        # Calculate grid size
        grid = (n, triton.cdiv(n, BLOCK_SIZE))
        
        triangular_matmul_kernel[grid](
            A, B, C,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            n, BLOCK_SIZE
        )
        return C

M = 4096

def get_inputs():
    A = torch.randn(M, M)
    B = torch.randn(M, M)
    A = torch.tril(A)
    B = torch.tril(B)
    return [A, B]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================