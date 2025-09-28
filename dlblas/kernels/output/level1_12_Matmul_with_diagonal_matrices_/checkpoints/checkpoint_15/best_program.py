# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        N, M = B.shape
        A = A.contiguous()
        B = B.contiguous()
        C = torch.empty_like(B)
        if N * M == 0:
            return C
            
        grid = (N,)
        BLOCK_SIZE = 1024
        self.diag_mult_kernel[grid](A, B, C, M, BLOCK_SIZE=BLOCK_SIZE)
        return C

    @staticmethod
    @triton.jit
    def diag_mult_kernel(
        A_ptr, 
        B_ptr, 
        C_ptr, 
        M, 
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        a_val = tl.load(A_ptr + pid)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        
        for col_start in range(0, M, BLOCK_SIZE):
            col_idx = col_start + col_offsets
            mask = col_idx < M
            
            b_ptrs = B_ptr + pid * M + col_idx
            b_vals = tl.load(b_ptrs, mask=mask, other=0.0)
            
            c_vals = a_val * b_vals
            c_ptrs = C_ptr + pid * M + col_idx
            tl.store(c_ptrs, c_vals, mask=mask)

M = 4096
N = 4096

def get_inputs():
    A = torch.randn(N)
    B = torch.randn(N, M)
    return [A, B]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================