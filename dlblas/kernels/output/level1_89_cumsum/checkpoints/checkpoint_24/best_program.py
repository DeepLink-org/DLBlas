# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def add_combine(a, b):
    return a + b

@triton.jit
def _cumsum_kernel(
    x_ptr,
    out_ptr,
    L,
    stride_x_row,
    stride_x_inner,
    stride_out_row,
    stride_out_inner,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start_x = x_ptr + pid * stride_x_row
    row_start_out = out_ptr + pid * stride_out_row
    cumulative = 0.0
    num_chunks = tl.cdiv(L, BLOCK_SIZE)
    
    for chunk_idx in range(0, num_chunks):
        offs_chunk = chunk_idx * BLOCK_SIZE
        offs = offs_chunk + tl.arange(0, BLOCK_SIZE)
        mask = offs < L
        
        x_ptrs = row_start_x + offs * stride_x_inner
        chunk = tl.load(x_ptrs, mask=mask, other=0.0)
        
        scanned_chunk = tl.associative_scan(chunk, axis=0, combine_fn=add_combine)
        total_chunk = tl.sum(chunk, axis=0)
        
        result_chunk = scanned_chunk + cumulative
        out_ptrs = row_start_out + offs * stride_out_inner
        tl.store(out_ptrs, result_chunk, mask=mask)
        
        cumulative += total_chunk

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        original_shape = x.shape
        L = x.shape[self.dim]
        x_flat = x.contiguous().view(-1, L)
        N, L = x_flat.shape
        output_flat = torch.empty_like(x_flat)
        
        grid = (N,)
        BLOCK_SIZE = 1024
        _cumsum_kernel[grid](
            x_flat,
            output_flat,
            L,
            x_flat.stride(0),
            x_flat.stride(1),
            output_flat.stride(0),
            output_flat.stride(1),
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output_flat.view(original_shape)

# Define input dimensions and parameters
batch_size = 128
input_shape = (4000,)  # Example shape (arbitrary)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]
# =================== EVOLVE-BLOCK-END ===================