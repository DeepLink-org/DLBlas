# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _masked_cumsum_kernel(
    output_ptr,
    x_ptr,
    mask_ptr,
    n_cols,
    stride,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    start = row_idx * stride
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < n_cols
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(mask_ptr + offsets, mask=mask, other=False).to(tl.float32)
    vals = x * m
    cumsum = tl.associative_scan(vals, 0, tl.sum)
    tl.store(output_ptr + offsets, cumsum, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        original_shape = x.shape
        dim = self.dim if self.dim >= 0 else self.dim + x.dim()
        
        n_rows = 1
        for i, s in enumerate(original_shape):
            if i != dim:
                n_rows *= s
        n_cols = original_shape[dim]
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        
        x_flat = x.moveaxis(dim, -1).contiguous().view(n_rows, n_cols)
        mask_flat = mask.moveaxis(dim, -1).contiguous().view(n_rows, n_cols)
        output_flat = torch.empty_like(x_flat)
        
        grid = (n_rows,)
        _masked_cumsum_kernel[grid](
            output_flat, x_flat, mask_flat, n_cols, 
            x_flat.stride(0), BLOCK_SIZE
        )
        
        output = output_flat.view(original_shape[:dim] + (n_cols,) + original_shape[dim+1:])
        return output.moveaxis(-1, dim)

batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    x = torch.randn(batch_size, *input_shape)
    mask = torch.randint(0, 2, x.shape).bool()
    return [x, mask]

def get_init_inputs():
    return [dim]
# =================== EVOLVE-BLOCK-END ===================