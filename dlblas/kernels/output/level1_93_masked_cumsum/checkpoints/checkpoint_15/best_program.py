# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.

        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        if self.dim != 1:
            return torch.cumsum(x * mask, dim=self.dim)
            
        x = x.contiguous()
        mask = mask.contiguous()
        output = torch.empty_like(x)
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        grid = (batch_size,)
        _masked_cumsum_kernel[grid](
            x, mask, output,
            x.stride(0), x.stride(1),
            seq_len
        )
        return output

@triton.jit
def _masked_cumsum_kernel(
    x_ptr,
    mask_ptr,
    output_ptr,
    stride_batch,
    stride_row,
    seq_len,
):
    row_idx = tl.program_id(0)
    base = row_idx * stride_batch
    total = 0.0
    for j in range(0, seq_len):
        offset = base + j * stride_row
        x_val = tl.load(x_ptr + offset)
        mask_val = tl.load(mask_ptr + offset).to(tl.float32)
        data = x_val * mask_val
        total += data
        tl.store(output_ptr + offset, total)

batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    x = torch.randn(batch_size, *input_shape)
    mask = torch.randint(0, 2, x.shape).bool()  # Random boolean mask
    return [x, mask]

def get_init_inputs():
    return [dim]

# =================== EVOLVE-BLOCK-END ===================