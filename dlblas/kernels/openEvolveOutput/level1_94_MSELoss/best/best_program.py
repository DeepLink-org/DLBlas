# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _mse_kernel(
    predictions_ptr,
    targets_ptr,
    partials_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    p = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    t = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    diff = p - t
    sq = diff * diff
    block_sum = tl.sum(sq, axis=0)
    tl.store(partials_ptr + pid, block_sum)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        
    def forward(self, predictions, targets):
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        n_elements = predictions.numel()
        if n_elements == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        partials = torch.zeros(grid[0], device=predictions.device, dtype=torch.float32)
        
        _mse_kernel[grid](
            predictions, 
            targets, 
            partials, 
            n_elements, 
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        total = partials.sum()
        return total / n_elements

batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================