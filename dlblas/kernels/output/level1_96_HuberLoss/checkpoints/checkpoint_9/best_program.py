# ================== EVOLVE-BLOCK-START ==================
import torch
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def smooth_l1_loss_kernel(
    predictions_ptr,
    targets_ptr,
    partial_sums_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    preds = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    diff = preds - targets
    abs_diff = tl.abs(diff)
    loss = tl.where(abs_diff < 1, 0.5 * diff * diff, abs_diff - 0.5)
    
    block_loss = tl.sum(loss, axis=0)
    tl.store(partial_sums_ptr + pid, block_loss)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        n_elements = predictions.numel()
        if n_elements == 0:
            return torch.tensor(0.0, device=predictions.device)
            
        predictions_flat = predictions.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)
        
        BLOCK_SIZE = 1024
        grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
        partial_sums = torch.zeros(grid_size, device=predictions.device, dtype=torch.float32)
        
        smooth_l1_loss_kernel[(grid_size,)](predictions_flat, targets_flat, partial_sums, 
                                            n_elements, BLOCK_SIZE=BLOCK_SIZE)
        
        return partial_sums.sum() / n_elements

batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return []

# =================== EVOLVE-BLOCK-END ===================