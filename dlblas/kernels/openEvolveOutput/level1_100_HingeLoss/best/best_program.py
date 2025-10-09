# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _hinge_loss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    p = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    t = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    val = 1.0 - p * t
    clamped = tl.where(val > 0, val, 0.0)
    
    # Efficient tree reduction within block
    sum_val = tl.sum(clamped, axis=0)
    
    if pid == 0:
        mean = sum_val / n_elements
        tl.store(output_ptr, mean)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, predictions, targets):
        predictions_flat = predictions.view(-1)
        targets_flat = targets.view(-1)
        n_elements = predictions_flat.numel()
        
        if n_elements == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        output = torch.empty(1, device=predictions.device, dtype=torch.float32)
        
        # Optimize block size based on problem scale
        BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        _hinge_loss_kernel[grid](
            predictions_flat, targets_flat, output, n_elements, 
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output[0]

batch_size = 128
input_shape = (1,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randint(0, 2, (batch_size, 1)).float() * 2 - 1]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================