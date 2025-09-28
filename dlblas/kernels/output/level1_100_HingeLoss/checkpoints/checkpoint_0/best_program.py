# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hinge_loss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    batch_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size

    p = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    t = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    product = p * t
    element = 1.0 - product
    clamped = tl.where(element > 0, element, 0.0)
    
    block_sum = tl.sum(clamped, axis=0)
    if pid == 0:
        tl.store(output_ptr, block_sum / batch_size)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, predictions, targets):
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        batch_size = predictions.size(0)
        output = torch.empty(1, device=predictions.device, dtype=torch.float32)
        
        if batch_size == 0:
            output[0] = 0.0
            return output
        
        BLOCK_SIZE = min(triton.next_power_of_2(batch_size), 1024)
        grid = (triton.cdiv(batch_size, BLOCK_SIZE),)
        
        hinge_loss_kernel[grid](
            predictions, targets, output, 
            batch_size, BLOCK_SIZE=BLOCK_SIZE
        )
        return output

batch_size = 128
input_shape = (1,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randint(0, 2, (batch_size, 1)).float() * 2 - 1]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================