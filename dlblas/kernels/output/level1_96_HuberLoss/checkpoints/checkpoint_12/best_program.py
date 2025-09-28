# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def smooth_l1_loss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    p = tl.load(predictions_ptr + offsets, mask=mask)
    t = tl.load(targets_ptr + offsets, mask=mask)
    
    diff = p - t
    abs_diff = tl.abs(diff)
    loss_element = tl.where(
        abs_diff < 1,
        0.5 * abs_diff * abs_diff,
        abs_diff - 0.5
    )
    tl.store(output_ptr + offsets, loss_element, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, predictions, targets):
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        total_elements = predictions.numel()
        
        if total_elements == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        output_loss = torch.empty(total_elements, device=predictions.device, dtype=torch.float32)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
        
        smooth_l1_loss_kernel[grid](
            predictions,
            targets,
            output_loss,
            total_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output_loss.mean()

batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================