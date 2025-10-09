# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def cross_entropy_kernel(
    logits_ptr,
    targets_ptr,
    output_ptr,
    n_classes,
    stride_logits_batch,
    stride_logits_class,
    batch_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    
    target_idx = tl.load(targets_ptr + pid)
    row_start = pid * stride_logits_batch
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_classes
    
    logits_ptrs = logits_ptr + row_start + col_offsets * stride_logits_class
    logits = tl.load(logits_ptrs, mask=mask, other=-float('inf'))
    
    max_val = tl.max(logits, axis=0)
    logits = logits - max_val
    exp_logits = tl.exp(logits)
    sum_exp = tl.sum(exp_logits, axis=0)
    
    mask_target = col_offsets == target_idx
    target_logit = tl.sum(tl.where(mask_target, logits, 0.0), axis=0)
    
    log_softmax = target_logit - tl.log(sum_exp)
    loss = -log_softmax
    tl.store(output_ptr + pid, loss)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, predictions, targets):
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        batch_size, n_classes = predictions.shape[0], predictions.shape[1]
        
        per_sample_loss = torch.empty(batch_size, device=predictions.device, dtype=predictions.dtype)
        BLOCK_SIZE = triton.next_power_of_2(n_classes)
        
        grid = (triton.cdiv(batch_size, 128) * 128,)
        cross_entropy_kernel[grid](
            predictions, targets, per_sample_loss,
            n_classes,
            predictions.stride(0), predictions.stride(1),
            batch_size,
            BLOCK_SIZE
        )
        return per_sample_loss.mean()

batch_size = 4096
num_classes = 10
input_shape = (num_classes,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randint(0, num_classes, (batch_size,))]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================