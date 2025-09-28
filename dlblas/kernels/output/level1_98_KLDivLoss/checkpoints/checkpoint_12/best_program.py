# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _kl_div_rowwise(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    ELEMENTS_PER_THREAD: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_offset = row_idx * n_cols
    
    # Each thread processes ELEMENTS_PER_THREAD consecutive elements
    base = tl.arange(0, BLOCK_SIZE) * ELEMENTS_PER_THREAD
    term = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for i in range(ELEMENTS_PER_THREAD):
        col = base + i
        mask = col < n_cols
        t_offsets = row_offset + col
        t = tl.load(targets_ptr + t_offsets, mask=mask, other=0.0)
        p = tl.load(predictions_ptr + t_offsets, mask=mask, other=0.0)
        
        log_t = tl.log(t)
        log_p = tl.log(p)
        term_i = t * (log_t - log_p)
        term_i = tl.where(t > 0, term_i, 0.0)
        term += term_i
    
    # Sum across all threads in the block
    row_kl = tl.sum(term, axis=0)
    tl.store(output_ptr + row_idx, row_kl)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, predictions, targets):
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        batch_size, n_cols = predictions.shape
        
        per_row_kl = torch.empty(batch_size, device=predictions.device, dtype=torch.float32)
        BLOCK_SIZE = 256
        ELEMENTS_PER_THREAD = 16
        
        grid = (batch_size,)
        _kl_div_rowwise[grid](
            predictions, targets, per_row_kl, 
            n_cols, BLOCK_SIZE, ELEMENTS_PER_THREAD
        )
        
        return per_row_kl.mean()

batch_size = 128
input_shape = (4096,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape).softmax(dim=-1), 
            torch.randn(batch_size, *input_shape).softmax(dim=-1)]

def get_init_inputs():
    return []
# =================== EVOLVE-BLOCK-END ===================