import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def cosine_similarity_loss_kernel(
    predictions_ptr,
    targets_ptr,
    loss_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    dot = 0.0
    norm_p = 0.0
    norm_t = 0.0
    
    # Process columns in chunks
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load data
        p = tl.load(predictions_ptr + row_start + offsets, mask=mask, other=0.0)
        t = tl.load(targets_ptr + row_start + offsets, mask=mask, other=0.0)
        
        # Update accumulators
        dot += tl.sum(p * t)
        norm_p += tl.sum(p * p)
        norm_t += tl.sum(t * t)
    
    # Compute cosine similarity with numerical stability
    norm = tl.sqrt(norm_p) * tl.sqrt(norm_t)
    cosine_sim = dot / (norm + 1e-8)
    loss_val = 1.0 - cosine_sim
    
    # Store per-row loss
    tl.store(loss_ptr + row_idx, loss_val)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, predictions, targets):
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        batch_size, n_cols = predictions.shape
        
        # Allocate output buffer
        loss_vec = torch.empty(batch_size, device=predictions.device, dtype=torch.float32)
        
        # Configure kernel launch
        BLOCK_SIZE = 1024  # Changed from 128 to 1024
        grid = (batch_size,)
        
        # Launch kernel
        cosine_similarity_loss_kernel[grid](
            predictions, targets, loss_vec,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Return mean loss
        return torch.mean(loss_vec)

batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return []