# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def cosine_similarity_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    row_length: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * row_length
    
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    
    # Process entire row in chunks
    for offset in range(0, row_length, BLOCK_SIZE):
        offs = offset + tl.arange(0, BLOCK_SIZE)
        mask = offs < row_length
        
        vec1 = tl.load(predictions_ptr + row_start + offs, mask=mask, other=0.0)
        vec2 = tl.load(targets_ptr + row_start + offs, mask=mask, other=0.0)
        
        dot += tl.sum(vec1 * vec2)
        norm1 += tl.sum(vec1 * vec1)
        norm2 += tl.sum(vec2 * vec2)
    
    # Compute cosine similarity with numerical stability
    norm = tl.sqrt(norm1) * tl.sqrt(norm2) + eps
    cos_sim = dot / norm
    loss_val = 1.0 - cos_sim
    tl.store(output_ptr + pid, loss_val)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        
    def forward(self, predictions, targets):
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        
        batch_size = predictions.shape[0]
        loss_per_row = torch.empty(batch_size, device=predictions.device, dtype=torch.float32)
        
        # Launch kernel with optimized block size
        grid = (batch_size,)
        BLOCK_SIZE = 1024  # Optimized for 4096-element vectors
        cosine_similarity_kernel[grid](
            predictions, targets, loss_per_row,
            row_length=predictions.shape[1],
            eps=1e-8,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return torch.mean(loss_per_row)

batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return []

# =================== EVOLVE-BLOCK-END ===================