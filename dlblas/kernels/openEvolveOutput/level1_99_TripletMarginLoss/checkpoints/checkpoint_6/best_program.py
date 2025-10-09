# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triplet_margin_kernel(
    anchor_ptr, positive_ptr, negative_ptr, output_ptr,
    n_triplets, feature_size, margin,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= n_triplets:
        return
    
    ap_acc = 0.0
    an_acc = 0.0
    
    for offset in range(0, feature_size, BLOCK_SIZE):
        idxs = offset + tl.arange(0, BLOCK_SIZE)
        mask = idxs < feature_size
        
        a = tl.load(anchor_ptr + pid * feature_size + idxs, mask=mask, other=0.0)
        p = tl.load(positive_ptr + pid * feature_size + idxs, mask=mask, other=0.0)
        n = tl.load(negative_ptr + pid * feature_size + idxs, mask=mask, other=0.0)
        
        ap_diff = a - p
        an_diff = a - n
        
        ap_acc += tl.sum(ap_diff * ap_diff, axis=0)
        an_acc += tl.sum(an_diff * an_diff, axis=0)
    
    d_ap = tl.sqrt(ap_acc)
    d_an = tl.sqrt(an_acc)
    loss = tl.maximum(d_ap - d_an + margin, 0.0)
    tl.store(output_ptr + pid, loss)

class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        n_triplets = anchor.shape[0]
        feature_size = anchor.shape[1]
        losses = torch.empty(n_triplets, device=anchor.device)
        
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_triplets, 1),)
        
        triplet_margin_kernel[grid](
            anchor, positive, negative, losses,
            n_triplets, feature_size, self.margin,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return losses.mean()

batch_size = 128
input_shape = (4096,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), 
            torch.randn(batch_size, *input_shape), 
            torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [1.0]
# =================== EVOLVE-BLOCK-END ===================