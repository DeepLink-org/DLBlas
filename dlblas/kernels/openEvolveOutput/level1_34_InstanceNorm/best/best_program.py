# ================== EVOLVE-BLOCK-START ==================
import torch
import triton
import triton.language as tl
import math

@triton.jit
def _inorm_kernel(
    x_ptr,
    output_ptr,
    stride_b,
    stride_f,
    stride_h,
    stride_w,
    H,
    W,
    eps,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_f = tl.program_id(1)
    
    x_channel_ptr = x_ptr + pid_b * stride_b + pid_f * stride_f
    output_channel_ptr = output_ptr + pid_b * stride_b + pid_f * stride_f
    
    sum_val = 0.0
    sum_sq = 0.0
    n_elements = 0
    
    for h in range(0, H):
        for w_block in range(0, tl.cdiv(W, BLOCK_SIZE)):
            w_offsets = w_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            w_mask = w_offsets < W
            
            ptr = x_channel_ptr + h * stride_h + w_offsets * stride_w
            vals = tl.load(ptr, mask=w_mask, other=0.0)
            
            block_sum = tl.sum(vals, axis=0)
            block_sum_sq = tl.sum(vals * vals, axis=0)
            block_count = tl.sum(tl.where(w_mask, 1, 0), axis=0)
            
            sum_val += block_sum
            sum_sq += block_sum_sq
            n_elements += block_count
    
    mean = sum_val / n_elements
    variance = (sum_sq / n_elements) - (mean * mean)
    std = tl.sqrt(variance + eps)
    
    for h in range(0, H):
        for w_block in range(0, tl.cdiv(W, BLOCK_SIZE)):
            w_offsets = w_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            w_mask = w_offsets < W
            
            ptr = x_channel_ptr + h * stride_h + w_offsets * stride_w
            vals = tl.load(ptr, mask=w_mask, other=0.0)
            
            norm_vals = (vals - mean) / std
            
            out_ptr = output_channel_ptr + h * stride_h + w_offsets * stride_w
            tl.store(out_ptr, norm_vals, mask=w_mask)

class ModelNew(torch.nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.eps = 1e-5
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, features, H, W = x.shape
        output = torch.empty_like(x)
        
        BLOCK_SIZE = 128
        P2 = int(2 ** math.ceil(math.log2(W)))
        grid = (batch, features)
        
        _inorm_kernel[grid](
            x, output,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            H, W, self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
            P2=P2
        )
        return output

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]
# =================== EVOLVE-BLOCK-END ===================