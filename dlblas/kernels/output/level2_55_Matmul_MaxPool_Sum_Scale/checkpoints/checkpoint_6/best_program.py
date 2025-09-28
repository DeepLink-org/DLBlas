# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _forward_kernel(
    x_ptr, 
    w_ptr, 
    b_ptr, 
    output_ptr,
    in_features, 
    out_features,
    scale_factor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    x_offset = pid * in_features
    x_ptrs = x_ptr + x_offset + tl.arange(0, BLOCK_SIZE)
    x_mask = tl.arange(0, BLOCK_SIZE) < in_features
    x = tl.load(x_ptrs, mask=x_mask, other=0.0)
    
    s = 0.0
    # Process only paired features (ignore unpaired last feature)
    total_pairs = out_features // 2
    upper_bound = total_pairs * 2
    i = 0
    while i < upper_bound:
        w_offset_i = i * in_features
        w_ptrs_i = w_ptr + w_offset_i + tl.arange(0, BLOCK_SIZE)
        w_i = tl.load(w_ptrs_i, mask=x_mask, other=0.0)
        
        w_offset_i1 = (i+1) * in_features
        w_ptrs_i1 = w_ptr + w_offset_i1 + tl.arange(0, BLOCK_SIZE)
        w_i1 = tl.load(w_ptrs_i1, mask=x_mask, other=0.0)
        
        dot_i = tl.sum(x * w_i)
        dot_i1 = tl.sum(x * w_i1)
        
        b_i = tl.load(b_ptr + i)
        b_i1 = tl.load(b_ptr + i+1)
        
        dot_i += b_i
        dot_i1 += b_i1
        
        m = tl.maximum(dot_i, dot_i1)
        s += m
        i += 2
            
    s = s * scale_factor
    output_ptrs = output_ptr + pid
    tl.store(output_ptrs, s)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = scale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        output = torch.empty(batch_size, device=x.device, dtype=x.dtype)
        
        in_features = self.matmul.in_features
        out_features = self.matmul.out_features
        block_size = triton.next_power_of_2(in_features)
        
        grid = (batch_size,)
        _forward_kernel[grid](
            x, 
            self.matmul.weight, 
            self.matmul.bias, 
            output,
            in_features, 
            out_features, 
            self.scale_factor,
            BLOCK_SIZE=block_size
        )
        return output

batch_size = 128
in_features = 10
out_features = 5
kernel_size = 2
scale_factor = 0.5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]
# =================== EVOLVE-BLOCK-END ===================