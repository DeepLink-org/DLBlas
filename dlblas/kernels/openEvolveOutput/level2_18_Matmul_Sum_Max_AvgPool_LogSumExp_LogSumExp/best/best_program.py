# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _forward_kernel(
    x_ptr,
    w_sum_ptr,
    b_sum_ptr,
    output_ptr,
    in_features,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * in_features + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < in_features
    
    x_row = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    w_row = tl.load(w_sum_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    
    dot = tl.sum(x_row * w_row)
    b_sum = tl.load(b_sum_ptr)
    result = dot + b_sum
    tl.store(output_ptr + pid, result)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer('weight_sum', self.linear.weight.sum(dim=0))
        self.register_buffer('bias_sum', self.linear.bias.sum().view(1))

    def forward(self, x):
        weight_sum = self.weight_sum.to(device=x.device, dtype=x.dtype).contiguous()
        bias_sum = self.bias_sum.to(device=x.device, dtype=x.dtype).contiguous()
        
        x = x.contiguous()
        output = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
        
        if x.numel() > 0:
            BLOCK_SIZE = triton.next_power_of_2(x.shape[1])
            grid = (x.shape[0],)
            
            _forward_kernel[grid](
                x, 
                weight_sum,
                bias_sum,
                output,
                x.shape[1],
                BLOCK_SIZE
            )
        
        return output.unsqueeze(1)

batch_size = 128
in_features = 10
out_features = 5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]
# =================== EVOLVE-BLOCK-END ===================