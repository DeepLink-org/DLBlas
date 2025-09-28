# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fused_linear_sigmoid_sum(
    x_ptr, w_ptr, b_ptr, output_ptr, 
    batch_size, input_size, hidden_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    
    row_sum = 0.0
    for j in range(0, hidden_size):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < input_size
        
        x_i = tl.load(x_ptr + pid * input_size + offsets, mask=mask, other=0.0)
        w_j = tl.load(w_ptr + j * input_size + offsets, mask=mask, other=0.0)
        
        dot = tl.sum(x_i * w_j)
        bias_j = tl.load(b_ptr + j)
        dot += bias_j
        s = 1.0 / (1.0 + tl.exp(-dot))
        row_sum += s
    
    tl.store(output_ptr + pid, row_sum)

@triton.jit
def _logsumexp(
    input_ptr, output_ptr, n, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= 1:
        return
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(input_ptr + offsets, mask=mask, other=-3e38)
    
    max_val = tl.max(x, axis=0)
    exp_x = tl.exp(x - max_val)
    sum_exp = tl.sum(exp_x, axis=0)
    log_sum_exp = tl.log(sum_exp) + max_val
    
    tl.store(output_ptr, log_sum_exp)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size, input_size = x.shape
        hidden_size = self.linear1.weight.shape[0]
        
        x = x.contiguous()
        w = self.linear1.weight.contiguous()
        b = self.linear1.bias.contiguous()
        
        output_stage1 = torch.empty(batch_size, device=x.device, dtype=torch.float32)
        grid1 = (batch_size,)
        BLOCK_SIZE_INPUT = triton.next_power_of_2(input_size)
        _fused_linear_sigmoid_sum[grid1](
            x, w, b, output_stage1, 
            batch_size, input_size, hidden_size,
            BLOCK_SIZE=BLOCK_SIZE_INPUT
        )
        
        output_stage2 = torch.empty(1, device=x.device, dtype=torch.float32)
        grid2 = (1,)
        BLOCK_SIZE_BATCH = triton.next_power_of_2(batch_size)
        _logsumexp[grid2](
            output_stage1, output_stage2, batch_size, 
            BLOCK_SIZE=BLOCK_SIZE_BATCH
        )
        
        return output_stage2[0]

batch_size = 128
input_size = 10
hidden_size = 20
output_size = 5

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, output_size]
# =================== EVOLVE-BLOCK-END ===================