# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def rnn_i2h_kernel(
    x_ptr, hidden_ptr, weight_ptr, bias_ptr, out_ptr,
    input_size, hidden_size,
    BLOCK_K: tl.constexpr
):
    pid0 = tl.program_id(0)  # Batch index
    pid1 = tl.program_id(1)  # Hidden feature index
    
    # Base pointers for current batch
    x_row_ptr = x_ptr + pid0 * input_size
    hidden_row_ptr = hidden_ptr + pid0 * hidden_size
    out_offset = pid0 * hidden_size + pid1
    
    # Accumulators
    sum1 = 0.0
    sum2 = 0.0
    offs_k = tl.arange(0, BLOCK_K)
    
    # Process input part (x)
    for k in range(0, input_size, BLOCK_K):
        mask = k + offs_k < input_size
        x_vals = tl.load(x_row_ptr + k + offs_k, mask=mask, other=0.0)
        w_vals = tl.load(weight_ptr + pid1 * (input_size + hidden_size) + k + offs_k, mask=mask, other=0.0)
        sum1 += tl.sum(x_vals * w_vals)
    
    # Process hidden part
    for k in range(0, hidden_size, BLOCK_K):
        w_offset = input_size + k
        mask = k + offs_k < hidden_size
        h_vals = tl.load(hidden_row_ptr + k + offs_k, mask=mask, other=0.0)
        w_vals = tl.load(weight_ptr + pid1 * (input_size + hidden_size) + w_offset + offs_k, mask=mask, other=0.0)
        sum2 += tl.sum(h_vals * w_vals)
    
    # Final computation
    bias_val = tl.load(bias_ptr + pid1)
    total = sum1 + sum2 + bias_val
    
    # Stable tanh implementation
    abs_total = tl.abs(total)
    t = tl.exp(-2 * abs_total)
    base = (1 - t) / (1 + t)
    output_val = tl.where(total >= 0, base, -base)
    
    tl.store(out_ptr + out_offset, output_val)

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden = torch.randn((batch_size, hidden_size))
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.hidden = self.hidden.to(x.device)
        new_hidden = torch.empty_like(self.hidden)
        
        grid = (batch_size, self.hidden_size)
        rnn_i2h_kernel[grid](
            x, self.hidden, 
            self.i2h.weight, self.i2h.bias, 
            new_hidden,
            self.input_size, self.hidden_size,
            BLOCK_K=128
        )
        
        self.hidden = new_hidden
        return self.hidden

batch_size = 8
input_size = 1024
hidden_size = 256
output_size = 128
sequence_length = 256

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, output_size]
# =================== EVOLVE-BLOCK-END ===================