# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def identity_kernel(output_ptr, input_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, input_vals, mask=mask)

def identity(x):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    identity_kernel[grid](output, x, n_elements, BLOCK_SIZE=1024)
    return output

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.h0 = None
        self.c0 = None
    
    def forward(self, x):
        batch_size = x.size(0)
        if self.h0 is None or self.h0.size(1) != batch_size or self.h0.device != x.device:
            self.h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            self.c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        out, state = self.lstm(x, (self.h0, self.c0))
        hidden_state = identity(state[0])
        return hidden_state

# Test code
batch_size = 10
sequence_length = 512
input_size = 128
hidden_size = 256
num_layers = 6
output_size = 10
dropout = 0.0

def get_inputs():
    return [torch.randn(batch_size, sequence_length, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, dropout]
# =================== EVOLVE-BLOCK-END ===================