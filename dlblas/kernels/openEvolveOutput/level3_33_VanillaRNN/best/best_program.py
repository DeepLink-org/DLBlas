# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def fused_rnn_kernel(
    x_ptr, hidden_ptr, w1_ptr, w2_ptr, bias_ptr, output_hidden_ptr,
    input_size, hidden_size, batch_size,
    BLOCK_HIDDEN: tl.constexpr, BLOCK_INPUT: tl.constexpr
):
    # Compute program IDs
    pid_batch = tl.program_id(0)
    pid_col = tl.program_id(1)
    
    # Create ranges
    col_offsets = pid_col * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    col_mask = col_offsets < hidden_size
    
    # Initialize accumulators
    acc1 = tl.zeros((BLOCK_HIDDEN,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_HIDDEN,), dtype=tl.float32)
    
    # Loop over input dimension (x part)
    for k in range(0, tl.cdiv(input_size, BLOCK_INPUT)):
        k_offsets = k * BLOCK_INPUT + tl.arange(0, BLOCK_INPUT)
        k_mask = k_offsets < input_size
        
        # Load x block
        x_offs = pid_batch * input_size + k_offsets
        x_val = tl.load(x_ptr + x_offs, mask=k_mask, other=0.0)
        
        # Load w1 block
        w1_offs = col_offsets[:, None] * input_size + k_offsets[None, :]
        w1_val = tl.load(w1_ptr + w1_offs, mask=col_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Accumulate
        acc1 += tl.sum(x_val[None, :] * w1_val, axis=1)
    
    # Loop over hidden dimension (hidden part)
    for k in range(0, tl.cdiv(hidden_size, BLOCK_INPUT)):
        k_offsets = k * BLOCK_INPUT + tl.arange(0, BLOCK_INPUT)
        k_mask = k_offsets < hidden_size
        
        # Load hidden block
        h_offs = pid_batch * hidden_size + k_offsets
        h_val = tl.load(hidden_ptr + h_offs, mask=k_mask, other=0.0)
        
        # Load w2 block
        w2_offs = col_offsets[:, None] * hidden_size + k_offsets[None, :]
        w2_val = tl.load(w2_ptr + w2_offs, mask=col_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Accumulate
        acc2 += tl.sum(h_val[None, :] * w2_val, axis=1)
    
    # Load bias and compute total
    bias_val = tl.load(bias_ptr + col_offsets, mask=col_mask, other=0.0)
    total = acc1 + acc2 + bias_val
    new_hidden = libdevice.tanh(total)
    
    # Store result
    tl.store(output_hidden_ptr + pid_batch * hidden_size + col_offsets, 
             new_hidden, mask=col_mask)

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
        
        # Split weights for fused kernel
        W1 = self.i2h.weight[:, :self.input_size].contiguous()
        W2 = self.i2h.weight[:, self.input_size:].contiguous()
        bias = self.i2h.bias.contiguous()
        
        # Prepare new hidden tensor
        new_hidden = torch.empty_like(self.hidden)
        
        # Launch kernel
        BLOCK_HIDDEN = 128
        BLOCK_INPUT = 32
        grid = (batch_size, triton.cdiv(self.hidden_size, BLOCK_HIDDEN))
        
        fused_rnn_kernel[grid](
            x, self.hidden, W1, W2, bias, new_hidden,
            self.input_size, self.hidden_size, batch_size,
            BLOCK_HIDDEN=BLOCK_HIDDEN, BLOCK_INPUT=BLOCK_INPUT
        )
        
        self.hidden = new_hidden
        output = self.h2o(self.hidden)
        return output

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