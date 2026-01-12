# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def gru_forward_kernel(
    # Input tensors
    input_ptr,
    hidden_ptr,
    w_ih_ptr,
    w_hh_ptr,
    b_ih_ptr,
    b_hh_ptr,
    output_ptr,
    final_hidden_ptr,
    # Dimensions
    seq_len,
    batch_size,
    input_size,
    hidden_size,
    # Strides
    stride_input_batch,
    stride_input_seq,
    stride_input_feat,
    stride_hidden_layer,
    stride_hidden_batch,
    stride_hidden_feat,
    # Tensor core dimensions
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // hidden_size
    feat_idx = pid % hidden_size

    # Boundary check
    if batch_idx >= batch_size or feat_idx >= hidden_size:
        return

    # Initialize current hidden state
    h_prev = tl.load(hidden_ptr + batch_idx * stride_hidden_batch + feat_idx)

    # Pointers to weight matrices
    w_ih_ptr += feat_idx * input_size
    w_hh_ptr += feat_idx * hidden_size

    # Accumulators for gates
    reset_gate = 0.0
    update_gate = 0.0
    new_gate = 0.0

    # Compute input part
    for k in range(0, input_size, BLOCK_SIZE):
        k_offs = k + tl.arange(0, BLOCK_SIZE)
        mask = k_offs < input_size
        
        # Load input weights and features
        w_ih = tl.load(w_ih_ptr + k_offs, mask=mask, other=0.0)
        input_val = tl.load(
            input_ptr + batch_idx * stride_input_batch + k_offs,
            mask=mask, other=0.0
        )
        reset_gate += tl.sum(w_ih * input_val)

    # Compute hidden part
    for k in range(0, hidden_size, BLOCK_SIZE):
        k_offs = k + tl.arange(0, BLOCK_SIZE)
        mask = k_offs < hidden_size
        
        # Load hidden weights and features
        w_hh = tl.load(w_hh_ptr + k_offs, mask=mask, other=0.0)
        hidden_val = tl.load(
            hidden_ptr + batch_idx * stride_hidden_batch + k_offs,
            mask=mask, other=0.0
        )
        reset_gate += tl.sum(w_hh * hidden_val)

    # Apply activation functions
    reset_gate = 1.0 / (1.0 + tl.exp(-reset_gate))
    update_gate = 1.0 / (1.0 + tl.exp(-update_gate))
    new_gate = tl.tanh(new_gate)

    # Update hidden state
    h_next = (1 - update_gate) * new_gate + update_gate * h_prev
    tl.store(final_hidden_ptr + batch_idx * stride_hidden_batch + feat_idx, h_next)
    tl.store(output_ptr + batch_idx * stride_input_batch + feat_idx, h_next)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        
        # Initialize weights
        self.w_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.w_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
        
        # Initialize hidden state
        self.h0 = torch.randn((num_layers, 10, hidden_size))
        nn.init.xavier_uniform_(self.w_ih)
        nn.init.xavier_uniform_(self.w_hh)
        if bias:
            nn.init.zeros_(self.b_ih)
            nn.init.zeros_(self.b_hh)
    
    def forward(self, x):
        self.h0 = self.h0.to(x.device)
        
        # Handle batch_first option
        if self.batch_first:
            x = x.transpose(0, 1)
        
        # Initialize output tensor
        output = torch.zeros_like(x)
        
        # Triton kernel configuration
        grid = lambda meta: (x.size(1) * self.hidden_size,)
        
        # Execute GRU kernel
        gru_forward_kernel[grid](
            x, self.h0, self.w_ih, self.w_hh, 
            self.b_ih, self.b_hh, output, self.h0,
            x.size(0), x.size(1), self.input_size, self.hidden_size,
            x.stride(1), x.stride(0), x.stride(2),
            self.h0.stride(0), self.h0.stride(1), self.h0.stride(2),
            BLOCK_SIZE=32
        )
        
        # Restore batch dimension if needed
        if self.batch_first:
            output = output.transpose(0, 1)
        
        return output

# Test code
batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

def get_inputs():
    return [torch.randn(seq_len, batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, num_layers]
# =================== EVOLVE-BLOCK-END ===================