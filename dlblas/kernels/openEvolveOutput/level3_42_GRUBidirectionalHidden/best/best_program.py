# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def gru_forward_kernel(
    # Input tensors
    input_ptr,
    weight_ih_ptr,
    weight_hh_ptr,
    bias_ih_ptr,
    bias_hh_ptr,
    hx_ptr,
    output_ptr,
    # Tensor dimensions
    seq_len,
    input_size,
    hidden_size,
    # Strides
    stride_input_batch,
    stride_input_seq,
    stride_input_feature,
    stride_weight_ih_feature,
    stride_weight_ih_hidden,
    stride_weight_hh_feature,
    stride_weight_hh_hidden,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # Batch index
    pid_t = tl.program_id(1)  # Time step index

    # Initialize pointers
    h_ptr = hx_ptr + pid * hidden_size
    input_start = input_ptr + pid * stride_input_batch + pid_t * stride_input_seq
    output_start = output_ptr + pid * stride_input_batch * 3 + pid_t * stride_input_seq * 3
    
    # Initialize hidden state
    h_prev = tl.zeros((hidden_size,), dtype=tl.float32)
    if pid_t == 0:
        h_prev = tl.load(h_ptr + tl.arange(0, HIDDEN_BLOCK), mask=tl.arange(0, HIDDEN_BLOCK) < hidden_size, other=0.0)
    
    # Load current input
    input_vals = tl.load(input_start + tl.arange(0, BLOCK_SIZE) * stride_input_feature, 
                         mask=tl.arange(0, BLOCK_SIZE) < input_size, other=0.0)
    
    # Compute gates
    for i in range(0, 3 * hidden_size, HIDDEN_BLOCK):
        idx = i + tl.arange(0, HIDDEN_BLOCK)
        mask = idx < 3 * hidden_size
        
        # Weight-ih part
        wih_ptr = weight_ih_ptr + idx * stride_weight_ih_feature
        ih_vals = tl.sum(input_vals[None, :] * tl.load(wih_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_weight_ih_hidden, 
                                                      mask=mask & (tl.arange(0, BLOCK_SIZE)[:, None] < input_size), 
                                                      other=0.0), axis=0)
        
        # Weight-hh part
        whh_ptr = weight_hh_ptr + idx * stride_weight_hh_feature
        hh_vals = tl.sum(h_prev[None, :] * tl.load(whh_ptr + tl.arange(0, HIDDEN_BLOCK)[:, None] * stride_weight_hh_hidden, 
                                                  mask=mask & (tl.arange(0, HIDDEN_BLOCK)[:, None] < hidden_size), 
                                                  other=0.0), axis=0)
        
        # Add biases
        if bias_ih_ptr is not None:
            ih_vals += tl.load(bias_ih_ptr + idx, mask=mask, other=0.0)
        if bias_hh_ptr is not None:
            hh_vals += tl.load(bias_hh_ptr + idx, mask=mask, other=0.0)
        
        # Gate calculations
        if i < hidden_size:  # Update gate
            z = tl.sigmoid(ih_vals + hh_vals)
        elif i < 2 * hidden_size:  # Reset gate
            r = tl.sigmoid(ih_vals + hh_vals)
        else:  # New gate
            n = tl.tanh(ih_vals + r * hh_vals)
    
    # Compute new hidden state
    h_new = (1 - z) * n + z * h_prev
    
    # Store results
    tl.store(output_start + tl.arange(0, HIDDEN_BLOCK), h_new, mask=tl.arange(0, HIDDEN_BLOCK) < hidden_size)
    
    # Update hidden state for next time step
    if pid_t < seq_len - 1:
        tl.store(h_ptr + tl.arange(0, HIDDEN_BLOCK), h_new, mask=tl.arange(0, HIDDEN_BLOCK) < hidden_size)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        
        # Initialize weights for each layer and direction
        self.weights_ih = nn.ParameterList()
        self.weights_hh = nn.ParameterList()
        self.biases_ih = nn.ParameterList()
        self.biases_hh = nn.ParameterList()
        
        for _ in range(num_layers * 2):  # *2 for bidirectional
            self.weights_ih.append(nn.Parameter(torch.empty(3 * hidden_size, input_size)))
            self.weights_hh.append(nn.Parameter(torch.empty(3 * hidden_size, hidden_size)))
            if bias:
                self.biases_ih.append(nn.Parameter(torch.empty(3 * hidden_size)))
                self.biases_hh.append(nn.Parameter(torch.empty(3 * hidden_size)))
            else:
                self.biases_ih.append(None)
                self.biases_hh.append(None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for weight in self.weights_ih:
            nn.init.xavier_uniform_(weight)
        for weight in self.weights_hh:
            nn.init.orthogonal_(weight)
        for bias in self.biases_ih:
            if bias is not None:
                nn.init.zeros_(bias)
        for bias in self.biases_hh:
            if bias is not None:
                nn.init.zeros_(bias)
    
    def forward(self, x):
        if self.batch_first:
            x = x.transpose(0, 1)
        
        seq_len, batch_size, _ = x.shape
        device = x.device
        
        # Initialize hidden states
        hx = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=device, dtype=x.dtype)
        output = torch.zeros(seq_len, batch_size, self.hidden_size * 2, device=device, dtype=x.dtype)
        
        # Process each layer
        for layer_idx in range(self.num_layers):
            # Process forward direction
            self.process_direction(
                x, hx, output, 
                layer_idx * 2, 
                self.weights_ih[layer_idx * 2],
                self.weights_hh[layer_idx * 2],
                self.biases_ih[layer_idx * 2] if self.bias else None,
                self.biases_hh[layer_idx * 2] if self.bias else None,
                direction=1
            )
            
            # Process backward direction
            self.process_direction(
                x, hx, output, 
                layer_idx * 2 + 1, 
                self.weights_ih[layer_idx * 2 + 1],
                self.weights_hh[layer_idx * 2 + 1],
                self.biases_ih[layer_idx * 2 + 1] if self.bias else None,
                self.biases_hh[layer_idx * 2 + 1] if self.bias else None,
                direction=-1
            )
            
            # Update input for next layer
            x = output
        
        return hx
    
    def process_direction(self, x, hx, output, layer_idx, weight_ih, weight_hh, bias_ih, bias_hh, direction):
        seq_len, batch_size, input_size = x.shape
        hidden_size = self.hidden_size
        
        # Configure kernel
        BLOCK_SIZE = triton.next_power_of_2(input_size)
        HIDDEN_BLOCK = triton.next_power_of_2(hidden_size)
        
        # Adjust for direction
        if direction == -1:
            x = x.flip(0)
        
        # Launch kernel
        grid = (batch_size, seq_len)
        gru_forward_kernel[grid](
            x, weight_ih, weight_hh, bias_ih, bias_hh, 
            hx[layer_idx], output[:, :, layer_idx * hidden_size: (layer_idx + 1) * hidden_size],
            seq_len, input_size, hidden_size,
            x.stride(1), x.stride(0), x.stride(2),
            weight_ih.stride(1), weight_ih.stride(0),
            weight_hh.stride(1), weight_hh.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            HIDDEN_BLOCK=HIDDEN_BLOCK
        )
        
        # Restore original order for backward direction
        if direction == -1:
            output[:, :, layer_idx * hidden_size: (layer_idx + 1) * hidden_size] = output[:, :, layer_idx * hidden_size: (layer_idx + 1) * hidden_size].flip(0)

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