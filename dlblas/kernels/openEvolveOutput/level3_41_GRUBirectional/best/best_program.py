# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def gru_cell_kernel(
    # Pointers to input tensors
    x_ptr, h_prev_ptr,
    # Weight matrices
    w_ih_ptr, w_hh_ptr,
    # Output tensor
    h_new_ptr,
    # Tensor dimensions and strides
    input_size, hidden_size, 
    x_batch_stride, x_feature_stride,
    h_batch_stride, h_feature_stride,
    w_ih_row_stride, w_ih_col_stride,
    w_hh_row_stride, w_hh_col_stride,
    # Blocking parameters
    BLOCK_SIZE: tl.constexpr,
    # Meta-parameters
    ACTIVATION: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid
    num_pids = tl.num_programs(0)
    
    if batch_idx >= num_pids:
        return

    # Initialize pointers for current batch
    x_start_ptr = x_ptr + batch_idx * x_batch_stride
    h_prev_start_ptr = h_prev_ptr + batch_idx * h_batch_stride
    h_new_start_ptr = h_new_ptr + batch_idx * h_batch_stride

    # Compute input and hidden contributions
    input_contrib = tl.zeros((3 * hidden_size,), dtype=tl.float32)
    hidden_contrib = tl.zeros((3 * hidden_size,), dtype=tl.float32)
    
    # Compute input contribution
    for i in range(0, input_size, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < input_size
        x_val = tl.load(x_start_ptr + cols * x_feature_stride, mask=mask, other=0.0)
        for j in tl.static_range(3):
            gate_offset = j * hidden_size
            for k in tl.static_range(BLOCK_SIZE):
                if cols[k] < input_size:
                    w_ptr = w_ih_ptr + (gate_offset + tl.arange(0, hidden_size))[:, None] * w_ih_row_stride + cols[k] * w_ih_col_stride
                    w_val = tl.load(w_ptr, mask=(gate_offset + tl.arange(0, hidden_size))[:, None] < 3*hidden_size, other=0.0)
                    input_contrib = input_contrib + w_val * x_val[k]

    # Compute hidden contribution
    for i in range(0, hidden_size, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        h_val = tl.load(h_prev_start_ptr + cols * h_feature_stride, mask=mask, other=0.0)
        for j in tl.static_range(3):
            gate_offset = j * hidden_size
            for k in tl.static_range(BLOCK_SIZE):
                if cols[k] < hidden_size:
                    w_ptr = w_hh_ptr + (gate_offset + tl.arange(0, hidden_size))[:, None] * w_hh_row_stride + cols[k] * w_hh_col_stride
                    w_val = tl.load(w_ptr, mask=(gate_offset + tl.arange(0, hidden_size))[:, None] < 3*hidden_size, other=0.0)
                    hidden_contrib = hidden_contrib + w_val * h_val[k]

    # Split into gates
    r_in, z_in, n_in = input_contrib[0:hidden_size], input_contrib[hidden_size:2*hidden_size], input_contrib[2*hidden_size:3*hidden_size]
    r_h, z_h, n_h = hidden_contrib[0:hidden_size], hidden_contrib[hidden_size:2*hidden_size], hidden_contrib[2*hidden_size:3*hidden_size]

    # Compute gates
    r = tl.sigmoid(r_in + r_h)
    z = tl.sigmoid(z_in + z_h)
    n = tl.tanh(n_in + r * n_h)
    h_new = (1 - z) * n + z * tl.load(h_prev_start_ptr + tl.arange(0, hidden_size) * h_feature_stride)
    
    # Store new hidden state
    tl.store(h_new_start_ptr + tl.arange(0, hidden_size) * h_feature_stride, h_new)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        
        # Initialize weights for each layer and direction
        self.weight_ih = nn.ParameterList()
        self.weight_hh = nn.ParameterList()
        
        for _ in range(num_layers * 2):  # *2 for bidirectional
            # Input to hidden weights
            w_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
            # Hidden to hidden weights
            w_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
            
            # Initialize weights
            nn.init.xavier_uniform_(w_ih)
            nn.init.orthogonal_(w_hh)
            
            self.weight_ih.append(w_ih)
            self.weight_hh.append(w_hh)
            
            # For next layer, input is hidden_size*2 (bidirectional)
            input_size = hidden_size * 2

    def forward(self, x):
        if self.batch_first:
            x = x.transpose(0, 1)  # Convert to (seq_len, batch, features)
        
        seq_len, batch_size, _ = x.shape
        device = x.device
        
        # Initialize hidden states
        h = torch.zeros((self.num_layers * 2, batch_size, self.hidden_size), device=device)
        layer_input = x
        outputs = []
        
        for layer in range(self.num_layers):
            layer_outputs = []
            for direction in range(2):
                # Get weights for this layer/direction
                idx = layer * 2 + direction
                w_ih = self.weight_ih[idx]
                w_hh = self.weight_hh[idx]
                
                # Process sequence
                h_prev = h[idx]
                h_new = torch.zeros((seq_len, batch_size, self.hidden_size), device=device)
                
                for t in range(seq_len):
                    time_step = t if direction == 0 else seq_len - 1 - t
                    x_t = layer_input[time_step]
                    
                    # Launch Triton kernel
                    grid = (batch_size,)
                    gru_cell_kernel[grid](
                        x_t, h_prev,
                        w_ih, w_hh,
                        h_new[time_step],
                        self.input_size, self.hidden_size,
                        x_t.stride(0), x_t.stride(1),
                        h_prev.stride(0), h_prev.stride(1),
                        w_ih.stride(0), w_ih.stride(1),
                        w_hh.stride(0), w_hh.stride(1),
                        BLOCK_SIZE=32,
                        ACTIVATION=1
                    )
                    h_prev = h_new[time_step].clone()
                
                layer_outputs.append(h_new)
            
            # Concatenate directions
            forward_out, backward_out = layer_outputs
            if backward_out is not None:
                backward_out = backward_out.flip(0)
            layer_output = torch.cat((forward_out, backward_out), dim=-1)
            outputs.append(layer_output)
            layer_input = layer_output
        
        # Stack outputs from all layers
        output = torch.cat([o.unsqueeze(0) for o in outputs], dim=0)
        output = output[-1]  # Only return last layer output
        
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