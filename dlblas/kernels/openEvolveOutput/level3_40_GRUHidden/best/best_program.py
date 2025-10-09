# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def gru_step_kernel(
    input_proj_ptr,
    h_prev_ptr,
    weight_hh_ptr,
    bias_hh_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_HS: tl.constexpr,
):
    pid = tl.program_id(0)
    off_input = pid * 3 * hidden_size
    input_proj = tl.load(input_proj_ptr + off_input + tl.arange(0, 3 * hidden_size))
    
    off_h = pid * hidden_size
    h_prev = tl.load(h_prev_ptr + off_h + tl.arange(0, hidden_size))
    
    hidden_proj = tl.zeros((3 * hidden_size,), dtype=tl.float32)
    for i in range(0, hidden_size):
        h_i = tl.load(h_prev_ptr + off_h + i)
        col_i = tl.load(weight_hh_ptr + i + tl.arange(0, 3 * hidden_size) * hidden_size,
                        mask=tl.arange(0, 3 * hidden_size) < 3 * hidden_size, other=0.0)
        hidden_proj += col_i * h_i

    if has_bias:
        bias = tl.load(bias_hh_ptr + tl.arange(0, 3 * hidden_size),
                       mask=tl.arange(0, 3 * hidden_size) < 3 * hidden_size, other=0.0)
        hidden_proj += bias

    r = tl.sigmoid(input_proj[0:hidden_size] + hidden_proj[0:hidden_size])
    z = tl.sigmoid(input_proj[hidden_size:2*hidden_size] + hidden_proj[hidden_size:2*hidden_size])
    n = tl.tanh(input_proj[2*hidden_size:3*hidden_size] + r * hidden_proj[2*hidden_size:3*hidden_size])
    new_h = (1 - z) * n + z * h_prev
    
    tl.store(output_ptr + off_h + tl.arange(0, hidden_size), new_h)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        
        self.weight_ih = nn.ParameterList()
        self.weight_hh = nn.ParameterList()
        if bias:
            self.bias_ih = nn.ParameterList()
            self.bias_hh = nn.ParameterList()
        else:
            self.bias_ih = self.bias_hh = None

        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            stdv = 1.0 / math.sqrt(hidden_size)
            
            w_ih = nn.Parameter(torch.empty(3 * hidden_size, layer_input_size))
            w_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
            nn.init.uniform_(w_ih, -stdv, stdv)
            nn.init.uniform_(w_hh, -stdv, stdv)
            self.weight_ih.append(w_ih)
            self.weight_hh.append(w_hh)
            
            if bias:
                b_ih = nn.Parameter(torch.empty(3 * hidden_size))
                b_hh = nn.Parameter(torch.empty(3 * hidden_size))
                nn.init.uniform_(b_ih, -stdv, stdv)
                nn.init.uniform_(b_hh, -stdv, stdv)
                self.bias_ih.append(b_ih)
                self.bias_hh.append(b_hh)
        
        self.h0 = torch.randn((num_layers, batch_size, hidden_size))
    
    def forward(self, x):
        if self.batch_first:
            x = x.transpose(0, 1)
            
        batch_size = x.size(1)
        h_n = []
        current_input = x
        
        for layer in range(self.num_layers):
            weight_ih = self.weight_ih[layer]
            weight_hh = self.weight_hh[layer]
            bias_ih = self.bias_ih[layer] if self.bias else None
            bias_hh = self.bias_hh[layer] if self.bias else None
            
            # Precompute input projection
            input_proj = torch.matmul(current_input, weight_ih.t())
            if bias_ih is not None:
                input_proj += bias_ih
            
            h_prev = self.h0[layer].to(x.device)
            output_sequence = []
            
            for t in range(current_input.size(0)):
                input_proj_t = input_proj[t]
                h_next = torch.empty_like(h_prev)
                
                gru_step_kernel[(batch_size,)](
                    input_proj_t.data_ptr(),
                    h_prev.data_ptr(),
                    weight_hh.data_ptr(),
                    bias_hh.data_ptr() if bias_hh is not None else 0,
                    h_next.data_ptr(),
                    hidden_size=self.hidden_size,
                    has_bias=bias_hh is not None,
                    BLOCK_HS=triton.next_power_of_2(self.hidden_size)
                )
                
                h_prev = h_next
                output_sequence.append(h_next)
            
            current_input = torch.stack(output_sequence)
            h_n.append(h_prev)
        
        h_n = torch.stack(h_n)
        return h_n

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