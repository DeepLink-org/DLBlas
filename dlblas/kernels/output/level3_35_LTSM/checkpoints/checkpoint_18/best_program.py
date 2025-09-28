# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel for optimized linear layer
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, output_ptr,
    in_features, out_features,
    stride_xb, stride_xf,
    stride_wo, stride_wi,
    stride_outb, stride_outf,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)  # Batch index
    pid_o = tl.program_id(1)  # Output feature index
    
    # Create feature index range
    idx = tl.arange(0, BLOCK_SIZE)
    
    # Calculate pointers and load data with masking
    x_offset = pid_b * stride_xb + idx * stride_xf
    x_mask = idx < in_features
    x = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)
    
    w_offset = pid_o * stride_wo + idx * stride_wi
    w_mask = idx < in_features
    w = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)
    
    # Compute dot product
    dot = tl.sum(x * w)
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_o)
        dot += bias
    
    # Store result
    output_offset = pid_b * stride_outb + pid_o * stride_outf
    tl.store(output_ptr + output_offset, dot)

# Optimized linear layer using Triton
class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        if x.dim() != 2:
            raise ValueError("Input must be 2D")
        
        batch_size, in_features = x.shape
        out_features = self.weight.shape[0]
        
        # Ensure contiguous memory
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias.contiguous() if self.bias is not None else None
        
        # Create output tensor
        output = torch.empty(batch_size, out_features, 
                             device=x.device, dtype=x.dtype)
        
        # Compute block size for kernel
        block_size = triton.next_power_of_2(in_features)
        
        # Launch kernel with 2D grid
        grid = (batch_size, out_features)
        linear_kernel[grid](
            x, weight, bias, output,
            in_features, out_features,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_SIZE=block_size
        )
        return output

# Optimized LSTM model
class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initialize weights directly on device in forward pass
        self.h0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size))
        self.c0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size))
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout, 
            bidirectional=False
        )
        # Use Triton-optimized linear layer
        self.fc = TritonLinear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Expand initial states to match batch size
        h0 = self.h0.expand(-1, batch_size, -1).contiguous()
        c0 = self.c0.expand(-1, batch_size, -1).contiguous()
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Apply optimized linear layer to last timestep
        return self.fc(out[:, -1, :])

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