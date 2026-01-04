# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    stride_x_batch,
    stride_x_feature,
    stride_weight_hidden,
    stride_weight_feature,
    stride_bias,
    stride_out_batch,
    BLOCK_INPUT: tl.constexpr,
    input_size: tl.constexpr,
    hidden_size: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Load input row with vectorized access
    input_start = pid * stride_x_batch
    input_offsets = input_start + tl.arange(0, BLOCK_INPUT) * stride_x_feature
    mask = tl.arange(0, BLOCK_INPUT) < input_size
    x_row = tl.load(x_ptr + input_offsets, mask=mask, other=0.0)
    
    row_sum = 0.0
    
    # Unroll hidden dimension processing
    for j in tl.static_range(hidden_size):
        # Load weight row with vectorized access
        w_start = j * stride_weight_hidden
        w_offsets = w_start + tl.arange(0, BLOCK_INPUT) * stride_weight_feature
        w_row = tl.load(weight_ptr + w_offsets, mask=mask, other=0.0)
        
        # Load bias element
        b = tl.load(bias_ptr + j * stride_bias)
        
        # Compute dot product + bias
        dot = tl.sum(x_row * w_row) + b
        
        # Sigmoid activation
        s = 1.0 / (1.0 + tl.exp(-dot))
        row_sum += s
    
    # Store reduced result
    output_offset = pid * stride_out_batch
    tl.store(output_ptr + output_offset, row_sum)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        batch_size = x.shape[0]
        output = torch.empty(batch_size, 1, device=x.device, dtype=x.dtype)
        
        # Ensure tensor memory layout is contiguous
        weight = self.linear.weight.contiguous()
        bias = self.linear.bias.contiguous()
        
        # Calculate next power-of-two block size
        BLOCK_INPUT = triton.next_power_of_2(self.input_size)
        
        # Launch kernel
        grid = (batch_size,)
        _forward_kernel[grid](
            x, weight, bias, output,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            bias.stride(0),
            output.stride(0),
            BLOCK_INPUT,
            self.input_size,
            self.hidden_size,
        )
        return output

batch_size = 128
input_size = 10
hidden_size = 20

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size]
# =================== EVOLVE-BLOCK-END ===================