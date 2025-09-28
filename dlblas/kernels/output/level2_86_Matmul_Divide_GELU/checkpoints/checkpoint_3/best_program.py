# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64, 'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128, 'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['input_size', 'output_size'],
)
@triton.jit
def _fused_linear_div_gelu(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    input_size, output_size, divisor,
    batch_size, stride_x, stride_out,
    BLOCK_SIZE: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Compute program IDs
    pid_batch = tl.program_id(0)
    pid_output_block = tl.program_id(1)
    
    # Create output block offset
    output_offset = pid_output_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = output_offset < output_size
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over input features in blocks
    for k in range(0, input_size, BLOCK_K):
        k_offset = k + tl.arange(0, BLOCK_K)
        k_mask = k_offset < input_size
        
        # Load input block
        x_ptr_offset = pid_batch * stride_x + k_offset
        x_block = tl.load(x_ptr + x_ptr_offset, mask=k_mask, other=0.0)
        
        # Load weight block
        w_ptr_offset = output_offset[:, None] * input_size + k_offset[None, :]
        w_block = tl.load(weight_ptr + w_ptr_offset, mask=output_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Compute partial dot product
        acc += tl.sum(x_block[None, :] * w_block, axis=1)
    
    # Load bias if exists
    if HAS_BIAS:
        bias_block = tl.load(bias_ptr + output_offset, mask=output_mask, other=0.0)
        acc += bias_block
    
    # Apply division and approximate GELU
    scaled = acc / divisor
    gelu = scaled * 0.5 * (1.0 + tl.erf(scaled * 0.7071067811865475))
    
    # Store output
    out_ptr_offset = pid_batch * stride_out + output_offset
    tl.store(output_ptr + out_ptr_offset, gelu, mask=output_mask)

class ModelNew(nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.divisor = divisor
        
        # Initialize parameters as in nn.Linear
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_size, _ = x.shape
        output = torch.empty((batch_size, self.output_size), 
                             device=x.device, dtype=x.dtype)
        
        # Determine grid dimensions
        grid = (batch_size, triton.cdiv(self.output_size, 128))
        
        # Launch kernel
        _fused_linear_div_gelu[grid](
            x, self.weight, self.bias, output,
            self.input_size, self.output_size, self.divisor,
            batch_size, x.stride(0), output.stride(0),
            HAS_BIAS=True
        )
        return output

batch_size = 128
input_size = 512
output_size = 1024
divisor = 10.0

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, output_size, divisor]
# =================== EVOLVE-BLOCK-END ===================