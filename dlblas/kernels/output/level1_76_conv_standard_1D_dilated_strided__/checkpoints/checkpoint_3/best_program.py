# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def _conv1d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    length,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    dilation: tl.constexpr,
    L_out,
    HAS_BIAS: tl.constexpr,
    BLOCK_IN: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_o = tl.program_id(2)
    
    acc = 0.0
    base_idx = pid_o * stride
    
    # Process input channels in blocks
    for in_c in range(0, in_channels, BLOCK_IN):
        # Create channel offsets with power-of-two size
        in_c_offsets = tl.arange(0, BLOCK_IN)
        channel_mask = in_c_offsets < (in_channels - in_c)
        
        # Initialize accumulator for this block
        block_acc = 0.0
        
        # Unroll kernel loop for better performance
        for k in tl.static_range(kernel_size):
            idx = base_idx + k * dilation
            pos_ok = idx < length
            
            # Coalesced memory access for input
            x_offsets = pid_b * in_channels * length + (in_c + in_c_offsets) * length + idx
            x_vals = tl.load(
                x_ptr + x_offsets, 
                mask=channel_mask & pos_ok, 
                other=0.0
            )
            
            # Coalesced memory access for weights
            w_offsets = pid_c * in_channels * kernel_size + (in_c + in_c_offsets) * kernel_size + k
            w_vals = tl.load(
                weight_ptr + w_offsets, 
                mask=channel_mask, 
                other=0.0
            )
            
            block_acc += tl.sum(x_vals * w_vals)
        
        acc += block_acc
    
    # Add bias if present
    if HAS_BIAS:
        bias_val = tl.load(bias_ptr + pid_c)
        acc += bias_val
        
    # Store result with coalesced access
    output_offset = pid_b * out_channels * L_out + pid_c * L_out + pid_o
    tl.store(output_ptr + output_offset, acc)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        # Initialize weights
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Reset parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, length = x.shape
        L_out = (length - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        
        # Preallocate output tensor
        output = torch.empty((batch_size, self.out_channels, L_out), 
                             device=x.device, dtype=x.dtype)
        
        # Determine grid size
        grid = (batch_size, self.out_channels, L_out)
        
        # Kernel parameters - increased block size for better vectorization
        HAS_BIAS = self.bias is not None
        BLOCK_IN = 32  # Increased from 16 to 32 for better vectorization
        
        # Launch kernel
        _conv1d_kernel[grid](
            x, self.weight, self.bias, output,
            batch_size, self.in_channels, self.out_channels, 
            length, self.kernel_size, self.stride, self.dilation, L_out,
            HAS_BIAS, BLOCK_IN
        )
        
        return output

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
length = 256
stride = 3
dilation = 4

def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, dilation]
# =================== EVOLVE-BLOCK-END ===================