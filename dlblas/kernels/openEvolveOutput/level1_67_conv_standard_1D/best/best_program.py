# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    x_ptr, 
    weight_ptr, 
    output_ptr,
    batch_size, 
    in_channels, 
    length, 
    out_channels, 
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    stride_x_batch, stride_x_in, stride_x_len,
    stride_weight_oc, stride_weight_ic, stride_weight_k,
    stride_output_batch, stride_output_oc, stride_output_len,
    output_length,
    BLOCK_IC: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid0 = tl.program_id(0)   # batch
    pid1 = tl.program_id(1)   # output channel
    pid2 = tl.program_id(2)   # output position

    # Check if within bounds
    if pid0 >= batch_size or pid1 >= out_channels or pid2 >= output_length:
        return

    # Compute the starting position in the input for the current output position
    start_index = pid2 * stride - padding
    
    # Create ranges for kernel positions and input channels
    d_offsets = tl.arange(0, BLOCK_K)
    ic_offsets = tl.arange(0, BLOCK_IC)
    
    # Compute positions and masks
    pos = start_index + d_offsets * dilation
    mask_d = d_offsets < kernel_size
    mask_pos = (pos >= 0) & (pos < length)
    mask_ic = ic_offsets < in_channels
    full_mask = mask_ic[:, None] & mask_pos[None, :] & mask_d[None, :]

    # Compute base pointers
    x_batch_ptr = x_ptr + pid0 * stride_x_batch
    weight_oc_ptr = weight_ptr + pid1 * stride_weight_oc
    
    # Initialize accumulator
    acc = 0.0
    
    # Compute input and weight pointers
    input_ptrs = x_batch_ptr + (ic_offsets[:, None] * stride_x_in) + (pos[None, :] * stride_x_len)
    weight_ptrs = weight_oc_ptr + (ic_offsets[:, None] * stride_weight_ic) + (d_offsets[None, :] * stride_weight_k)
    
    # Load and compute using vectorization
    input_vals = tl.load(input_ptrs, mask=full_mask, other=0.0)
    weight_vals = tl.load(weight_ptrs, mask=full_mask, other=0.0)
    
    # Compute dot product using efficient element-wise operations
    acc = tl.sum(input_vals * weight_vals)
    
    # Store result
    output_ptr_pos = pid0 * stride_output_batch + pid1 * stride_output_oc + pid2 * stride_output_len
    tl.store(output_ptr + output_ptr_pos, acc)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weight
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.groups != 1:
            # Fallback to PyTorch for grouped convolutions
            return nn.functional.conv1d(
                x, self.weight, self.bias, self.stride, 
                self.padding, self.dilation, self.groups
            )
        
        # Compute output shape
        batch_size, _, length = x.shape
        output_length = (length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        output = torch.empty(batch_size, self.out_channels, output_length, device=x.device, dtype=x.dtype)
        
        # Get strides
        stride_x_batch, stride_x_in, stride_x_len = x.stride()
        stride_weight_oc, stride_weight_ic, stride_weight_k = self.weight.stride()
        stride_output_batch, stride_output_oc, stride_output_len = output.stride()
        
        # Configure grid
        grid = (batch_size, self.out_channels, output_length)
        
        # Calculate block sizes optimized for tensor cores
        BLOCK_IC = min(triton.next_power_of_2(self.in_channels), 128)
        BLOCK_K = min(triton.next_power_of_2(self.kernel_size), 128)
        
        # Adjust block sizes to be multiples of 16 for tensor core efficiency
        if BLOCK_IC >= 16:
            BLOCK_IC = (BLOCK_IC + 15) // 16 * 16
        if BLOCK_K >= 16:
            BLOCK_K = (BLOCK_K + 15) // 16 * 16
        
        conv1d_kernel[grid](
            x, self.weight, output,
            batch_size, self.in_channels, length, self.out_channels, self.kernel_size,
            self.stride, self.padding, self.dilation, self.groups,
            stride_x_batch, stride_x_in, stride_x_len,
            stride_weight_oc, stride_weight_ic, stride_weight_k,
            stride_output_batch, stride_output_oc, stride_output_len,
            output_length,
            BLOCK_IC, BLOCK_K
        )
        
        # Add bias if needed
        if self.bias is not None:
            output += self.bias[:, None]
            
        return output

# Test code
import math
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
length = 512

def get_inputs():
    x = torch.randn(batch_size, in_channels, length, device='cuda')
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
# =================== EVOLVE-BLOCK-END ===================