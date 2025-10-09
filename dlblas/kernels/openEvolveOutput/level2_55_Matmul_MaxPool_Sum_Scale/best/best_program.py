# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def _forward_kernel(
    x_ptr, 
    weight_ptr, 
    bias_ptr, 
    output_ptr, 
    in_features, 
    out_features, 
    scale_factor, 
    stride_x, 
    stride_out,
    BLOCK_SIZE_IN: tl.constexpr, 
    BLOCK_SIZE_OUT: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Compute starting position for the current batch
    x_offset = pid * stride_x
    total_sum = 0.0

    # Calculate number of complete pairs
    num_pairs = out_features // 2
    
    # Process each complete pair
    for pair_idx in range(num_pairs):
        j = pair_idx * 2
        # Load biases for both features in pair
        bias0 = tl.load(bias_ptr + j)
        bias1 = tl.load(bias_ptr + j + 1)
            
        acc0 = bias0
        acc1 = bias1
        
        # Process input features in blocks
        for k_block in range(0, tl.cdiv(in_features, BLOCK_SIZE_IN)):
            k_start = k_block * BLOCK_SIZE_IN
            k_offsets = k_start + tl.arange(0, BLOCK_SIZE_IN)
            mask = k_offsets < in_features
            
            # Load input block
            x_vals = tl.load(x_ptr + x_offset + k_offsets, mask=mask, other=0.0)
            
            # Load weight blocks for both features
            w0_offsets = j * in_features + k_offsets
            w0_vals = tl.load(weight_ptr + w0_offsets, mask=mask, other=0.0)
            w1_offsets = (j + 1) * in_features + k_offsets
            w1_vals = tl.load(weight_ptr + w1_offsets, mask=mask, other=0.0)
            
            # Accumulate dot products
            acc0 += tl.sum(x_vals * w0_vals)
            acc1 += tl.sum(x_vals * w1_vals)
        
        # Apply max pooling to pair
        max_val = tl.maximum(acc0, acc1)
        total_sum += max_val
    
    # Apply scaling and store result
    total_sum = total_sum * scale_factor
    tl.store(output_ptr + pid * stride_out, total_sum)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_size = x.shape[0]
        output = torch.empty(batch_size, device=x.device, dtype=torch.float32)
        
        # Use optimized kernel for kernel_size=2
        if self.kernel_size == 2:
            grid = (batch_size,)
            BLOCK_SIZE_IN = 128
            BLOCK_SIZE_OUT = 2
            
            _forward_kernel[grid](
                x, self.weight, self.bias, output,
                self.in_features, self.out_features, self.scale_factor,
                x.stride(0), output.stride(0),
                BLOCK_SIZE_IN=BLOCK_SIZE_IN,
                BLOCK_SIZE_OUT=BLOCK_SIZE_OUT
            )
        else:
            # Fallback for other kernel sizes
            x = torch.nn.functional.linear(x, self.weight, self.bias)
            x = x.unsqueeze(1)
            x = torch.nn.functional.max_pool1d(x, kernel_size=self.kernel_size)
            x = x.squeeze(1)
            x = torch.sum(x, dim=1)
            x = x * self.scale_factor
            output = x
        
        return output

batch_size = 128
in_features = 10
out_features = 5
kernel_size = 2
scale_factor = 0.5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]
# =================== EVOLVE-BLOCK-END ===================