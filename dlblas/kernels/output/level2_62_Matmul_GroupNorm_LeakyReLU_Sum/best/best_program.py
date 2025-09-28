# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def group_norm_fused_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    hidden_size,
    num_groups,
    eps,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    group_id = pid % num_groups
    row_id = pid // num_groups
    group_size = hidden_size // num_groups
    
    start_idx = row_id * hidden_size + group_id * group_size
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < group_size
    
    # Load input data for this group
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean and variance for group normalization
    group_sum = tl.sum(x, axis=0)
    group_mean = group_sum / group_size
    x_centered = tl.where(mask, x - group_mean, 0.0)
    group_var = tl.sum(x_centered * x_centered, axis=0) / group_size + eps
    
    # Normalize and apply affine transformation
    x_norm = x_centered * tl.math.rsqrt(group_var)
    w = tl.load(weight_ptr + group_id * group_size + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
    b = tl.load(bias_ptr + group_id * group_size + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = x_norm * w + b
    
    # Apply LeakyReLU and double the output
    y = tl.where(y >= 0, y, y * negative_slope) * 2.0
    tl.store(output_ptr + offsets, y, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn_weight = nn.Parameter(torch.ones(hidden_size))
        self.gn_bias = nn.Parameter(torch.zeros(hidden_size))
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.fc(x)
        output = torch.empty_like(x)
        
        batch_size = x.shape[0]
        group_size = self.hidden_size // self.num_groups
        n_elements = batch_size * self.num_groups
        
        # Ensure group size is power of two for efficient reduction
        BLOCK_SIZE = triton.next_power_of_2(group_size)
        grid = (n_elements,)
        
        group_norm_fused_kernel[grid](
            x, self.gn_weight, self.gn_bias, output,
            n_elements,
            self.hidden_size,
            self.num_groups,
            self.eps,
            self.negative_slope,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return output

batch_size = 128
input_size = 512
hidden_size = 256
num_groups = 8

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, num_groups]
# =================== EVOLVE-BLOCK-END ===================