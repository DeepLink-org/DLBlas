# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_group_norm_activation(
    input_ptr, 
    gamma_ptr, 
    beta_ptr, 
    output_ptr,
    hidden_size,
    group_size,
    num_groups,
    eps,
    negative_slope,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid // num_groups
    group_idx = pid % num_groups
    
    base_offset = batch_idx * hidden_size + group_idx * group_size
    offsets = base_offset + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < group_size
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean and variance with stable reduction
    mean = tl.sum(x, axis=0) / group_size
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / group_size
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Group normalization with fused activation
    normalized = x_centered * inv_std
    gamma = tl.load(gamma_ptr + group_idx * group_size + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
    beta = tl.load(beta_ptr + group_idx * group_size + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = normalized * gamma + beta
    
    # Leaky ReLU and element-wise addition (x + x equivalent)
    y = tl.where(y >= 0, y, y * negative_slope) * 2.0
    
    tl.store(output_ptr + offsets, y, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.group_size = hidden_size // num_groups

    def forward(self, x):
        x = self.fc(x)
        
        if x.is_cuda:
            batch_size = x.shape[0]
            output = torch.empty_like(x)
            
            grid = (batch_size * self.num_groups,)
            BLOCK_SIZE = triton.next_power_of_2(self.group_size)
            
            fused_group_norm_activation[grid](
                x, 
                self.gn.weight, 
                self.gn.bias, 
                output,
                self.hidden_size,
                self.group_size,
                self.num_groups,
                self.gn.eps,
                self.leaky_relu.negative_slope,
                BLOCK_SIZE=BLOCK_SIZE
            )
            return output
        else:
            # CPU fallback
            x = self.gn(x)
            x = self.leaky_relu(x)
            return x + x

batch_size = 128
input_size = 512
hidden_size = 256
num_groups = 8

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, num_groups]
# =================== EVOLVE-BLOCK-END ===================