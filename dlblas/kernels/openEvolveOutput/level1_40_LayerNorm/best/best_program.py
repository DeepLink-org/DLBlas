# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _layer_norm_reduce_kernel(
    x_ptr,
    sum_ptr,
    sum_sq_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_block = tl.program_id(1)
    
    block_start = pid_block * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x_ptrs = x_ptr + pid_batch * N + offsets
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    block_sum = tl.sum(x, axis=0)
    block_sum_sq = tl.sum(x * x, axis=0)
    
    tl.atomic_add(sum_ptr + pid_batch, block_sum)
    tl.atomic_add(sum_sq_ptr + pid_batch, block_sum_sq)

@triton.jit
def _layer_norm_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    var_ptr,
    output_ptr,
    eps,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_block = tl.program_id(1)
    
    block_start = pid_block * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x_ptrs = x_ptr + pid_batch * N + offsets
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    mean = tl.load(mean_ptr + pid_batch)
    var_val = tl.load(var_ptr + pid_batch)
    std = tl.sqrt(var_val + eps)
    
    x_normalized = (x - mean) / std
    
    w_ptrs = weight_ptr + offsets
    b_ptrs = bias_ptr + offsets
    w = tl.load(w_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)
    
    out = x_normalized * w + b
    out_ptrs = output_ptr + pid_batch * N + offsets
    tl.store(out_ptrs, out, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(*normalized_shape))
        self.bias = nn.Parameter(torch.zeros(*normalized_shape))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        normalized_ndim = len(self.normalized_shape)
        batch_dims = input_shape[:-normalized_ndim]
        batch_size = 1
        for d in batch_dims:
            batch_size *= d
        N = 1
        for d in self.normalized_shape:
            N *= d
            
        x_flat = x.reshape(batch_size, N)
        sums = torch.zeros(batch_size, device=x.device, dtype=torch.float32)
        sum_sqs = torch.zeros(batch_size, device=x.device, dtype=torch.float32)
        
        BLOCK_SIZE = 4096
        grid_reduce = (batch_size, triton.cdiv(N, BLOCK_SIZE))
        _layer_norm_reduce_kernel[grid_reduce](x_flat, sums, sum_sqs, N, BLOCK_SIZE=BLOCK_SIZE)
        
        mean = sums / N
        variance = (sum_sqs / N) - (mean * mean)
        
        output = torch.empty_like(x)
        output_flat = output.reshape(batch_size, N)
        weight_flat = self.weight.reshape(-1)
        bias_flat = self.bias.reshape(-1)
        
        grid_forward = (batch_size, triton.cdiv(N, BLOCK_SIZE))
        _layer_norm_forward_kernel[grid_forward](
            x_flat, weight_flat, bias_flat, mean, variance, output_flat, self.eps, N, BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [(features, dim1, dim2)]
# =================== EVOLVE-BLOCK-END ===================