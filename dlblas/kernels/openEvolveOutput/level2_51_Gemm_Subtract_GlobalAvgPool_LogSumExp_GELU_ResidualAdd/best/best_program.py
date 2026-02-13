# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def _reduction_kernel(
    input1_ptr,
    input2_ptr,
    output_ptr,
    out_features,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * out_features + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < out_features

    row = tl.load(input1_ptr + offsets, mask=mask, other=0.0)
    sub_vec = tl.load(input2_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    
    diff = row - sub_vec
    row_sum = tl.sum(diff, axis=0)
    mean_val = row_sum / out_features
    
    erf_val = tl.math.erf(mean_val * 0.7071067811865475)
    gelu_val = 0.5 * mean_val * (1.0 + erf_val)
    
    tl.store(output_ptr + pid, gelu_val)

@triton.jit
def _add_kernel(
    input3_ptr,
    scalars_ptr,
    output_ptr,
    in_features,
    BLOCK_SIZE: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    
    col_offsets = pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < in_features
    
    scalar = tl.load(scalars_ptr + pid0)
    x = tl.load(input3_ptr + pid0 * in_features + col_offsets, mask=mask, other=0.0)
    result = x + scalar
    
    tl.store(output_ptr + pid0 * in_features + col_offsets, result, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        original_x = x
        x_linear = self.gemm(x)
        
        batch_size = x_linear.shape[0]
        out_features = self.subtract.shape[0]
        in_features = original_x.shape[1]
        
        scalars = torch.empty(batch_size, device=x_linear.device, dtype=torch.float32)
        P2 = triton.next_power_of_2(out_features)
        grid1 = (batch_size,)
        _reduction_kernel[grid1](x_linear, self.subtract, scalars, out_features, BLOCK_SIZE=P2)
        
        output = torch.empty_like(original_x)
        P3 = 1024
        grid2 = (batch_size, (in_features + P3 - 1) // P3)
        _add_kernel[grid2](original_x, scalars, output, in_features, BLOCK_SIZE=P3)
        
        return output

batch_size = 128
in_features = 1024
out_features = 512

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]
# =================== EVOLVE-BLOCK-END ===================