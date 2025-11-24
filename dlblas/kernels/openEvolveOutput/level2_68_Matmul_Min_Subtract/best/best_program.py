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
    stride_x_batch,
    stride_x_feature,
    stride_weight_row,
    stride_weight_col,
    stride_bias,
    stride_output_batch,
    stride_output_feature,
    constant_val: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    batch_size: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = pid
    if row_idx >= batch_size:
        return
    
    # Initialize accumulator with zeros
    acc = tl.zeros((BLOCK_F,), dtype=tl.float32)
    
    # Vectorized computation for input features
    for k in range(0, in_features):
        x_val = tl.load(x_ptr + row_idx * stride_x_batch + k * stride_x_feature)
        w_offsets = tl.arange(0, BLOCK_F) * stride_weight_row + k * stride_weight_col
        w_mask = tl.arange(0, BLOCK_F) < out_features
        w_val = tl.load(weight_ptr + w_offsets, mask=w_mask, other=0.0)
        acc += x_val * w_val
    
    # Load bias and apply operations
    b_offsets = tl.arange(0, BLOCK_F)
    b_mask = tl.arange(0, BLOCK_F) < out_features
    bias_val = tl.load(bias_ptr + b_offsets, mask=b_mask, other=0.0)
    acc += bias_val
    
    # Apply min and subtract operations
    result = tl.minimum(acc, constant_val) - constant_val
    
    # Store results with mask
    out_offsets = row_idx * stride_output_batch + tl.arange(0, BLOCK_F) * stride_output_feature
    out_mask = tl.arange(0, BLOCK_F) < out_features
    tl.store(output_ptr + out_offsets, result, mask=out_mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous()
        batch_size = x.shape[0]
        output = torch.empty((batch_size, self.out_features), device=x.device, dtype=x.dtype)
        
        # Determine optimal BLOCK_F (next power of two >= out_features)
        BLOCK_F = triton.next_power_of_2(self.out_features)
        
        grid = (batch_size,)
        _forward_kernel[grid](
            x,
            self.linear.weight,
            self.linear.bias,
            output,
            x.stride(0),
            x.stride(1),
            self.linear.weight.stride(0),
            self.linear.weight.stride(1),
            self.linear.bias.stride(0),
            output.stride(0),
            output.stride(1),
            self.constant.item(),
            self.in_features,
            self.out_features,
            batch_size,
            BLOCK_F,
        )
        return output

batch_size = 128
in_features = 10
out_features = 5
constant = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, constant]
# =================== EVOLVE-BLOCK-END ===================