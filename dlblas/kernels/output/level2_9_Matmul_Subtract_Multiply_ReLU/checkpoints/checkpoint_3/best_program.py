# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def _forward_kernel(
    x_ptr, 
    w_ptr, 
    b_ptr,
    out_ptr,
    in_features,
    out_features,
    subtract_value,
    multiply_value,
    stride_xb, 
    stride_xf,
    stride_wf, 
    stride_ws,
    stride_bs,
    stride_ob, 
    stride_of,
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(0)   # row index in batch
    feature_index = tl.arange(0, BLOCK_F)
    mask = feature_index < out_features

    # Compute base pointer for current row in x
    x_row_ptr = x_ptr + pid * stride_xb
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_F,), dtype=tl.float32)
    
    # Loop over columns (in_features)
    for k in range(0, in_features):
        # Load input element (same for all features in row)
        x_val = tl.load(x_row_ptr + k * stride_xf)
        
        # Load weight element (each feature has its own weight)
        w_offset = feature_index * stride_wf + k * stride_ws
        w_val = tl.load(w_ptr + w_offset, mask=mask, other=0)
        
        # Accumulate
        acc += x_val * w_val
    
    # Load bias
    bias_val = tl.load(b_ptr + feature_index * stride_bs, mask=mask, other=0)
    acc += bias_val
    
    # Apply operations
    y = (acc - subtract_value) * multiply_value
    y = tl.where(y > 0, y, 0.0)   # ReLU
    
    # Store result
    out_row_ptr = out_ptr + pid * stride_ob
    tl.store(out_row_ptr + feature_index * stride_of, y, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        x = x.contiguous()
        batch_size = x.shape[0]
        output = torch.empty((batch_size, self.out_features), device=x.device, dtype=x.dtype)
        
        if x.is_cuda:
            BLOCK_F = triton.next_power_of_2(self.out_features)
            grid = (batch_size,)
            
            _forward_kernel[grid](
                x, 
                self.weight, 
                self.bias, 
                output,
                self.in_features,
                self.out_features,
                self.subtract_value,
                self.multiply_value,
                x.stride(0), 
                x.stride(1),
                self.weight.stride(0), 
                self.weight.stride(1),
                self.bias.stride(0),
                output.stride(0), 
                output.stride(1),
                BLOCK_F=BLOCK_F,
            )
        else:
            # Fallback for CPU (not expected in this context)
            linear_out = torch.nn.functional.linear(x, self.weight, self.bias)
            output = torch.relu((linear_out - self.subtract_value) * self.multiply_value)
        
        return output

batch_size = 128
in_features = 10
out_features = 5
subtract_value = 2.0
multiply_value = 1.5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]
# =================== EVOLVE-BLOCK-END ===================