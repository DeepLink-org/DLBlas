# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def fused_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    output_ptr,
    subtract_value,
    multiply_value,
    stride_xb, stride_xk,
    stride_wn, stride_wk,
    stride_ob, stride_on,
    batch_size: tl.constexpr,
    out_features: tl.constexpr,
    in_features: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    if pid_b >= batch_size or pid_n >= out_features:
        return
        
    x_row_ptr = x_ptr + pid_b * stride_xb
    w_row_ptr = w_ptr + pid_n * stride_wn
    
    acc = 0.0
    for k in range(0, in_features):
        x_val = tl.load(x_row_ptr + k * stride_xk)
        w_val = tl.load(w_row_ptr + k * stride_wk)
        acc += x_val * w_val
        
    b_val = tl.load(b_ptr + pid_n)
    acc += b_val
    
    acc = (acc - subtract_value) * multiply_value
    acc = tl.maximum(acc, 0.0)
    
    output_offset = pid_b * stride_ob + pid_n * stride_on
    tl.store(output_ptr + output_offset, acc)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        batch_size = x.shape[0]
        output = torch.empty((batch_size, self.out_features), 
                             device=x.device, dtype=x.dtype)
        
        grid = (batch_size, self.out_features)
        
        fused_kernel[grid](
            x, 
            self.weight, 
            self.bias, 
            output,
            self.subtract_value,
            self.multiply_value,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            output.stride(0), output.stride(1),
            batch_size=batch_size,
            out_features=self.out_features,
            in_features=self.in_features,
        )
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