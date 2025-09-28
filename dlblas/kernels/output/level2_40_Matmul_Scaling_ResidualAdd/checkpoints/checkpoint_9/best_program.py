# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _forward_kernel(
    x_ptr, w_ptr, b_ptr, output_ptr,
    scaling_factor,
    batch_size, in_features, out_features,
    stride_x_batch, stride_x_feat,
    stride_w_out, stride_w_feat,
    stride_b_out,
    stride_out_batch, stride_out_feat,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    if pid_b >= batch_size or pid_j >= out_features:
        return
    
    idx = tl.arange(0, BLOCK_SIZE)
    mask = idx < in_features
    
    x_row_ptr = x_ptr + pid_b * stride_x_batch + idx * stride_x_feat
    w_row_ptr = w_ptr + pid_j * stride_w_out + idx * stride_w_feat
    
    x_row = tl.load(x_row_ptr, mask=mask, other=0.0)
    w_row = tl.load(w_row_ptr, mask=mask, other=0.0)
    
    dot = tl.sum(x_row * w_row)
    bias = tl.load(b_ptr + pid_j * stride_b_out)
    result = (dot + bias) * scaling_factor
    
    output_ptr_val = output_ptr + pid_b * stride_out_batch + pid_j * stride_out_feat
    tl.store(output_ptr_val, result)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.scaling_factor = scaling_factor
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        batch_size, in_feat = x.shape
        out_feat = self.weight.shape[0]
        output = torch.empty((batch_size, out_feat), device=x.device, dtype=torch.float32)
        
        if not x.is_contiguous():
            x = x.contiguous()
        if not self.weight.is_contiguous():
            self.weight = self.weight.contiguous()
        if not self.bias.is_contiguous():
            self.bias = self.bias.contiguous()
            
        scale_val = 1.0 + self.scaling_factor
        grid = (batch_size, out_feat)
        BLOCK_SIZE = triton.next_power_of_2(in_feat)
        
        _forward_kernel[grid](
            x, self.weight, self.bias, output,
            scale_val,
            batch_size, in_feat, out_feat,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            self.bias.stride(0),
            output.stride(0), output.stride(1),
            BLOCK_SIZE=BLOCK_SIZE
        )
        return output

batch_size = 128
in_features = 64
out_features = 128
scaling_factor = 0.5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]
# =================== EVOLVE-BLOCK-END ===================