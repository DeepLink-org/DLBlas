# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_matmul_scale_kernel(
    x_ptr, w_ptr, b_ptr, output_ptr,
    scaling_factor,
    batch_size, in_features, out_features,
    stride_xb, stride_xf,
    stride_wo, stride_wf,
    stride_bo,
    stride_ob, stride_of,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    if pid_b >= batch_size or pid_o >= out_features:
        return
    
    k_offsets = tl.arange(0, BLOCK_K)
    mask = k_offsets < in_features
    
    # Load input row
    x_ptr_row = x_ptr + pid_b * stride_xb + k_offsets * stride_xf
    x_row = tl.load(x_ptr_row, mask=mask, other=0.0)
    
    # Load weight row
    w_ptr_row = w_ptr + pid_o * stride_wo + k_offsets * stride_wf
    w_row = tl.load(w_ptr_row, mask=mask, other=0.0)
    
    # Compute dot product
    dot = tl.sum(x_row * w_row)
    
    # Load bias and apply scaling
    bias = tl.load(b_ptr + pid_o * stride_bo)
    result = (dot + bias) * scaling_factor
    
    # Store result
    out_ptr = output_ptr + pid_b * stride_ob + pid_o * stride_of
    tl.store(out_ptr, result)

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
        output = torch.empty((batch_size, out_feat), device=x.device, dtype=x.dtype)
        
        # Ensure contiguous tensors
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias.contiguous()
        
        # Compute fused operation
        scale_val = 1.0 + self.scaling_factor
        grid = (batch_size, out_feat)
        BLOCK_K = triton.next_power_of_2(in_feat)
        
        fused_matmul_scale_kernel[grid](
            x, weight, bias, output,
            scale_val,
            batch_size, in_feat, out_feat,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            bias.stride(0),
            output.stride(0), output.stride(1),
            BLOCK_K=BLOCK_K
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