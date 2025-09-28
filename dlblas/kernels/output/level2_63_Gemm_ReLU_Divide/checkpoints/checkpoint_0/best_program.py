# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def linear_relu_div_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_features, out_features, divisor,
    stride_input, stride_weight, stride_output,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_F: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_f = tl.program_id(1)
    
    b_offset = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    f_offset = pid_f * BLOCK_SIZE_F + tl.arange(0, BLOCK_SIZE_F)
    
    output_block_ptr = (
        output_ptr + 
        b_offset[:, None] * stride_output + 
        f_offset[None, :] * stride_output
    )
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_F), dtype=tl.float32)
    
    for k in range(0, in_features, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        
        input_mask = (b_offset[:, None] < out_features) & (k_offsets[None, :] < in_features)
        weight_mask = (f_offset[:, None] < in_features) & (k_offsets[None, :] < out_features)
        
        input_block = tl.load(
            input_ptr + b_offset[:, None] * stride_input + k_offsets[None, :],
            mask=input_mask,
            other=0.0
        )
        weight_block = tl.load(
            weight_ptr + f_offset[:, None] * stride_weight + k_offsets[None, :],
            mask=weight_mask,
            other=0.0
        )
        
        acc += tl.dot(input_block, weight_block)
    
    bias_block = tl.load(bias_ptr + f_offset, mask=f_offset < out_features, other=0.0)
    acc += bias_block[None, :]
    acc = tl.maximum(acc, 0.0)
    acc = acc / divisor
    
    output_mask = (b_offset[:, None] < out_features) & (f_offset[None, :] < out_features)
    tl.store(output_block_ptr, acc, mask=output_mask)

class TritonLinearReLUDiv(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.divisor = divisor
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        output = torch.empty(
            x.shape[0], self.out_features, 
            device=x.device, dtype=x.dtype
        )
        
        grid = lambda meta: (
            triton.cdiv(x.shape[0], meta['BLOCK_SIZE_B']),
            triton.cdiv(self.out_features, meta['BLOCK_SIZE_F'])
        )
        
        linear_relu_div_kernel[grid](
            x, self.weight, self.bias, output,
            self.in_features, self.out_features, self.divisor,
            x.stride(0), self.weight.stride(0), output.stride(0),
            BLOCK_SIZE_B=32, BLOCK_SIZE_F=64, BLOCK_SIZE_K=64
        )
        return output

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.linear = TritonLinearReLUDiv(in_features, out_features, divisor)
    
    def forward(self, x):
        return self.linear(x)

batch_size = 128
in_features = 1024
out_features = 512
divisor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, divisor]
# =================== EVOLVE-BLOCK-END ===================