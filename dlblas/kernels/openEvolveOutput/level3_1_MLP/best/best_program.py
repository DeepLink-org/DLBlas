# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _linear_relu_kernel(
    x_ptr, 
    weight_ptr,
    bias_ptr,   
    output_ptr,
    in_features,
    out_features,
    stride_x0, stride_x1,
    stride_weight0, stride_weight1,
    stride_bias,
    stride_output0, stride_output1,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = n_offsets < out_features
    
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    for k in range(0, in_features, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offsets < in_features
        
        input_ptr_row = x_ptr + pid_batch * stride_x0 + k_offsets
        a = tl.load(input_ptr_row, mask=mask_k, other=0.0)
        
        weight_ptr_block = weight_ptr + n_offsets[:, None] * stride_weight0 + k_offsets[None, :] * stride_weight1
        b = tl.load(weight_ptr_block, mask=mask_k[None, :] & mask_n[:, None], other=0.0)
        
        acc += tl.sum(a[None, :] * b, axis=1)
    
    if bias_ptr is not None:
        bias_ptr_block = bias_ptr + n_offsets
        bias_values = tl.load(bias_ptr_block, mask=mask_n, other=0.0)
        acc += bias_values
    
    acc = tl.where(acc > 0, acc, 0.0)
    
    output_ptr_row = output_ptr + pid_batch * stride_output0 + n_offsets * stride_output1
    tl.store(output_ptr_row, acc, mask=mask_n)

class TritonLinearReLU(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x):
        x = x.contiguous()
        output = torch.empty((x.shape[0], self.out_features), device=x.device, dtype=x.dtype)
        
        batch_size = x.shape[0]
        grid = lambda meta: (batch_size, triton.cdiv(self.out_features, meta['BLOCK_SIZE_N']))
        
        _linear_relu_kernel[grid](
            x, self.weight, self.bias, output,
            self.in_features, self.out_features,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            self.bias.stride(0) if self.bias is not None else 0,
            output.stride(0), output.stride(1),
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32
        )
        return output

class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(ModelNew, self).__init__()
        layers = []
        current_input_size = input_size
        
        for layer_size in layer_sizes:
            layers.append(TritonLinearReLU(current_input_size, layer_size))
            current_input_size = layer_size
        
        layers.append(nn.Linear(current_input_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Test code
batch_size = 1
input_size = 1000
layer_sizes = [400, 800]
output_size = 500

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, layer_sizes, output_size]
# =================== EVOLVE-BLOCK-END ===================