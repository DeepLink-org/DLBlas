# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_stride, output_stride, weight_stride0, weight_stride1,
    in_features, out_features, batch_size,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    n_start = pid * BLOCK_SIZE_N
    n_offs = n_start + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offs < out_features

    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    for k in range(0, in_features, BLOCK_SIZE_K):
        k_offs = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offs < in_features
        
        # Load input block
        input_block = tl.load(
            input_ptr + k_offs,
            mask=k_mask,
            other=0.0
        )
        
        # Load weight block
        weight_ptrs = weight_ptr + n_offs[:, None] * weight_stride0 + k_offs[None, :] * weight_stride1
        weight_block = tl.load(
            weight_ptrs,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0
        )
        
        # Compute partial dot product
        acc += tl.sum(input_block[None, :] * weight_block, axis=1)
    
    # Add bias
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + n_offs, mask=n_mask, other=0.0)
        acc += bias
    
    # Store result
    output_ptrs = output_ptr + n_offs * output_stride
    tl.store(output_ptrs, acc, mask=n_mask)

class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        output = torch.empty(x.shape[0], self.out_features, device=x.device, dtype=x.dtype)
        
        # 1D launch configuration - each program computes one output feature
        grid = lambda meta: (triton.cdiv(self.out_features, meta['BLOCK_SIZE_N']),)
        
        # Launch kernel
        linear_kernel[grid](
            x, self.weight, self.bias, output,
            x.stride(0), output.stride(0),
            self.weight.stride(0), self.weight.stride(1),
            self.in_features, self.out_features, x.shape[0],
            BLOCK_SIZE_K=64, BLOCK_SIZE_N=128
        )
        return output

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(ModelNew, self).__init__()
        layers = []
        current_input_size = input_size
        
        for hidden_size in hidden_layer_sizes:
            layers.append(TritonLinear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            current_input_size = hidden_size
        
        layers.append(TritonLinear(current_input_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Test code
batch_size = 1
input_size = 1000
hidden_layer_sizes = [2000, 2000]
output_size = 10

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]
# =================== EVOLVE-BLOCK-END ===================