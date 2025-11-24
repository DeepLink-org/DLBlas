# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel for fused linear + ReLU operation
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def linear_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_im, stride_in,
    stride_wk, stride_wn,
    stride_om, stride_on,
    ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    input_ptr_block = input_ptr + offs_m[:, None] * stride_im
    weight_ptr_block = weight_ptr + offs_n[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        
        # Vectorized loads with tensor core optimization
        input_block = tl.load(
            input_ptr_block + k_offs[None, :] * stride_in,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0
        )
        weight_block = tl.load(
            weight_ptr_block + k_offs[:, None] * stride_wn,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0
        )
        
        # Tensor core optimized dot product
        acc += tl.dot(input_block, weight_block, allow_tf32=True)

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    if ACTIVATION:
        acc = tl.where(acc > 0, acc, 0.0)
    
    mask_output = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    output_ptr += offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(output_ptr, acc, mask=mask_output)

# Optimized linear layer with Triton
class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, activation=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        output = torch.empty(x.shape[0], self.out_features, device=x.device, dtype=x.dtype)
        
        # Kernel configuration
        grid = lambda meta: (triton.cdiv(x.shape[0], meta['BLOCK_M']) * triton.cdiv(self.out_features, meta['BLOCK_N']),)
        
        # Launch kernel
        linear_relu_kernel[grid](
            x, self.weight, self.bias, output,
            x.shape[0], self.out_features, self.in_features,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            output.stride(0), output.stride(1),
            ACTIVATION=1 if self.activation else 0,
            BLOCK_K=64
        )
        return output

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(ModelNew, self).__init__()
        
        layers = []
        current_input_size = input_size
        
        # Hidden layers with fused linear + ReLU
        for hidden_size in hidden_layer_sizes:
            layers.append(TritonLinear(current_input_size, hidden_size, activation=True))
            current_input_size = hidden_size
        
        # Final layer without activation
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