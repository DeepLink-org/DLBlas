# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel for fused linear + swish + scale
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_linear_swish_kernel(
    # Pointers to matrices
    x_ptr, w_ptr, b_ptr, output_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    scaling_factor: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr = 64,
):
    pid = tl.program_id(0)
    pid_m = pid // tl.cdiv(N, BLOCK_N)
    pid_n = pid % tl.cdiv(N, BLOCK_N)
    
    # Offsets for output block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Compute pointer offsets for input x and weights
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + tl.arange(0, BLOCK_K)[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + tl.arange(0, BLOCK_K)[:, None] * stride_wk
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load input block
        k_offs = k + tl.arange(0, BLOCK_K)
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load weight block
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Compute block matrix multiplication
        acc += tl.dot(x, w, allow_tf32=True)
        
        # Update pointers for next block
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
    
    # Apply bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b[None, :]
    
    # Apply Swish activation and scaling
    sig = tl.sigmoid(acc)
    output = acc * sig * scaling_factor
    
    # Write back result
    out_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, output, mask=out_mask)

def fused_linear_swish(x, weight, bias, scaling_factor):
    # Check sizes
    M, K = x.shape
    N, _ = weight.shape
    
    # Allocate output
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Grid definition
    grid = lambda opt: (triton.cdiv(M, opt['BLOCK_M']) * triton.cdiv(N, opt['BLOCK_N']),)
    
    # Launch kernel
    _fused_linear_swish_kernel[grid](
        x, weight, bias, output,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        scaling_factor
    )
    return output

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_linear_swish(x, self.weight, self.bias, self.scaling_factor)

batch_size = 128
in_features = 1024
out_features = 512
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]
# =================== EVOLVE-BLOCK-END ===================