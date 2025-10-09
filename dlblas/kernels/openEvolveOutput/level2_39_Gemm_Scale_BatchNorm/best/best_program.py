# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_linear_scale_kernel(
    x_ptr, w_ptr, b_ptr, s_ptr, out_ptr,
    in_features, out_features,
    stride_x, stride_w, stride_b, stride_s, stride_out,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid0 = tl.program_id(0)  # Batch index
    pid1 = tl.program_id(1)  # Output feature block index
    
    # Create ranges for current block
    offs_n = pid1 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for boundaries
    mask_n = offs_n < out_features
    mask_k = offs_k < in_features
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Loop over input features
    for k in range(0, in_features, BLOCK_SIZE_K):
        # Load block of input data
        x_vals = tl.load(
            x_ptr + pid0 * stride_x + k + offs_k, 
            mask=mask_k, other=0.0
        )
        
        # Load block of weights
        w_vals = tl.load(
            w_ptr + (offs_n[:, None]) * stride_w + (k + offs_k[None, :]), 
            mask=mask_n[:, None] & mask_k[None, :], other=0.0
        )
        
        # Compute partial dot product
        x_vals_f16 = x_vals.to(tl.float16)
        w_vals_f16 = w_vals.to(tl.float16)
        partial = tl.sum(x_vals_f16[None, :] * w_vals_f16, axis=1)
        acc += partial.to(tl.float32)
    
    # Load bias and scale
    b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    s_vals = tl.load(s_ptr + offs_n, mask=mask_n, other=0.0)
    
    # Apply bias and scale
    output = (acc + b_vals) * s_vals
    
    # Store result
    tl.store(
        out_ptr + pid0 * stride_out + offs_n, 
        output, mask=mask_n
    )

def fused_linear_scale(x, weight, bias, scale):
    batch_size, in_features = x.shape
    out_features = weight.shape[0]
    
    # Preallocate output tensor
    output = torch.empty((batch_size, out_features), 
                         device=x.device, dtype=torch.float32)
    
    # Calculate grid dimensions
    grid = lambda meta: (
        batch_size, 
        triton.cdiv(out_features, meta['BLOCK_SIZE_N'])
    )
    
    # Launch kernel with optimized block sizes
    fused_linear_scale_kernel[grid](
        x, weight, bias, scale, output,
        in_features, out_features,
        x.stride(0), weight.stride(0), 
        bias.stride(0), scale.stride(0), output.stride(0),
        BLOCK_SIZE_N=128, BLOCK_SIZE_K=64
    )
    return output

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
    
    def forward(self, x):
        # Fused linear layer + scaling
        x = fused_linear_scale(x, self.gemm.weight, self.gemm.bias, self.scale)
        x = self.bn(x)
        return x

batch_size = 128
in_features = 1024
out_features = 512
scale_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scale_shape]
# =================== EVOLVE-BLOCK-END ===================