# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fused_gemm_bias_relu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    output_ptr,
    batch_size,
    in_features,
    out_features,
    stride_x0,
    stride_x1,
    stride_w0,
    stride_w1,
    stride_output0,
    stride_output1,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_bm = tl.program_id(0)
    pid_bn = tl.program_id(1)
    
    offs_bm = pid_bm * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_bn * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    x_ptrs = x_ptr + offs_bm[:, None] * stride_x0 + offs_k[None, :] * stride_x1
    w_ptrs = w_ptr + offs_k[:, None] * stride_w0 + offs_bn[None, :] * stride_w1
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
        k_remaining = in_features - k * BLOCK_SIZE_K
        k_valid = tl.min(BLOCK_SIZE_K, k_remaining)
        
        x_mask = (offs_bm[:, None] < batch_size) & (offs_k[None, :] < k_valid)
        w_mask = (offs_k[:, None] < k_valid) & (offs_bn[None, :] < out_features)
        
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc += tl.dot(x_block, w_block)
        
        x_ptrs += BLOCK_SIZE_K * stride_x1
        w_ptrs += BLOCK_SIZE_K * stride_w0
    
    b_ptrs = b_ptr + offs_bn
    b_mask = offs_bn < out_features
    b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)
    
    acc += b_block[None, :]
    acc = tl.maximum(acc, 0.0)
    
    offs_output = offs_bm[:, None] * stride_output0 + offs_bn[None, :] * stride_output1
    output_mask = (offs_bm[:, None] < batch_size) & (offs_bn[None, :] < out_features)
    tl.store(output_ptr + offs_output, acc, mask=output_mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
    
    def forward(self, x):
        batch_size, _ = x.shape
        out_features = self.bias.shape[0]
        
        output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
        
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
        
        grid = (
            triton.cdiv(batch_size, BLOCK_SIZE_M),
            triton.cdiv(out_features, BLOCK_SIZE_N),
        )
        
        _fused_gemm_bias_relu_kernel[grid](
            x,
            self.gemm.weight,
            self.bias,
            output,
            batch_size,
            self.gemm.weight.size(1),
            out_features,
            x.stride(0),
            x.stride(1),
            self.gemm.weight.stride(0),
            self.gemm.weight.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        
        return output

batch_size = 128
in_features = 1024
out_features = 512
bias_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, bias_shape]
# =================== EVOLVE-BLOCK-END ===================