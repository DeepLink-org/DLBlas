# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fused_gemm_bias_act_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    batch_size, in_features, out_features,
    stride_xb, stride_xm,
    stride_wn, stride_wk,
    stride_yb, stride_ym,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    
    # Create block offsets
    offs_m = pid0 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid1 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
        k_offs = k * BLOCK_SIZE_K + offs_k
        # Create masks
        mask_x = (offs_m[:, None] < batch_size) & (k_offs[None, :] < in_features)
        mask_w = (offs_n[:, None] < out_features) & (k_offs[None, :] < in_features)
        
        # Load input and weight blocks
        x = tl.load(x_ptr + offs_m[:, None]*stride_xb + k_offs[None, :]*stride_xm, 
                    mask=mask_x, other=0.0)
        w = tl.load(w_ptr + offs_n[:, None]*stride_wn + k_offs[None, :]*stride_wk, 
                    mask=mask_w, other=0.0)
        
        # Compute matrix product
        w = tl.trans(w)
        acc += tl.dot(x, w)
    
    # Load bias and add
    bias = tl.load(b_ptr + offs_n, mask=offs_n < out_features, other=0.0)
    acc += bias[None, :]
    
    # Apply Hardtanh
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)
    
    # Apply Mish activation
    softplus = tl.log(1.0 + tl.exp(acc))
    tanh_sp = tl.tanh(softplus)
    acc = acc * tanh_sp
    
    # Create output mask and store
    mask_y = (offs_m[:, None] < batch_size) & (offs_n[None, :] < out_features)
    tl.store(y_ptr + offs_m[:, None]*stride_yb + offs_n[None, :]*stride_ym, 
             acc, mask=mask_y)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
    
    def forward(self, x):
        batch_size, in_feats = x.shape
        out_feats = self.weight.shape[0]
        
        # Allocate output tensor
        y = torch.empty((batch_size, out_feats), device=x.device, dtype=x.dtype)
        
        # Compute grid size
        grid_m = triton.cdiv(batch_size, 64)
        grid_n = triton.cdiv(out_feats, 64)
        
        # Launch kernel
        _fused_gemm_bias_act_kernel[(grid_m, grid_n)](
            x, self.weight, self.bias, y,
            batch_size, in_feats, out_feats,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32
        )
        
        # Apply group normalization
        y = self.groupnorm(y)
        return y

batch_size = 128
in_features = 512
out_features = 1024
bias_shape = (out_features,)
num_groups = 32

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, bias_shape, num_groups]
# =================== EVOLVE-BLOCK-END ===================