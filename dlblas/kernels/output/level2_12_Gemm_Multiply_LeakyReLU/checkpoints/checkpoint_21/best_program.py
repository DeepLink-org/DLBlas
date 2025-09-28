# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def _forward_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    output_ptr,
    multiplier,
    negative_slope,
    batch_size,
    in_features,
    out_features,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_bn,
    stride_om,
    stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    
    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    w_ptrs = w_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
        k_remaining = in_features - k * BLOCK_SIZE_K
        k_valid = rk < k_remaining
        x_mask = k_valid[None, :] & (rm[:, None] < batch_size)
        w_mask = k_valid[:, None] & (rn[None, :] < out_features)
        
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x, w, allow_tf32=True)
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk
    
    b_ptrs = b_ptr + rn
    b = tl.load(b_ptrs, mask=rn < out_features, other=0.0)
    acc += b[None, :]
    
    acc = acc * multiplier
    condition = acc < 0
    acc = tl.where(condition, acc * negative_slope, acc)
    
    out_ptrs = output_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    mask = (rm[:, None] < batch_size) & (rn[None, :] < out_features)
    tl.store(out_ptrs, acc, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_size = x.shape[0]
        output = torch.empty(batch_size, self.out_features, device=x.device, dtype=x.dtype)
        
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 64
        
        grid = (
            triton.cdiv(batch_size, BLOCK_SIZE_M),
            triton.cdiv(self.out_features, BLOCK_SIZE_N)
        )
        
        _forward_kernel[grid](
            x, self.weight, self.bias, output,
            self.multiplier, self.negative_slope,
            batch_size, self.in_features, self.out_features,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            self.bias.stride(0),
            output.stride(0), output.stride(1),
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
        )
        return output

batch_size = 128
in_features = 1024
out_features = 512
multiplier = 2.0
negative_slope = 0.1

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]
# =================== EVOLVE-BLOCK-END ===================