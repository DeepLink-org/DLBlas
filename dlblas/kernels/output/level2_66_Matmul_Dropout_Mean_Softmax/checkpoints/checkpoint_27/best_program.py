# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def linear_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N,
    stride_xb, stride_xm,
    stride_wn, stride_wm,
    stride_outb, stride_outn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    n_off = pid_n * BLOCK_SIZE_N
    n_offs = n_off + tl.arange(0, BLOCK_SIZE_N)
    mask_n = n_offs < N
    
    out = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    for m in range(0, M, BLOCK_SIZE_M):
        m_offs = m + tl.arange(0, BLOCK_SIZE_M)
        mask_m = m_offs < M
        
        x_vals = tl.load(x_ptr + pid_b * stride_xb + m_offs * stride_xm, mask=mask_m, other=0.0)
        w_ptrs = weight_ptr + n_offs[:, None] * stride_wn + m_offs[None, :] * stride_wm
        w_block = tl.load(w_ptrs, mask=mask_n[:, None] & mask_m[None, :], other=0.0)
        
        out += tl.sum(x_vals[None, :] * w_block, axis=1)
    
    bias_vals = tl.load(bias_ptr + n_offs, mask=mask_n, other=0.0)
    out += bias_vals
    
    out_ptrs = output_ptr + pid_b * stride_outb + n_offs * stride_outn
    tl.store(out_ptrs, out, mask=mask_n)

def linear_triton(x, weight, bias):
    B, M = x.shape
    N = weight.shape[0]
    output = torch.empty(B, N, device=x.device, dtype=x.dtype)
    
    stride_xb = x.stride(0)
    stride_xm = x.stride(1)
    stride_wn = weight.stride(0)
    stride_wm = weight.stride(1)
    stride_outb = output.stride(0)
    stride_outn = output.stride(1)
    
    grid = (B, triton.cdiv(N, 32))
    linear_kernel[grid](
        x, weight, bias, output,
        M, N,
        stride_xb, stride_xm,
        stride_wn, stride_wm,
        stride_outb, stride_outn,
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32
    )
    return output

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.dropout = nn.Dropout(dropout_p)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        x = linear_triton(x, self.weight, self.bias)
        x = self.dropout(x)
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.softmax(x, dim=1)
        return x

batch_size = 128
in_features = 100
out_features = 50
dropout_p = 0.2

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, dropout_p]
# =================== EVOLVE-BLOCK-END ===================