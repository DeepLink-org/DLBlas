# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_N': 64}, num_warps=8),
    ],
    key=['in_features', 'out_features'],
)
@triton.jit
def linear_gelu_kernel(
    x_ptr, w_ptr, b_ptr, output_ptr,
    in_features, out_features,
    stride_x0, stride_x1,
    stride_w0, stride_w1,
    stride_out0, stride_out1,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    
    n_offsets = pid1 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < out_features
    
    x_row_ptr = x_ptr + pid0 * stride_x0
    w_block_ptr = w_ptr + n_offsets[:, None] * stride_w0 + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_w1
    
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for k in range(0, in_features, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < in_features
        x_val = tl.load(x_row_ptr + k_offsets, mask=k_mask, other=0.0)
        w_val = tl.load(w_block_ptr + k_offsets[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        acc += tl.sum(w_val * x_val[None, :], axis=1)
    
    b_val = tl.load(b_ptr + n_offsets, mask=n_mask, other=0.0)
    acc += b_val
    
    # GELU approximation
    acc = acc * 0.5 * (1.0 + tl.erf(acc * 0.7071067811865475))
    
    out_offsets = pid0 * stride_out0 + n_offsets * stride_out1
    tl.store(output_ptr + out_offsets, acc, mask=n_mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        x = x.contiguous()
        output = torch.empty((x.size(0), self.out_features), device=x.device, dtype=torch.float32)
        grid = (x.size(0), triton.cdiv(self.out_features, 64))
        linear_gelu_kernel[grid](
            x, self.weight, self.bias, output,
            self.in_features, self.out_features,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            output.stride(0), output.stride(1)
        )
        return torch.softmax(output, dim=1)

batch_size = 128
in_features = 100
out_features = 10

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]
# =================== EVOLVE-BLOCK-END ===================