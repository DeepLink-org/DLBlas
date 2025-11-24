# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fused_matmul_scale_kernel(
    x_ptr, w_ptr, b_ptr, output_ptr,
    x_batch_stride, x_feature_stride,
    w_out_stride, w_in_stride,
    b_stride,
    output_batch_stride, output_feature_stride,
    batch, in_features, out_features,
    scaling_factor,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    batch_offsets = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    out_offsets = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for k in range(0, in_features, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        
        # Load input block
        x_mask = (batch_offsets[:, None] < batch) & (k_offsets[None, :] < in_features)
        x_ptrs = x_ptr + batch_offsets[:, None] * x_batch_stride + k_offsets[None, :] * x_feature_stride
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load weight block (transposed layout: [K, O])
        w_mask = (k_offsets[:, None] < in_features) & (out_offsets[None, :] < out_features)
        w_ptrs = w_ptr + k_offsets[:, None] * w_in_stride + out_offsets[None, :] * w_out_stride
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Matrix multiplication (no transposition needed)
        acc += tl.dot(x_block, w_block)
    
    # Load bias block
    b_ptrs = b_ptr + out_offsets * b_stride
    b_mask = out_offsets < out_features
    b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)
    
    # Scale and add residual
    acc = acc * (scaling_factor + 1.0) + b_block[None, :] * (scaling_factor + 1.0)
    
    # Store results
    out_ptrs = output_ptr + batch_offsets[:, None] * output_batch_stride + out_offsets[None, :] * output_feature_stride
    out_mask = (batch_offsets[:, None] < batch) & (out_offsets[None, :] < out_features)
    tl.store(out_ptrs, acc, mask=out_mask)

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
        batch, _ = x.shape
        out = torch.empty((batch, self.out_features), device=x.device, dtype=x.dtype)
        
        BLOCK_SIZE_B = 16
        BLOCK_SIZE_O = 64
        BLOCK_SIZE_K = 32
        
        grid_b = triton.cdiv(batch, BLOCK_SIZE_B)
        grid_o = triton.cdiv(self.out_features, BLOCK_SIZE_O)
        
        _fused_matmul_scale_kernel[(grid_b, grid_o)](
            x, self.weight, self.bias, out,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            self.bias.stride(0),
            out.stride(0), out.stride(1),
            batch, self.in_features, self.out_features,
            self.scaling_factor,
            BLOCK_SIZE_B=BLOCK_SIZE_B, BLOCK_SIZE_O=BLOCK_SIZE_O, BLOCK_SIZE_K=BLOCK_SIZE_K
        )
        return out

batch_size = 128
in_features = 64
out_features = 128
scaling_factor = 0.5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]
# =================== EVOLVE-BLOCK-END ===================