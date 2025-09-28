# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64, "VEC_SIZE": 16}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 128, "VEC_SIZE": 16}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256, "VEC_SIZE": 16}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 128, "VEC_SIZE": 8}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 64, "VEC_SIZE": 8}, num_warps=2),
    ],
    key=["in_channels", "out_channels", "kernel_size"],
)
@triton.jit
def conv_transpose1d_forward(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    in_channels, in_length,
    out_channels, out_length,
    kernel_size, stride, padding, dilation,
    stride_xb, stride_xc, stride_xl,
    stride_wic, stride_woc, stride_wk,
    stride_ob, stride_oc, stride_ol,
    BLOCK_SIZE: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_block = tl.program_id(2)
    
    o_idx_start = pid_block * BLOCK_SIZE
    o_idx = o_idx_start + tl.arange(0, BLOCK_SIZE)
    valid_mask = o_idx < out_length
    
    weight_ptr += pid_oc * stride_woc
    
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Precompute indices and masks
    for k in range(0, kernel_size):
        pos = o_idx + padding - k * dilation
        residue = pos % stride
        i_idx = tl.where(residue == 0, pos // stride, -1)
        valid_pos = (i_idx >= 0) & (i_idx < in_length) & valid_mask
        
        # Vectorized input channel processing
        for c_in_base in range(0, in_channels, VEC_SIZE):
            c_offsets = tl.arange(0, VEC_SIZE)
            c_mask = c_offsets < (in_channels - c_in_base)
            c_in = c_in_base + c_offsets
            
            # Load weights vector
            w_vals = tl.load(
                weight_ptr + c_in[:, None] * stride_wic + k * stride_wk,
                mask=c_mask[:, None],
                other=0.0
            )
            
            # Load input values vector
            x_vals = tl.load(
                x_ptr + pid_b * stride_xb + c_in[:, None] * stride_xc + i_idx[None, :] * stride_xl,
                mask=c_mask[:, None] & valid_pos[None, :],
                other=0.0
            )
            
            # Accumulate
            product = w_vals * x_vals
            acc += tl.sum(product, axis=0)
    
    # Add bias
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + pid_oc)
        acc += bias_val
        
    # Store results
    output_offsets = pid_b * stride_ob + pid_oc * stride_oc + o_idx
    tl.store(output_ptr + output_offsets, acc, mask=valid_mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=bias) if bias else None
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, in_length = x.shape
        out_length = (in_length - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1
        
        output = torch.empty(batch_size, self.out_channels, out_length, 
                             device=x.device, dtype=x.dtype)
        
        grid = lambda meta: (batch_size, self.out_channels, triton.cdiv(out_length, meta['BLOCK_SIZE']))
        
        conv_transpose1d_forward[grid](
            x, self.weight, self.bias, output,
            self.in_channels, in_length,
            self.out_channels, out_length,
            self.kernel_size, self.stride, self.padding, self.dilation,
            x.stride(0), x.stride(1), x.stride(2),
            self.weight.stride(0), self.weight.stride(1), self.weight.stride(2),
            output.stride(0), output.stride(1), output.stride(2)
        )
        
        return output

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 5
length = 256
stride = 1
padding = 0
dilation = 3

def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
# =================== EVOLVE-BLOCK-END ===================