# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_OC': 128, 'BLOCK_SIZE_IC': 32, 'BLOCK_SIZE_SPATIAL': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_OC': 64, 'BLOCK_SIZE_IC': 32, 'BLOCK_SIZE_SPATIAL': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_OC': 128, 'BLOCK_SIZE_IC': 64, 'BLOCK_SIZE_SPATIAL': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_OC': 64, 'BLOCK_SIZE_IC': 64, 'BLOCK_SIZE_SPATIAL': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_OC': 128, 'BLOCK_SIZE_IC': 16, 'BLOCK_SIZE_SPATIAL': 128}, num_warps=4),
    ],
    key=['in_channels', 'out_channels', 'n'],
)
@triton.jit
def _conv1x1_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    n,
    stride_x_b,
    stride_x_ic,
    stride_x_s,
    stride_weight_oc,
    stride_weight_ic,
    stride_output_b,
    stride_output_oc,
    stride_output_s,
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_IC: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_s = tl.program_id(2)

    oc_offsets = pid_oc * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC)
    s_offsets = pid_s * BLOCK_SIZE_SPATIAL + tl.arange(0, BLOCK_SIZE_SPATIAL)
    
    oc_mask = oc_offsets < out_channels
    s_mask = s_offsets < n
    
    acc = tl.zeros((BLOCK_SIZE_OC, BLOCK_SIZE_SPATIAL), dtype=tl.float32)
    
    for ic0 in range(0, in_channels, BLOCK_SIZE_IC):
        ic_offsets = ic0 + tl.arange(0, BLOCK_SIZE_IC)
        ic_mask = ic_offsets < in_channels
        
        w = tl.load(weight_ptr + oc_offsets[:, None] * stride_weight_oc + ic_offsets[None, :] * stride_weight_ic,
                    mask=oc_mask[:, None] & ic_mask[None, :],
                    other=0.0)
        
        i_ptr = x_ptr + pid_b * stride_x_b + ic_offsets[:, None] * stride_x_ic + s_offsets[None, :] * stride_x_s
        i = tl.load(i_ptr,
                    mask=ic_mask[:, None] & s_mask[None, :],
                    other=0.0)
        
        acc += tl.dot(w, i, allow_tf32=True)
    
    if bias_ptr is not None:
        b = tl.load(bias_ptr + oc_offsets, mask=oc_mask, other=0.0)
        acc += b[:, None]
    
    o_ptr = output_ptr + pid_b * stride_output_b + oc_offsets[:, None] * stride_output_oc + s_offsets[None, :] * stride_output_s
    tl.store(o_ptr, acc, mask=oc_mask[:, None] & s_mask[None, :])

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        n = height * width
        x_reshaped = x.reshape(batch_size, self.in_channels, n)
        
        output = torch.empty(batch_size, self.out_channels, n, 
                             device=x.device, dtype=x.dtype)
        
        # Use minimum block sizes to ensure sufficient coverage
        grid = (batch_size, 
                triton.cdiv(self.out_channels, 64), 
                triton.cdiv(n, 64))
        
        bias_ptr = self.bias if self.bias is not None else None
        
        # Removed explicit block size arguments to avoid autotuner conflicts
        _conv1x1_forward_kernel[grid](
            x_reshaped, self.weight, bias_ptr, output,
            batch_size, self.in_channels, self.out_channels, n,
            x_reshaped.stride(0), x_reshaped.stride(1), x_reshaped.stride(2),
            self.weight.stride(0), self.weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2)
        )
        
        return output.reshape(batch_size, self.out_channels, height, width)

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
width = 256
height = 256

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels]
# =================== EVOLVE-BLOCK-END ===================