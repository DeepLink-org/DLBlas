# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose1d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    length_in,
    length_out,
    kernel_size: tl.constexpr,
    stride,
    padding,
    dilation,
    numel,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    output_channels_length = out_channels * length_out
    b_idx = offs // output_channels_length
    remainder = offs % output_channels_length
    c_out_idx = remainder // length_out
    i_out = remainder % length_out

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + c_out_idx, mask=mask, other=0.0)
        acc += bias

    for k in tl.static_range(kernel_size):
        numerator = i_out + padding - k * dilation
        cond1 = numerator >= 0
        cond2 = (numerator % stride) == 0
        j = numerator // stride
        cond3 = (j >= 0) & (j < length_in)
        cond = cond1 & cond2 & cond3

        for c_in_block in range(0, in_channels, BLOCK_C):
            c_in_offsets = c_in_block + tl.arange(0, BLOCK_C)
            c_in_mask = c_in_offsets < in_channels

            input_ptrs = (
                x_ptr + 
                b_idx[:, None] * (in_channels * length_in) + 
                c_in_offsets[None, :] * length_in + 
                j[:, None]
            )
            weight_ptrs = (
                weight_ptr + 
                c_in_offsets[None, :] * (out_channels * kernel_size) + 
                c_out_idx[:, None] * kernel_size + 
                k
            )
            
            mask2d = mask[:, None] & cond[:, None] & c_in_mask[None, :]
            x_vals = tl.load(input_ptrs, mask=mask2d, other=0.0)
            w_vals = tl.load(weight_ptrs, mask=mask2d, other=0.0)
            
            product = x_vals * w_vals
            acc += tl.sum(product, axis=1)

    tl.store(output_ptr + offs, acc, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, length_in = x.shape
        out_channels = self.conv1d_transpose.out_channels
        kernel_size = self.conv1d_transpose.kernel_size[0]
        stride = self.conv1d_transpose.stride[0]
        padding = self.conv1d_transpose.padding[0]
        dilation = self.conv1d_transpose.dilation[0]
        
        length_out = (length_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
        output = torch.empty((batch_size, out_channels, length_out), device=x.device, dtype=x.dtype)
        
        x_contig = x.contiguous()
        weight_contig = self.conv1d_transpose.weight.contiguous()
        bias_contig = self.conv1d_transpose.bias if self.conv1d_transpose.bias is not None else None
        
        numel = batch_size * out_channels * length_out
        BLOCK_SIZE = 1024
        BLOCK_C = 16
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        
        conv_transpose1d_kernel[grid](
            x_contig, 
            weight_contig, 
            bias_contig, 
            output,
            batch_size, 
            in_channels, 
            out_channels, 
            length_in, 
            length_out,
            kernel_size, 
            stride, 
            padding, 
            dilation,
            numel,
            BLOCK_SIZE=BLOCK_SIZE,
            BLOCK_C=BLOCK_C
        )
        
        return output

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
length = 128
stride = 2
padding = 1
dilation = 2

def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
# =================== EVOLVE-BLOCK-END ===================