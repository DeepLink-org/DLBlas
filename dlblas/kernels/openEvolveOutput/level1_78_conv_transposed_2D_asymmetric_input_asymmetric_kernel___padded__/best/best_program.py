# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = nn.Parameter(torch.empty(
            in_channels,
            out_channels,
            kernel_size[0],
            kernel_size[1]
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        kernel_h, kernel_w = self.kernel_size
        
        height_out = (height - 1) * stride_h - 2 * padding_h + kernel_h
        width_out = (width - 1) * stride_w - 2 * padding_w + kernel_w
        
        output = torch.empty(
            (batch_size, self.out_channels, height_out, width_out),
            device=x.device,
            dtype=x.dtype
        )
        
        x_stride = x.stride()
        weight_stride = self.weight.stride()
        output_stride = output.stride()
        
        grid = lambda opt: (
            batch_size,
            self.out_channels,
            (height_out + opt['BLOCK_H'] - 1) // opt['BLOCK_H'] * 
            (width_out + opt['BLOCK_W'] - 1) // opt['BLOCK_W']
        )
        
        _conv_transpose2d_kernel[grid](
            x, self.weight,
            self.bias if self.bias is not None else None, 
            output,
            batch_size,
            self.in_channels,
            self.out_channels,
            height,
            width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            height_out,
            width_out,
            x_stride[0],
            x_stride[1],
            x_stride[2],
            x_stride[3],
            weight_stride[0],
            weight_stride[1],
            weight_stride[2],
            weight_stride[3],
            output_stride[0],
            output_stride[1],
            output_stride[2],
            output_stride[3],
            HAS_BIAS=(self.bias is not None)
        )
        
        return output

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16}, num_warps=4),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32}, num_warps=4),
        triton.Config({'BLOCK_H': 64, 'BLOCK_W': 64}, num_warps=4),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16}, num_warps=8),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32}, num_warps=8),
        triton.Config({'BLOCK_H': 64, 'BLOCK_W': 64}, num_warps=8),
    ],
    key=['height_out', 'width_out'],
)
@triton.jit
def _conv_transpose2d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    height_out,
    width_out,
    x_batch_stride,
    x_in_channel_stride,
    x_height_stride,
    x_width_stride,
    weight_in_channel_stride,
    weight_out_channel_stride,
    weight_kh_stride,
    weight_kw_stride,
    output_batch_stride,
    output_out_channel_stride,
    output_height_stride,
    output_width_stride,
    HAS_BIAS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_block = tl.program_id(2)
    
    num_blocks_w = (width_out + BLOCK_W - 1) // BLOCK_W
    block_h_idx = pid_block // num_blocks_w
    block_w_idx = pid_block % num_blocks_w
    
    h_offsets = block_h_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = block_w_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    
    h_mask = h_offsets < height_out
    w_mask = w_offsets < width_out
    
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    for kh in range(0, kernel_h):
        for kw in range(0, kernel_w):
            for ic in range(0, in_channels):
                ih = (h_offsets[:, None] + padding_h - kh) / stride_h
                iw = (w_offsets[None, :] + padding_w - kw) / stride_w
                
                is_int_h = (h_offsets[:, None] + padding_h - kh) % stride_h == 0
                is_int_w = (w_offsets[None, :] + padding_w - kw) % stride_w == 0
                
                ih_int = tl.math.floor(ih).to(tl.int32)
                iw_int = tl.math.floor(iw).to(tl.int32)
                
                in_bounds = (ih_int >= 0) & (ih_int < height) & (iw_int >= 0) & (iw_int < width)
                
                mask = is_int_h & is_int_w & in_bounds & h_mask[:, None] & w_mask[None, :]
                
                x_offset = (
                    pid_b * x_batch_stride +
                    ic * x_in_channel_stride +
                    ih_int * x_height_stride +
                    iw_int * x_width_stride
                )
                
                weight_offset = (
                    ic * weight_in_channel_stride +
                    pid_oc * weight_out_channel_stride +
                    kh * weight_kh_stride +
                    kw * weight_kw_stride
                )
                
                x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
                w_val = tl.load(weight_ptr + weight_offset)
                
                acc += tl.where(mask, x_val * w_val, 0.0)
    
    if HAS_BIAS:
        bias_val = tl.load(bias_ptr + pid_oc)
        acc += bias_val
    
    output_offset = (
        pid_b * output_batch_stride +
        pid_oc * output_out_channel_stride +
        h_offsets[:, None] * output_height_stride +
        w_offsets[None, :] * output_width_stride
    )
    
    tl.store(output_ptr + output_offset, acc, mask=h_mask[:, None] & w_mask[None, :])

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height = 128
width = 256
stride = (1, 1)
padding = (1, 2)

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
# =================== EVOLVE-BLOCK-END ===================