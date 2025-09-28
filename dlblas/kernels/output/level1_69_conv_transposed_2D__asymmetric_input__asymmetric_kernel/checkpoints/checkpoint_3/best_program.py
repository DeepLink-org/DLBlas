# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_channels, out_channels, H_in, W_in, H_out, W_out,
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups,
    kH, kW,
    input_bs_stride, input_c_stride, input_h_stride, input_w_stride,
    weight_ic_stride, weight_oc_stride, weight_h_stride, weight_w_stride,
    output_bs_stride, output_c_stride, output_h_stride, output_w_stride,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_IC: tl.constexpr
):
    pid_nc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    n = pid_nc // out_channels
    oc = pid_nc % out_channels
    group_id = oc // (out_channels // groups)
    oc_in_group = oc % (out_channels // groups)
    in_channels_per_group = in_channels // groups
    start_ic = group_id * in_channels_per_group
    
    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    spatial_mask = h_mask[:, None] & w_mask[None, :]
    
    output = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    for kh in range(kH):
        for kw in range(kW):
            base_h = (h_offsets + padding_h - kh * dilation_h)
            base_w = (w_offsets + padding_w - kw * dilation_w)
            valid_base = (base_h[:, None] >= 0) & (base_w[None, :] >= 0)
            ih = base_h // stride_h
            iw = base_w // stride_w
            valid_div = (base_h[:, None] % stride_h == 0) & (base_w[None, :] % stride_w == 0)
            valid_bounds = (ih[:, None] < H_in) & (iw[None, :] < W_in)
            valid = valid_base & valid_div & valid_bounds
            
            for ic_offset in range(0, in_channels_per_group, BLOCK_IC):
                ic_offsets = ic_offset + tl.arange(0, BLOCK_IC)
                ic_mask = ic_offsets < in_channels_per_group
                ic = start_ic + ic_offsets
                
                # Precompute input pointers
                input_ptrs = (
                    input_ptr + 
                    n * input_bs_stride + 
                    ic[:, None, None] * input_c_stride + 
                    ih[None, :, None] * input_h_stride + 
                    iw[None, None, :] * input_w_stride
                )
                input_val = tl.load(input_ptrs, mask=valid[None, :, :] & ic_mask[:, None, None], other=0.0)
                
                # Vectorized weight loading
                weight_ptrs = (
                    weight_ptr + 
                    ic[:, None, None] * weight_ic_stride + 
                    oc_in_group * weight_oc_stride + 
                    kh * weight_h_stride + 
                    kw * weight_w_stride
                )
                weight_val = tl.load(weight_ptrs, mask=ic_mask[:, None, None], other=0.0)
                
                # Accumulate
                output += tl.sum(input_val * weight_val[:, None, None], axis=0)
    
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + oc)
        output += bias_val
    
    output_ptrs = (
        output_ptr + 
        n * output_bs_stride + 
        oc * output_c_stride + 
        h_offsets[:, None] * output_h_stride + 
        w_offsets[None, :] * output_w_stride
    )
    tl.store(output_ptrs, output, mask=spatial_mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 output_padding: tuple = (0, 0), dilation: tuple = (1, 1), 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        
        self.weight = nn.Parameter(torch.empty(
            in_channels, out_channels // groups, *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H_in, W_in = x.shape
        kH, kW = self.kernel_size
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation
        output_padding_h, output_padding_w = self.output_padding
        
        H_out = (H_in - 1) * stride_h - 2 * padding_h + dilation_h * (kH - 1) + output_padding_h + 1
        W_out = (W_in - 1) * stride_w - 2 * padding_w + dilation_w * (kW - 1) + output_padding_w + 1
        
        output = torch.empty((N, self.out_channels, H_out, W_out), 
                             device=x.device, dtype=x.dtype)
        
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        BLOCK_H, BLOCK_W = 16, 16
        grid = (
            N * self.out_channels,
            triton.cdiv(H_out, BLOCK_H),
            triton.cdiv(W_out, BLOCK_W)
        )
        
        conv_transpose2d_kernel[grid](
            x, weight, self.bias, output,
            self.in_channels, self.out_channels, 
            H_in, W_in, H_out, W_out,
            stride_h, stride_w, padding_h, padding_w, 
            dilation_h, dilation_w, self.groups,
            kH, kW,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_IC=32
        )
        return output

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height_in = 16
width_in = 32

def get_inputs():
    x = torch.randn(batch_size, in_channels, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
# =================== EVOLVE-BLOCK-END ===================