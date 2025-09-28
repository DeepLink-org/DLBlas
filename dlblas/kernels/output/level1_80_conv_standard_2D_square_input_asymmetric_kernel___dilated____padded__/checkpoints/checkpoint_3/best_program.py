# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    H, W, KH, KW, OH, OW,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    x_batch_stride, x_channel_stride, x_h_stride, x_w_stride,
    w_outc_stride, w_inc_stride, w_h_stride, w_w_stride,
    y_batch_stride, y_channel_stride, y_h_stride, y_w_stride,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_oh = tl.program_id(2)
    pid_ow = tl.program_id(3)
    
    oh_start = pid_oh * BLOCK_H
    ow_start = pid_ow * BLOCK_W
    oh_idx = oh_start + tl.arange(0, BLOCK_H)
    ow_idx = ow_start + tl.arange(0, BLOCK_W)
    
    oh_mask = oh_idx < OH
    ow_mask = ow_idx < OW
    
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    for ic in range(0, tl.load(w_ptr + w_inc_stride)):
        for kh in range(KH):
            for kw in range(KW):
                ih = oh_idx * stride_h + kh * dilation_h - padding_h
                iw = ow_idx * stride_w + kw * dilation_w - padding_w
                
                ih_mask = (ih >= 0) & (ih < H)
                iw_mask = (iw >= 0) & (iw < W)
                mask = oh_mask[:, None] & ow_mask[None, :] & ih_mask[:, None] & iw_mask[None, :]
                
                x_offsets = pid_batch * x_batch_stride + ic * x_channel_stride + ih[:, None] * x_h_stride + iw[None, :] * x_w_stride
                x_val = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
                
                w_offsets = pid_oc * w_outc_stride + ic * w_inc_stride + kh * w_h_stride + kw * w_w_stride
                w_val = tl.load(w_ptr + w_offsets)
                
                acc += x_val * w_val
    
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_oc)
        acc += bias
    
    y_offsets = pid_batch * y_batch_stride + pid_oc * y_channel_stride + oh_idx[:, None] * y_h_stride + ow_idx[None, :] * y_w_stride
    tl.store(y_ptr + y_offsets, acc, mask=oh_mask[:, None] & ow_mask[None, :])

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
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
        batch_size, _, H, W = x.shape
        kH, kW = self.kernel_size
        dilation_h, dilation_w = self.dilation
        padding_h, padding_w = self.padding
        stride_h, stride_w = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride
        
        OH = (H + 2 * padding_h - dilation_h * (kH - 1) - 1) // stride_h + 1
        OW = (W + 2 * padding_w - dilation_w * (kW - 1) - 1) // stride_w + 1
        
        y = torch.empty((batch_size, self.out_channels, OH, OW), device=x.device, dtype=x.dtype)
        
        BLOCK_H, BLOCK_W = 16, 16
        grid = (
            batch_size,
            self.out_channels,
            triton.cdiv(OH, BLOCK_H),
            triton.cdiv(OW, BLOCK_W),
        )
        
        conv2d_kernel[grid](
            x, self.weight, self.bias, y,
            H, W, kH, kW, OH, OW,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            *x.stride(),
            *self.weight.stride(),
            *y.stride(),
            BLOCK_H, BLOCK_W
        )
        
        return y

# Test code
import math
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5)
width = 256
height = 256
stride = 1
padding = (1, 2)
dilation = (2, 1)

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
# =================== EVOLVE-BLOCK-END ===================