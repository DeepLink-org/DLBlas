# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv1x1_kernel(
    x_ptr, w_ptr, output_ptr,
    in_channels, out_channels, H, W,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_woc, stride_wic,
    stride_ob, stride_oc, stride_oh, stride_ow,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_hw = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    n_offset = pid_n * BLOCK_N
    n_mask = n_offset + tl.arange(0, BLOCK_N) < out_channels
    
    b = pid_hw // (H * W)
    hw = pid_hw % (H * W)
    h = hw // W
    w = hw % W
    
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in range(0, tl.cdiv(in_channels, BLOCK_K)):
        k_offset = k * BLOCK_K
        k_mask = k_offset + tl.arange(0, BLOCK_K) < in_channels
        
        x_ptr_offset = b*stride_xb + h*stride_xh + w*stride_xw + (k_offset + tl.arange(0, BLOCK_K))*stride_xc
        x_vals = tl.load(x_ptr + x_ptr_offset, mask=k_mask, other=0.0)
        
        w_ptr_offset = (n_offset + tl.arange(0, BLOCK_N)[:, None])*stride_woc + (k_offset + tl.arange(0, BLOCK_K)[None, :])*stride_wic
        w_vals = tl.load(w_ptr + w_ptr_offset, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)
    
    out_ptr_offset = b*stride_ob + h*stride_oh + w*stride_ow + (n_offset + tl.arange(0, BLOCK_N))*stride_oc
    tl.store(output_ptr + out_ptr_offset, acc, mask=n_mask)

class TritonConv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, x):
        batch_size, _, H, W = x.shape
        output = torch.empty((batch_size, self.out_channels, H, W), device=x.device, dtype=x.dtype)
        
        grid = (batch_size * H * W, triton.cdiv(self.out_channels, 128))
        conv1x1_kernel[grid](
            x, self.weight, output,
            self.in_channels, self.out_channels, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.weight.stride(0), self.weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            BLOCK_N=128, BLOCK_K=64
        )
        return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(ModelNew, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                TritonConv1x1(in_channels, hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, 
                      padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        
        self.project_conv = nn.Sequential(
            TritonConv1x1(hidden_dim, out_channels),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        
        if self.use_residual:
            x += identity
        
        return x

import math
batch_size = 10
in_channels = 112
out_channels = 192
kernel_size = 5
stride = 2
expand_ratio = 6

def get_inputs():
    return [torch.randn(batch_size, in_channels, 224, 224)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, expand_ratio]
# =================== EVOLVE-BLOCK-END ===================