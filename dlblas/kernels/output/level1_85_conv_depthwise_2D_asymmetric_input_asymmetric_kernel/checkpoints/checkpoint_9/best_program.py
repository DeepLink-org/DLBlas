# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def _depthwise_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C, H, W, OH, OW, KH, KW,
    stride_n, stride_c, stride_h, stride_w,
    weight_stride_c, weight_stride_kh, weight_stride_kw,
    stride_h_conv, stride_w_conv, padding_h, padding_w, dilation_h, dilation_w,
    output_stride_n, output_stride_c, output_stride_h, output_stride_w,
    has_bias: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_c = C
    num_pid_oh = tl.cdiv(OH, BLOCK_H)
    num_pid_ow = tl.cdiv(OW, BLOCK_W)
    
    pid_batch = pid // (num_pid_c * num_pid_oh * num_pid_ow)
    pid_channel = (pid // (num_pid_oh * num_pid_ow)) % num_pid_c
    pid_oh = (pid // num_pid_ow) % num_pid_oh
    pid_ow = pid % num_pid_ow

    oh_start = pid_oh * BLOCK_H
    ow_start = pid_ow * BLOCK_W

    ohs = oh_start + tl.arange(0, BLOCK_H)
    ows = ow_start + tl.arange(0, BLOCK_W)
    
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    if has_bias:
        bias_val = tl.load(bias_ptr + pid_channel)
        acc += bias_val

    for kh in range(KH):
        for kw in range(KW):
            ihs = ohs * stride_h_conv + kh * dilation_h - padding_h
            iws = ows * stride_w_conv + kw * dilation_w - padding_w
            
            mask_ih = (ihs >= 0) & (ihs < H)
            mask_iw = (iws >= 0) & (iws < W)
            mask = mask_ih[:, None] & mask_iw[None, :]
            
            input_ptrs = input_ptr + pid_batch * stride_n + pid_channel * stride_c + ihs[:, None] * stride_h + iws[None, :] * stride_w
            input_vals = tl.load(input_ptrs, mask=mask, other=0.0)
            
            weight_val = tl.load(weight_ptr + pid_channel * weight_stride_c + kh * weight_stride_kh + kw * weight_stride_kw)
            acc += input_vals * weight_val

    mask_output = (ohs < OH)[:, None] & (ows < OW)[None, :]
    output_ptrs = output_ptr + pid_batch * output_stride_n + pid_channel * output_stride_c + ohs[:, None] * output_stride_h + ows[None, :] * output_stride_w
    tl.store(output_ptrs, acc, mask=mask_output)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        assert groups == in_channels, "Depthwise convolution requires groups == in_channels"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        
        self.weight = nn.Parameter(torch.empty(out_channels, 1, kernel_size_h, kernel_size_w))
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
        N, C, H, W = x.shape
        OH = (H + 2 * self.padding_h - self.dilation_h * (self.kernel_size_h - 1) - 1) // self.stride_h + 1
        OW = (W + 2 * self.padding_w - self.dilation_w * (self.kernel_size_w - 1) - 1) // self.stride_w + 1
        
        output = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
        
        stride_n, stride_c, stride_h, stride_w = x.stride()
        weight_flat = self.weight.squeeze(1)
        weight_stride_c, weight_stride_kh, weight_stride_kw = weight_flat.stride()
        output_stride_n, output_stride_c, output_stride_h, output_stride_w = output.stride()
        
        grid = (N * C * triton.cdiv(OH, 16) * triton.cdiv(OW, 16),)
        
        _depthwise_conv2d_kernel[grid](
            x, weight_flat, self.bias, output,
            N, C, H, W, OH, OW, self.kernel_size_h, self.kernel_size_w,
            stride_n, stride_c, stride_h, stride_w,
            weight_stride_c, weight_stride_kh, weight_stride_kw,
            self.stride_h, self.stride_w, self.padding_h, self.padding_w, 
            self.dilation_h, self.dilation_w,
            output_stride_n, output_stride_c, output_stride_h, output_stride_w,
            self.bias is not None,
            BLOCK_H=16, BLOCK_W=16
        )
        
        return output

# Test code
batch_size = 16
in_channels = 3
out_channels = in_channels
kernel_size_h = 3
kernel_size_w = 5
width = 256
height = 128
stride_h = 1
stride_w = 1
padding_h = 0
padding_w = 0
dilation_h = 1
dilation_w = 1
groups = in_channels

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups]
# =================== EVOLVE-BLOCK-END ===================