# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _forward_kernel(
    T_ptr, 
    bias_ptr, 
    output_ptr,
    B, 
    C, 
    H, 
    W,
    T_stride_b, 
    T_stride_c, 
    T_stride_h, 
    T_stride_w,
    output_stride_b, 
    output_stride_c, 
    output_stride_w,
    dtype: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    if pid_batch >= B or pid_w >= W:
        return
    
    s = 0.0
    for h in range(H):
        offs0 = pid_batch * T_stride_b + 0 * T_stride_c + h * T_stride_h + pid_w * T_stride_w
        min_val = tl.load(T_ptr + offs0)
        for c in range(1, C):
            offs = pid_batch * T_stride_b + c * T_stride_c + h * T_stride_h + pid_w * T_stride_w
            val = tl.load(T_ptr + offs)
            min_val = tl.minimum(min_val, val)
        s += min_val
    
    base = tl.gelu(s)
    
    for c in range(C):
        bias_val = tl.load(bias_ptr + c)
        out_val = base + bias_val
        offs_out = pid_batch * output_stride_b + c * output_stride_c + pid_w * output_stride_w
        tl.store(output_ptr + offs_out, out_val)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
    def forward(self, x):
        x = self.conv_transpose(x)
        B, C, H, W = x.shape
        output = torch.empty(B, C, 1, W, device=x.device, dtype=x.dtype)
        
        T_stride_b = x.stride(0)
        T_stride_c = x.stride(1)
        T_stride_h = x.stride(2)
        T_stride_w = x.stride(3)
        
        output_stride_b = output.stride(0)
        output_stride_c = output.stride(1)
        output_stride_w = output.stride(3)
        
        bias_1d = self.bias.to(x.dtype).view(-1)
        
        grid = (B, W)
        _forward_kernel[grid](
            x, 
            bias_1d, 
            output,
            B, C, H, W,
            T_stride_b, T_stride_c, T_stride_h, T_stride_w,
            output_stride_b, output_stride_c, output_stride_w,
            dtype=tl.float32 if x.dtype == torch.float32 else tl.float16
        )
        
        return output

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]
# =================== EVOLVE-BLOCK-END ===================