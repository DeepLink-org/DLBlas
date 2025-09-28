# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    ],
    key=["num_spatial"],
)
@triton.jit
def _conv_transpose3d_kernel(
    input_ptr, weight_ptr, output_ptr, bias_ptr,
    in_channels, out_channels, groups,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    num_spatial,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_block = tl.program_id(2)
    
    spatial_start = pid_block * BLOCK_SIZE
    spatial_offsets = spatial_start + tl.arange(0, BLOCK_SIZE)
    mask = spatial_offsets < num_spatial
    
    w_out = spatial_offsets % W_out
    h_out = (spatial_offsets // W_out) % H_out
    d_out = spatial_offsets // (W_out * H_out)
    
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    group_size = out_channels // groups
    group_id = pid_oc // group_size
    in_channels_per_group = in_channels // groups
    start_ic = group_id * in_channels_per_group
    end_ic = start_ic + in_channels_per_group
    
    for ic in range(start_ic, end_ic):
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    d_in_val = d_out + padding_d - kd
                    h_in_val = h_out + padding_h - kh
                    w_in_val = w_out + padding_w - kw
                    
                    d_in = tl.where(stride_d != 0, d_in_val // stride_d, 0)
                    h_in = tl.where(stride_h != 0, h_in_val // stride_h, 0)
                    w_in = tl.where(stride_w != 0, w_in_val // stride_w, 0)
                    
                    rem_d = d_in_val - d_in * stride_d
                    rem_h = h_in_val - h_in * stride_h
                    rem_w = w_in_val - w_in * stride_w
                    
                    mask_div = (rem_d == 0) & (rem_h == 0) & (rem_w == 0)
                    mask_bounds = (d_in >= 0) & (d_in < D_in) & (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
                    mask_combined = mask_div & mask_bounds & mask
                    
                    base = pid_b * in_channels * D_in * H_in * W_in + ic * D_in * H_in * W_in
                    spatial_offset = d_in * (H_in * W_in) + h_in * W_in + w_in
                    input_ptrs = input_ptr + base + spatial_offset
                    
                    input_val = tl.load(input_ptrs, mask=mask_combined, other=0.0)
                    
                    weight_index = ic * (out_channels // groups) * kernel_d * kernel_h * kernel_w + \
                                  (pid_oc % group_size) * (kernel_d * kernel_h * kernel_w) + \
                                  (kd * kernel_h * kernel_w + kh * kernel_w + kw)
                    weight_val = tl.load(weight_ptr + weight_index)
                    
                    acc += input_val * weight_val
    
    output_base = pid_b * out_channels * D_out * H_out * W_out + \
                  pid_oc * D_out * H_out * W_out + spatial_offsets
    if bias_ptr != 0:
        bias_val = tl.load(bias_ptr + pid_oc)
        acc += bias_val
        
    tl.store(output_ptr + output_base, acc, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        kd, kh, kw = kernel_size
        self.weight = nn.Parameter(torch.empty(
            in_channels, 
            out_channels // groups, 
            kd, kh, kw
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
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, D_in, H_in, W_in = x.shape
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding
        opd, oph, opw = self.output_padding
        
        D_out = (D_in - 1) * sd - 2 * pd + kd + opd
        H_out = (H_in - 1) * sh - 2 * ph + kh + oph
        W_out = (W_in - 1) * sw - 2 * pw + kw + opw
        
        num_spatial = D_out * H_out * W_out
        output = torch.empty(batch_size, self.out_channels, D_out, H_out, W_out, 
                             device=x.device, dtype=x.dtype)
        
        bias_ptr = 0
        if self.bias is not None:
            bias_ptr = self.bias.data_ptr()
        
        grid = (batch_size, self.out_channels, triton.cdiv(num_spatial, 128))
        _conv_transpose3d_kernel[grid](
            x, self.weight, output, bias_ptr,
            self.in_channels, self.out_channels, self.groups,
            D_in, H_in, W_in,
            D_out, H_out, W_out,
            kd, kh, kw,
            sd, sh, sw,
            pd, ph, pw,
            num_spatial,
            BLOCK_SIZE=128
        )
        return output

# Test code
batch_size = 16
in_channels = 32
out_channels = 16
kernel_size = (3, 5, 7)  # Asymmetric kernel size
depth_in = 16
height_in = 32
width_in = 64

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth_in, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
# =================== EVOLVE-BLOCK-END ===================