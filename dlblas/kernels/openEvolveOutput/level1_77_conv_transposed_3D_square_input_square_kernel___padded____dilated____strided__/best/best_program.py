# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 4, 'BLOCK_H': 32, 'BLOCK_W': 32}, num_warps=8),
        triton.Config({'BLOCK_D': 8, 'BLOCK_H': 16, 'BLOCK_W': 16}, num_warps=4),
        triton.Config({'BLOCK_D': 16, 'BLOCK_H': 8, 'BLOCK_W': 8}, num_warps=8),
        triton.Config({'BLOCK_D': 8, 'BLOCK_H': 32, 'BLOCK_W': 32}, num_warps=4),
        triton.Config({'BLOCK_D': 16, 'BLOCK_H': 16, 'BLOCK_W': 16}, num_warps=8),
    ],
    key=['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation']
)
@triton.jit
def conv_transpose3d_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    in_channels,
    out_channels,
    depth,
    height,
    width,
    kernel_size,
    stride,
    padding,
    dilation,
    output_depth,
    output_height,
    output_width,
    stride_x_b, stride_x_c, stride_x_d, stride_x_h, stride_x_w,
    stride_w_ic, stride_w_oc, stride_w_d, stride_w_h, stride_w_w,
    stride_out_b, stride_out_c, stride_out_d, stride_out_h, stride_out_w,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_d = tl.cdiv(output_depth, BLOCK_D)
    num_blocks_h = tl.cdiv(output_height, BLOCK_H)
    num_blocks_w = tl.cdiv(output_width, BLOCK_W)
    
    b = pid // (out_channels * num_blocks_d * num_blocks_h * num_blocks_w)
    pid_remain = pid % (out_channels * num_blocks_d * num_blocks_h * num_blocks_w)
    oc = pid_remain // (num_blocks_d * num_blocks_h * num_blocks_w)
    pid_spatial = pid_remain % (num_blocks_d * num_blocks_h * num_blocks_w)
    
    block_d = pid_spatial // (num_blocks_h * num_blocks_w)
    block_h = (pid_spatial % (num_blocks_h * num_blocks_w)) // num_blocks_w
    block_w = pid_spatial % num_blocks_w
    
    d_start = block_d * BLOCK_D
    h_start = block_h * BLOCK_H
    w_start = block_w * BLOCK_W
    
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    
    acc = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    for ic in range(0, in_channels):
        for kd in range(0, kernel_size):
            for kh in range(0, kernel_size):
                for kw in range(0, kernel_size):
                    d_base = d_offsets + padding - kd * dilation
                    h_base = h_offsets + padding - kh * dilation
                    w_base = w_offsets + padding - kw * dilation
                    
                    d_in = d_base // stride
                    h_in = h_base // stride
                    w_in = w_base // stride
                    
                    valid_d = (d_base % stride == 0) & (d_in >= 0) & (d_in < depth)
                    valid_h = (h_base % stride == 0) & (h_in >= 0) & (h_in < height)
                    valid_w = (w_base % stride == 0) & (w_in >= 0) & (w_in < width)
                    
                    valid_mask = valid_d[:, None, None] & valid_h[None, :, None] & valid_w[None, None, :]
                    
                    x_ptrs = x_ptr + b * stride_x_b + \
                             ic * stride_x_c + \
                             d_in[:, None, None] * stride_x_d + \
                             h_in[None, :, None] * stride_x_h + \
                             w_in[None, None, :] * stride_x_w
                    
                    x_val = tl.load(x_ptrs, mask=valid_mask, other=0.0)
                    weight_val = tl.load(weight_ptr + ic * stride_w_ic + \
                                        oc * stride_w_oc + \
                                        kd * stride_w_d + \
                                        kh * stride_w_h + \
                                        kw * stride_w_w)
                    
                    acc += x_val * weight_val
    
    output_ptrs = output_ptr + b * stride_out_b + \
                 oc * stride_out_c + \
                 d_offsets[:, None, None] * stride_out_d + \
                 h_offsets[None, :, None] * stride_out_h + \
                 w_offsets[None, None, :] * stride_out_w
                 
    tl.store(output_ptrs, acc, mask=(
        (d_offsets[:, None, None] < output_depth) & 
        (h_offsets[None, :, None] < output_height) & 
        (w_offsets[None, None, :] < output_width)
    ))

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.weight = torch.nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, depth, height, width = x.shape
        
        output_depth = (depth - 1) * self.stride - 2 * self.padding + \
                      self.dilation * (self.kernel_size - 1) + 1
        output_height = (height - 1) * self.stride - 2 * self.padding + \
                       self.dilation * (self.kernel_size - 1) + 1
        output_width = (width - 1) * self.stride - 2 * self.padding + \
                      self.dilation * (self.kernel_size - 1) + 1
        
        output = torch.empty(
            batch_size, 
            self.out_channels, 
            output_depth, 
            output_height, 
            output_width,
            device=x.device, 
            dtype=x.dtype
        )
        
        grid = lambda meta: (batch_size * self.out_channels * 
                            math.ceil(output_depth / meta['BLOCK_D']) * 
                            math.ceil(output_height / meta['BLOCK_H']) * 
                            math.ceil(output_width / meta['BLOCK_W']),)
        
        conv_transpose3d_kernel[grid](
            x,
            self.weight,
            output,
            self.in_channels,
            self.out_channels,
            depth,
            height,
            width,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            output_depth,
            output_height,
            output_width,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            self.weight.stride(0), self.weight.stride(1), 
            self.weight.stride(2), self.weight.stride(3), self.weight.stride(4),
            output.stride(0), output.stride(1), 
            output.stride(2), output.stride(3), output.stride(4)
        )
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
            
        return output

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
depth = 16
height = 32
width = 32
stride = 2
padding = 1
dilation = 2

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
# =================== EVOLVE-BLOCK-END ===================