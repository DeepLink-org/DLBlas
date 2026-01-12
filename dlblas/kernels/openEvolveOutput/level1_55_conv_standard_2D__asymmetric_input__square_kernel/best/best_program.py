# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
            
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_C': 32}, num_warps=4),
            triton.Config({'BLOCK_C': 64}, num_warps=4),
            triton.Config({'BLOCK_C': 128}, num_warps=4),
            triton.Config({'BLOCK_C': 32}, num_warps=8),
            triton.Config({'BLOCK_C': 64}, num_warps=8),
            triton.Config({'BLOCK_C': 128}, num_warps=8),
        ],
        key=['C_in', 'H_out', 'W_out', 'kernel_size'],
    )
    @triton.jit
    def _conv_forward_kernel(
        # Tensors
        x_ptr, weight_ptr, output_ptr,
        # Input dimensions
        B, C_in, H, W,
        # Output dimensions
        C_out, kernel_size, H_out, W_out,
        # Strides
        stride_xb, stride_xc, stride_xh, stride_xw,
        stride_woc, stride_wic, stride_wh, stride_ww,
        stride_ob, stride_oc, stride_oh, stride_ow,
        # Convolution parameters
        stride_h: tl.constexpr, stride_w: tl.constexpr,
        padding_h: tl.constexpr, padding_w: tl.constexpr,
        dilation_h: tl.constexpr, dilation_w: tl.constexpr,
        # Blocking
        BLOCK_C: tl.constexpr,
    ):
        # Program IDs
        pid_b = tl.program_id(0)
        pid_oc = tl.program_id(1)
        pid_oh_ow = tl.program_id(2)
        
        oh = pid_oh_ow // W_out
        ow = pid_oh_ow % W_out
        
        # Initialize accumulator
        acc = 0.0
        
        # Loop over input channels in blocks
        for ic0 in range(0, C_in, BLOCK_C):
            ic_offsets = ic0 + tl.arange(0, BLOCK_C)
            c_mask = ic_offsets < C_in
            
            # Loop over kernel positions
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    # Calculate input position with dilation
                    h_in = oh * stride_h - padding_h + kh * dilation_h
                    w_in = ow * stride_w - padding_w + kw * dilation_w
                    
                    # Check input boundaries
                    in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                    
                    # Load input block
                    x_ptrs = (
                        x_ptr + 
                        pid_b * stride_xb + 
                        ic_offsets * stride_xc + 
                        h_in * stride_xh + 
                        w_in * stride_xw
                    )
                    x_vals = tl.load(x_ptrs, mask=c_mask & in_bounds, other=0.0)
                    
                    # Load weight block
                    w_ptrs = (
                        weight_ptr + 
                        pid_oc * stride_woc + 
                        ic_offsets * stride_wic + 
                        kh * stride_wh + 
                        kw * stride_ww
                    )
                    w_vals = tl.load(w_ptrs, mask=c_mask, other=0.0)
                    
                    # Update accumulator
                    acc += tl.sum(x_vals * w_vals)
        
        # Output pointer
        out_ptr = (
            output_ptr + 
            pid_b * stride_ob + 
            pid_oc * stride_oc + 
            oh * stride_oh + 
            ow * stride_ow
        )
        tl.store(out_ptr, acc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.groups != 1:
            return nn.functional.conv2d(
                x, self.weight, self.bias, self.stride, 
                self.padding, self.dilation, self.groups
            )
            
        # Calculate output dimensions
        B, C_in, H, W = x.shape
        H_out = (H + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
                ) // self.stride + 1
        W_out = (W + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
                ) // self.stride + 1
        
        # Prepare output
        output = torch.empty(
            (B, self.out_channels, H_out, W_out),
            device=x.device, dtype=x.dtype
        )
        
        # Launch kernel
        grid = (B, self.out_channels, H_out * W_out)
        self._conv_forward_kernel[grid](
            x, self.weight, output,
            B, C_in, H, W,
            self.out_channels, self.kernel_size, H_out, W_out,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.weight.stride(0), self.weight.stride(1), 
            self.weight.stride(2), self.weight.stride(3),
            output.stride(0), output.stride(1), 
            output.stride(2), output.stride(3),
            self.stride, self.stride,
            self.padding, self.padding,
            self.dilation, self.dilation
        )
        
        # Add bias if needed
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
            
        return output

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 128  # Asymmetric input

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
# =================== EVOLVE-BLOCK-END ===================