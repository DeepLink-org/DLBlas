# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose3d_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    B, C_in, C_out,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    kernel_size,
    stride,
    padding,
    dilation,
    stride_x_b, stride_x_c, stride_x_d, stride_x_h, stride_x_w,
    stride_w_oc, stride_w_ic, stride_w_d, stride_w_h, stride_w_k,
    stride_out_b, stride_out_c, stride_out_d, stride_out_h, stride_out_w,
    num_blocks_d, num_blocks_h, num_blocks_w,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Program indices
    pid_b = tl.program_id(0)
    pid_spatial = tl.program_id(1)
    pid_oc = tl.program_id(2)
    
    # Decompose spatial block index
    total_blocks_hw = num_blocks_h * num_blocks_w
    block_idx_d = pid_spatial // total_blocks_hw
    remainder = pid_spatial % total_blocks_hw
    block_idx_h = remainder // num_blocks_w
    block_idx_w = remainder % num_blocks_w
    
    # Create block ranges
    c_offsets = pid_oc * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    d_offsets = block_idx_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    h_offsets = block_idx_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = block_idx_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_C, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over kernel dimensions
    for di in range(kernel_size):
        for dj in range(kernel_size):
            for dk in range(kernel_size):
                # Compute input indices
                d_in = (d_offsets[:, None, None, None] - di * dilation + padding) // stride
                h_in = (h_offsets[None, :, None, None] - dj * dilation + padding) // stride
                w_in = (w_offsets[None, None, :, None] - dk * dilation + padding) // stride
                
                # Check bounds and divisibility
                d_mask = (d_in >= 0) & (d_in < D_in) & (d_in * stride == d_offsets[:, None, None, None] - di * dilation + padding)
                h_mask = (h_in >= 0) & (h_in < H_in) & (h_in * stride == h_offsets[None, :, None, None] - dj * dilation + padding)
                w_mask = (w_in >= 0) & (w_in < W_in) & (w_in * stride == w_offsets[None, None, :, None] - dk * dilation + padding)
                valid_mask = d_mask & h_mask & w_mask
                
                # Adjust out-of-bound indices
                d_in = tl.where(d_mask, d_in, 0)
                h_in = tl.where(h_mask, h_in, 0)
                w_in = tl.where(w_mask, w_in, 0)
                
                # Iterate over input channels
                for c_in in range(0, C_in):
                    # Compute input pointer offsets
                    x_offset = (
                        pid_b * stride_x_b + 
                        c_in * stride_x_c + 
                        d_in * stride_x_d + 
                        h_in * stride_x_h + 
                        w_in * stride_x_w
                    )
                    x_val = tl.load(x_ptr + x_offset, mask=valid_mask, other=0.0)
                    
                    # Compute weight pointer offsets
                    w_offset = (
                        c_offsets[None, None, None, :] * stride_w_oc + 
                        c_in * stride_w_ic + 
                        di * stride_w_d + 
                        dj * stride_w_h + 
                        dk * stride_w_k
                    )
                    w_val = tl.load(weight_ptr + w_offset)
                    
                    # Accumulate with explicit broadcasting
                    acc += w_val[:, None, None, None] * x_val[None, :, :, :]
    
    # Store output
    for c in range(BLOCK_SIZE_C):
        for d in range(BLOCK_SIZE_D):
            for h in range(BLOCK_SIZE_H):
                for w in range(BLOCK_SIZE_W):
                    c_idx = c_offsets[c]
                    d_idx = d_offsets[d]
                    h_idx = h_offsets[h]
                    w_idx = w_offsets[w]
                    
                    if c_idx < C_out and d_idx < D_out and h_idx < H_out and w_idx < W_out:
                        out_offset = (
                            pid_b * stride_out_b + 
                            c_idx * stride_out_c + 
                            d_idx * stride_out_d + 
                            h_idx * stride_out_h + 
                            w_idx * stride_out_w
                        )
                        tl.store(output_ptr + out_offset, acc[c, d, h, w])

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, 
                 output_padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weight
        self.weight = nn.Parameter(torch.empty(
            out_channels, 
            in_channels,
            kernel_size, 
            kernel_size, 
            kernel_size
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
            
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute output shape
        B, C_in, D_in, H_in, W_in = x.shape
        D_out = (D_in - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + self.output_padding + 1
        H_out = (H_in - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + self.output_padding + 1
        W_out = (W_in - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + self.output_padding + 1
        
        # Create output tensor
        output = torch.empty(B, self.out_channels, D_out, H_out, W_out, 
                             device=x.device, dtype=x.dtype)
        
        # Precompute strides
        stride_x = x.stride()
        stride_w = self.weight.stride()
        stride_out = output.stride()
        
        # Define block sizes
        BLOCK_SIZE_C = 16
        BLOCK_SIZE_D = 4
        BLOCK_SIZE_H = 4
        BLOCK_SIZE_W = 4
        
        # Precompute block counts
        num_blocks_d = (D_out + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D
        num_blocks_h = (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
        num_blocks_w = (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
        total_spatial_blocks = num_blocks_d * num_blocks_h * num_blocks_w
        
        # Grid dimensions
        grid_c = (self.out_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
        grid = (B, total_spatial_blocks, grid_c)
        
        # Launch kernel
        conv_transpose3d_kernel[grid](
            x, self.weight, output,
            B, C_in, self.out_channels,
            D_in, H_in, W_in,
            D_out, H_out, W_out,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            stride_x[0], stride_x[1], stride_x[2], stride_x[3], stride_x[4],
            stride_w[0], stride_w[1], stride_w[2], stride_w[3], stride_w[4],
            stride_out[0], stride_out[1], stride_out[2], stride_out[3], stride_out[4],
            num_blocks_d, num_blocks_h, num_blocks_w,
            BLOCK_SIZE_C,
            BLOCK_SIZE_D,
            BLOCK_SIZE_H,
            BLOCK_SIZE_W,
        )
        
        # Add bias if needed
        if self.bias is not None:
            output += self.bias[None, :, None, None, None]
            
        return output

# Test code
batch_size = 16
in_channels = 32
out_channels = 16
kernel_size = 3
depth = 16
height = 32
width = 64

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
# =================== EVOLVE-BLOCK-END ===================