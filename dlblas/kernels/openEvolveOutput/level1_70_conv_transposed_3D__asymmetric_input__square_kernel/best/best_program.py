# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _conv_transpose3d_kernel(
    # Pointers to tensors
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    # Input dimensions
    batch_size, in_channels, D_in, H_in, W_in,
    # Output dimensions
    D_out, H_out, W_out,
    # Kernel parameters
    kernel_size, stride, padding, dilation, groups,
    # Tensor strides
    stride_xb, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wic, stride_woc, stride_wd, stride_wh, stride_ww,
    stride_ob, stride_oc, stride_od, stride_oh, stride_ow,
    # Tile sizes
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    # 3D grid indices
    pid_b = tl.program_id(0)              # Batch index
    pid_oc = tl.program_id(1)              # Output channel index
    pid_tile = tl.program_id(2)            # Spatial tile index
    
    # Calculate number of tiles in each spatial dimension
    tiles_d = tl.cdiv(D_out, BLOCK_D)
    tiles_h = tl.cdiv(H_out, BLOCK_H)
    tiles_w = tl.cdiv(W_out, BLOCK_W)
    tiles_hw = tiles_h * tiles_w
    
    # Decompose spatial tile index into d, h, w components
    tile_d = pid_tile // tiles_hw
    tile_hw = pid_tile % tiles_hw
    tile_h = tile_hw // tiles_w
    tile_w = tile_hw % tiles_w
    
    # Calculate starting indices for the current tile
    d_start = tile_d * BLOCK_D
    h_start = tile_h * BLOCK_H
    w_start = tile_w * BLOCK_W
    
    # Create position offsets within the tile
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    
    # Create masks for boundary checks
    d_mask = d_offsets < D_out
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    block_mask = d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Group handling (assuming groups=1 for simplicity)
    # For groups>1, we would adjust the input channel range
    c_start = 0
    c_end = in_channels
    
    # Loop over input channels
    for c_in in range(c_start, c_end):
        # Loop over kernel dimensions
        for kd in range(kernel_size):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    # Calculate input positions
                    d_in = (d_offsets[:, None, None] - kd * dilation + padding) // stride
                    h_in = (h_offsets[None, :, None] - kh * dilation + padding) // stride
                    w_in = (w_offsets[None, None, :] - kw * dilation + padding) // stride
                    
                    # Check divisibility and bounds
                    d_div = (d_offsets[:, None, None] - kd * dilation + padding) % stride == 0
                    h_div = (h_offsets[None, :, None] - kh * dilation + padding) % stride == 0
                    w_div = (w_offsets[None, None, :] - kw * dilation + padding) % stride == 0
                    valid = d_div & h_div & w_div
                    
                    d_in_bound = (d_in >= 0) & (d_in < D_in)
                    h_in_bound = (h_in >= 0) & (h_in < H_in)
                    w_in_bound = (w_in >= 0) & (w_in < W_in)
                    in_bound = d_in_bound & h_in_bound & w_in_bound
                    
                    mask = valid & in_bound & block_mask
                    
                    # Compute input pointer offsets
                    input_offsets = (
                        pid_b * stride_xb + 
                        c_in * stride_xc + 
                        d_in * stride_xd + 
                        h_in * stride_xh + 
                        w_in * stride_xw
                    )
                    
                    # Load input values with masking
                    input_vals = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
                    
                    # Load weight value
                    weight_val = tl.load(
                        weight_ptr + 
                        c_in * stride_wic + 
                        pid_oc * stride_woc + 
                        kd * stride_wd + 
                        kh * stride_wh + 
                        kw * stride_ww
                    )
                    
                    # Accumulate
                    acc += input_vals * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + pid_oc)
        acc += bias_val
    
    # Compute output pointer offsets
    output_offsets = (
        pid_b * stride_ob + 
        pid_oc * stride_oc + 
        d_offsets[:, None, None] * stride_od + 
        h_offsets[None, :, None] * stride_oh + 
        w_offsets[None, None, :] * stride_ow
    )
    
    # Store results
    tl.store(output_ptr + output_offsets, acc, mask=block_mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, 
                 output_padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        
        # Weight parameters
        self.weight = nn.Parameter(torch.empty(
            in_channels, 
            out_channels, 
            kernel_size, 
            kernel_size, 
            kernel_size
        ))
        
        # Bias parameter
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, D_in, H_in, W_in = x.shape
        
        # Calculate output dimensions
        D_out = (D_in - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + self.output_padding + 1
        H_out = (H_in - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + self.output_padding + 1
        W_out = (W_in - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + self.output_padding + 1
        
        # Create output tensor
        output = torch.empty(
            batch_size, 
            self.out_channels, 
            D_out, 
            H_out, 
            W_out, 
            device=x.device, 
            dtype=x.dtype
        )
        
        # Calculate number of tiles with optimized block sizes
        BLOCK_D, BLOCK_H, BLOCK_W = 8, 4, 4  # Optimized tile size
        tiles_d = (D_out + BLOCK_D - 1) // BLOCK_D
        tiles_h = (H_out + BLOCK_H - 1) // BLOCK_H
        tiles_w = (W_out + BLOCK_W - 1) // BLOCK_W
        tiles_hw = tiles_h * tiles_w
        total_tiles = tiles_d * tiles_hw
        
        # Grid configuration
        grid = (batch_size, self.out_channels, total_tiles)
        
        # Launch kernel
        _conv_transpose3d_kernel[grid](
            x, self.weight, self.bias, output,
            batch_size, in_channels, D_in, H_in, W_in,
            D_out, H_out, W_out,
            self.kernel_size, self.stride, self.padding, self.dilation, self.groups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            self.weight.stride(0), self.weight.stride(1), self.weight.stride(2), self.weight.stride(3), self.weight.stride(4),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
            BLOCK_D, BLOCK_H, BLOCK_W
        )
        
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