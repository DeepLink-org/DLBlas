# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def _conv_transpose2d_kernel(
    x_ptr, 
    w_ptr,
    b_ptr,
    output_ptr,
    stride_h, stride_w,
    padding_h, padding_w,
    output_padding_h, output_padding_w,
    groups,
    in_channels, out_channels,
    kernel_size,
    in_h, in_w,
    out_h, out_w,
    x_batch_stride, x_channel_stride, x_h_stride, x_w_stride,
    w_in_channel_stride, w_out_channel_stride, w_h_stride, w_w_stride,
    output_batch_stride, output_channel_stride, output_h_stride, output_w_stride,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    GRID_W: tl.constexpr,  # Number of blocks in width dimension
):
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    # Decompose combined height-width dimension
    pid_h = pid_hw // GRID_W
    pid_w = pid_hw % GRID_W
    
    # Create block pointers
    channel_offsets = pid_channel * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    h_offsets = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for boundary checks
    channel_mask = channel_offsets < out_channels
    h_mask = h_offsets < out_h
    w_mask = w_offsets < out_w
    
    # Initialize 3D accumulator (height, width, channels)
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C), dtype=tl.float32)
    
    # Loop over kernel dimensions and input channels
    for di in range(kernel_size):
        for dj in range(kernel_size):
            for c_in in range(in_channels):
                # Compute input indices
                i_input = (h_offsets[:, None] - di + padding_h) // stride_h
                j_input = (w_offsets[None, :] - dj + padding_w) // stride_w
                
                # Check bounds and divisibility
                i_in_bounds = (i_input >= 0) & (i_input < in_h)
                j_in_bounds = (j_input >= 0) & (j_input < in_w)
                valid_i = (h_offsets[:, None] - di + padding_h) % stride_h == 0
                valid_j = (w_offsets[None, :] - dj + padding_w) % stride_w == 0
                valid_mask = i_in_bounds & j_in_bounds & valid_i & valid_j
                
                # Compute offsets and load input
                x_offset = (pid_batch * x_batch_stride + 
                           c_in * x_channel_stride + 
                           i_input * x_h_stride + 
                           j_input * x_w_stride)
                x_val = tl.load(x_ptr + x_offset, mask=valid_mask, other=0.0)
                
                # Load weights with 1D offset
                base = c_in * w_in_channel_stride + di * w_h_stride + dj * w_w_stride
                w_offset = base + channel_offsets * w_out_channel_stride
                w_val = tl.load(w_ptr + w_offset, mask=channel_mask, other=0.0)
                
                # Compute outer product via broadcasting (replace matrix multiplication)
                # Expand dimensions: [H, W, 1] * [1, 1, C] -> [H, W, C]
                product_3d = x_val[:, :, None] * w_val[None, None, :]
                acc += product_3d
    
    # Add bias if exists
    if b_ptr is not None:
        bias = tl.load(b_ptr + channel_offsets, mask=channel_mask, other=0.0)
        bias_3d = bias[None, None, :]  # Broadcast to [1, 1, C]
        acc += bias_3d
    
    # Prepare 3D indices for output
    h_idx = h_offsets[:, None, None]  # [H, 1, 1]
    w_idx = w_offsets[None, :, None]  # [1, W, 1]
    c_idx = channel_offsets[None, None, :]  # [1, 1, C]
    
    # Compute output offsets
    output_offset = (pid_batch * output_batch_stride +
                    c_idx * output_channel_stride +
                    h_idx * output_h_stride +
                    w_idx * output_w_stride)
    
    # Create 3D mask
    mask_3d = h_mask[:, None, None] & w_mask[None, :, None] & channel_mask[None, None, :]
    
    tl.store(output_ptr + output_offset, acc, mask=mask_3d)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 groups:极 int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.output_padding = (output_padding, output_padding)
        self.groups = groups
        
        # Initialize weights
        self.weight = nn.Parameter(torch.empty(
            in_channels, 
            out_channels // groups, 
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
        batch_size, in_channels, in_h, in_w = x.shape
        
        # Calculate output dimensions
        out_h = (in_h - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size + self.output_padding[0]
        out_w = (in_w - 1) * self.stride[1] - 极2 * self.padding[1] + self.kernel_size + self.output_padding[1]
        
        # Create output tensor
        output = torch.empty(
            batch_size, 
            self.out_channels, 
            out_h, 
            out_w, 
            device=x.device, 
            dtype=x.dtype
        )
        
        # Compute strides
        x_stride = x.stride()
        w_stride = self.weight.stride()
        
        # Define block sizes (tunable)
        BLOCK_SIZE_H = 16
        BLOCK_SIZE_W = 16
        BLOCK_SIZE_C = 32
        
        # Grid configuration
        grid_h = triton.cdiv(out_h, BLOCK_SIZE_H)
        grid_w = triton.cdiv(out_w, BLOCK_SIZE_W)
        grid_c = triton.cdiv(self.out_channels, BLOCK_SIZE_C)
        grid_hw = grid_h * grid_w
        
        # Launch kernel
        _conv_transpose2d_kernel[(
            batch_size, 
            grid_c, 
            grid_hw
        )](
            x, self.weight, self.bias, output,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.output_padding[0], self.output_padding[1],
            self.groups,
            self.in_channels, self.out_channels,
            self.kernel_size,
            in_h, in_w,
            out_h, out_w,
            x_stride[0], x_stride[1], x_stride[2], x_stride[3],
            w_stride[0], w_stride[1], w_stride[2], w_stride[3],
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C,
            grid_w  # Pass grid_w for dimension decomposition
        )
        
        return output

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
height_in = 128
width_in = 256

def get_inputs():
    x = torch.randn(batch_size, in_channels, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
# =================== EVOLVE-BLOCK-END ===================