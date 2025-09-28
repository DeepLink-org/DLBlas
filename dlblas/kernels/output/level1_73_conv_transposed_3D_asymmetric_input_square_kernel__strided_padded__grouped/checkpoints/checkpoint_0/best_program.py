# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose3d_kernel(
    # Pointers to input, weights, output, and bias
    x_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    # Tensor dimensions
    B, C_in, D_in, H_in, W_in,
    C_out, D_out, H_out, W_out,
    # Kernel parameters
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    groups: tl.constexpr,
    # Blocking parameters
    BLOCK_B: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_KD: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    BLOCK_KW: tl.constexpr,
):
    # Program ID mapping
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    # Group handling
    group_id = pid_c // (C_out // groups)
    in_channels_per_group = C_in // groups
    out_channels_per_group = C_out // groups
    c_local = pid_c % out_channels_per_group
    
    # Create ranges
    b_offsets = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    c_offsets = group_id * in_channels_per_group + tl.arange(0, BLOCK_C)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Initialize output
    output = tl.zeros((BLOCK_B, BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Precompute kernel ranges
    kd_offsets = tl.arange(0, BLOCK_KD)
    kh_offsets = tl.arange(0, BLOCK_KH)
    kw_offsets = tl.arange(0, BLOCK_KW)
    
    # Load kernel weights into shared memory
    weight = tl.load(
        weight_ptr + 
        (pid_c * in_channels_per_group * kernel_size * kernel_size * kernel_size) +
        (tl.arange(0, BLOCK_C)[:, None, None, None] * kernel_size * kernel_size * kernel_size) +
        (kd_offsets[None, :, None, None] * kernel_size * kernel_size) +
        (kh_offsets[None, None, :, None] * kernel_size) +
        kw_offsets[None, None, None, :],
        mask=tl.arange(0, BLOCK_C)[:, None, None, None] < in_channels_per_group,
        other=0.0
    )
    
    # Main computation
    for kd in range(kernel_size):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Compute input coordinates
                d_in = (d_offsets + padding - kd) / stride
                h_in = (h_offsets + padding - kh) / stride
                w_in = (w_offsets + padding - kw) / stride
                
                # Create masks
                d_mask = (d_in >= 0) & (d_in < D_in) & (d_in % 1 == 0)
                h_mask = (h_in >= 0) & (h_in < H_in) & (h_in % 1 == 0)
                w_mask = (w_in >= 0) & (w_in < W_in) & (w_in % 1 == 0)
                valid_mask = d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]
                
                # Convert to integer indices
                d_in_idx = tl.floor(d_in).to(tl.int32)
                h_in_idx = tl.floor(h_in).to(tl.int32)
                w_in_idx = tl.floor(w_in).to(tl.int32)
                
                # Load input values
                input_val = tl.load(
                    x_ptr +
                    (b_offsets[:, None, None, None] * C_in * D_in * H_in * W_in) +
                    (c_offsets[None, :, None, None, None] * D_in * H_in * W_in) +
                    (d_in_idx[None, None, :, None, None] * H_in * W_in) +
                    (h_in_idx[None, None, None, :, None] * W_in) +
                    w_in_idx[None, None, None, None, :],
                    mask=valid_mask[None, None, :, :, :] & 
                         (b_offsets[:, None, None, None, None] < B) &
                         (c_offsets[None, :, None, None, None] < (group_id + 1) * in_channels_per_group),
                    other=0.0
                )
                
                # Apply weights
                weight_val = weight[:, kd, kh, kw]
                output += tl.sum(input_val * weight_val[None, :, None, None, None], axis=1)
    
    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + pid_c)
        output += bias
    
    # Store results
    output_offset = (
        b_offsets[:, None, None, None] * C_out * D_out * H_out * W_out +
        pid_c * D_out * H_out * W_out +
        d_offsets[None, :, None, None] * H_out * W_out +
        h_offsets[None, None, :, None] * W_out +
        w_offsets[None, None, None, :]
    )
    
    tl.store(
        output_ptr + output_offset,
        output,
        mask=(
            (b_offsets[:, None, None, None] < B) &
            (d_offsets[None, :, None, None] < D_out) &
            (h_offsets[None, None, :, None] < H_out) &
            (w_offsets[None, None, None, :] < W_out)
        )
    )

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        
        # Weight parameter initialization
        self.weight = nn.Parameter(torch.empty(
            out_channels,
            in_channels // groups,
            kernel_size,
            kernel_size,
            kernel_size
        ))
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
        # Calculate output dimensions
        D_in, H_in, W_in = x.shape[2:]
        D_out = (D_in - 1) * self.stride + self.kernel_size - 2 * self.padding
        H_out = (H_in - 1) * self.stride + self.kernel_size - 2 * self.padding
        W_out = (W_in - 1) * self.stride + self.kernel_size - 2 * self.padding
        
        # Create output tensor
        output = torch.empty(
            (x.shape[0], self.out_channels, D_out, H_out, W_out),
            device=x.device,
            dtype=x.dtype
        )
        
        # Grid configuration
        grid = lambda META: (
            triton.cdiv(x.shape[0], META['BLOCK_B']),
            self.out_channels,
            triton.cdiv(D_out, META['BLOCK_D']),
            triton.cdiv(H_out, META['BLOCK_H']),
            triton.cdiv(W_out, META['BLOCK_W'])
        )
        
        # Launch kernel
        conv_transpose3d_kernel[grid](
            x, self.weight, output, self.bias,
            x.shape[0], self.in_channels, D_in, H_in, W_in,
            self.out_channels, D_out, H_out, W_out,
            self.kernel_size,
            self.stride,
            self.padding,
            self.groups,
            BLOCK_B=4,
            BLOCK_C=8,
            BLOCK_D=4,
            BLOCK_H=8,
            BLOCK_W=8,
            BLOCK_KD=1,
            BLOCK_KH=1,
            BLOCK_KW=1
        )
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
padding = 3
groups = 4

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups]
# =================== EVOLVE-BLOCK-END ===================