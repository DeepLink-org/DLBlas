# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def _conv3d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    B, C_in, C_out, D, H, W, 
    Kd, Kh, Kw,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil极, dil_h, dil_w,
    groups,
    D_out, H_out, W_out,
    x_bs, x_cs, x_ds, x_hs, x_ws,
    w_ocs, w_ics, w_ds, w_hs, w_ws,
    y_bs, y_cs, y_ds, y_hs, y_ws,
    BLOCK_IC: tl.constexpr,
    HAVE_BIAS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    num_blocks_d: tl.constexpr,
    num_blocks_h: tl.constexpr,
    num_blocks_w: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_spatial = tl.program_id(2)
    
    # Decompose spatial block index
    pid_w = pid_spatial % num_blocks_w
    pid_h = (pid_spatial // num_blocks_w) % num_blocks_h
    pid_d = pid_spatial // (num_blocks_w * num_blocks_h)
    
    d_offset = pid_d * BLOCK_D
    h_offset = pid_h * BLOCK_H
    w_offset = pid_w * BLOCK_W

    # Compute output channel group
    group_size = C_out // groups
    group_idx = pid_oc // group_size
    oc_in_group = pid_oc % group_size
    ic_per_group = C_in // groups
    ic_start = group_idx * ic_per_group

    # Offsets for spatial block
    d_offsets = d_offset + tl.arange(0, BLOCK_D)
    h_offsets = h_offset + tl.arange(0, BLOCK_H)
    w_offsets = w_offset + tl.arange(0, BLOCK_W)
    
    # Masks for boundary checks
    d_mask = d_offsets < D_out
    h_mask = h_offsets < H_out
    w_mask = w_offsets < W_out
    spatial_mask = d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for ic_block in range(0, ic_per_group, BLOCK_IC):
        ic_offsets = ic_block + tl.arange(0, BLOCK_IC)
        ic_mask = ic_offsets < ic_per_group
        
        # Loop over kernel dimensions
        for kd in range(Kd):
            for kh in range(Kh):
                for kw in range(Kw):
                    # Compute input positions
                    in_d = d_offsets * stride_d - pad_d + kd * dil_d
                    in_h = h_offsets * stride_h - pad_h + kh * dil_h
                    in_w = w_offsets * stride_w - pad_w + kw * dil_w
                    
                    # Check input boundaries with proper broadcasting
                    in_d_mask = (in_d >= 0) & (in_d < D)
                    in_h_mask = (in_h >= 0) & (in_h < H)
                    in_w_mask = (in_w >= 0) & (in_w < W)
                    # Reshape masks to 4D for broadcasting
                    in_mask = (
                        in_d_mask[:, None, None, None] & 
                        in_h_mask[None, :, None, None] & 
                        in_w_mask[None, None, :, None] & 
                        ic_mask[None, None, None, :]
                    )
                    
                    # Load input block
                    x_ptrs = (
                        x_ptr + 
                        pid_b * x_bs + 
                        (ic_start + ic_offsets)[None, None, None, :] * x_c极 + 
                        in_d[:, None, None, None] * x_ds + 
                        in_h[None, :, None, None] * x_hs + 
                        in_w[None, None, :, None] * x_ws
                    )
                    x_block = tl.load(x_ptrs, mask=in_mask, other=0.0)
                    
                    # Load weight block
                    w_ptrs = (
                        w_ptr + 
                        pid_oc * w_ocs + 
                        ic_offsets[None, None, None, :] * w_ics + 
                        kd * w_ds + 
                        kh * w_hs + 
                        kw * w_ws
                    )
                    w_block = tl.load(w_ptrs, mask=ic_mask, other=0.0)
                    
                    # Accumulate
                    w_val = w_block[0, 0, 0, :]
                    acc += tl.sum(x_block * w_val, axis=3)
    
    # Add bias if present
    if HAVE_BIAS:
        bias = tl.load(b_ptr + pid_oc)
        acc += bias
    
    # Store output
    y_ptrs = (
        y_ptr + 
        pid_b * y_bs + 
        pid_oc * y_cs + 
        d_offsets[:, None, None] * y_ds + 
        h_offsets[None, :, None] * y_hs + 
        w_offsets[None, None, :] * y_ws
    )
    tl.store(y_ptrs, acc, mask=spatial_mask)

def conv3d_triton(x, weight, bias, stride, padding, dilation, groups):
    B, C_in, D, H, W = x.shape
    C_out, C_g, Kd, Kh, Kw = weight.shape
    assert C_in // groups == C_g
    
    # Calculate output dimensions
    D_out = (D + 2 * padding[0] - dilation[0] * (Kd - 1) - 1) // stride[0] + 1
    H_out = (H + 2 * padding[1] - dilation[1] * (Kh - 1) - 1) // stride[1] + 1
    W_out = (W + 2 * padding[2] - dilation[2] * (Kw - 1) - 1) // stride[2] + 1
    
    # Create output tensor
    y = torch.empty((B, C_out, D_out, H_out, W_out), 
                   device=x.device, dtype=x.dtype)
    
    # Define block dimensions
    BLOCK_D, BLOCK_H, BLOCK_W = 8, 8, 8
    num_blocks_d = triton.cdiv(D_out, BLOCK_D)
    num_blocks_h = triton.cdiv(H_out, BLOCK_H)
    num_blocks_w = triton.cdiv(W_out, BLOCK_W)
    total_spatial_blocks = num_blocks_d * num_blocks_h * num_blocks_w
    
    # Define grid
    grid = (B, C_out, total_spatial_blocks)
    
    # Launch kernel
    _conv3d_kernel[grid](
        x, weight, bias, y,
        B, C_in, C_out, D, H, W, 
        Kd, Kh, Kw,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        groups,
        D_out, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4),
        BLOCK_IC=32,
        HAVE_BIAS=bias is not None,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        num_blocks_d=num_blocks_d,
        num_blocks_h=num_blocks_h,
        num_blocks_w=num_blocks_w
    )
    return y

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights
        self.weight = nn.Parameter(torch.empty(
            out_channels,
            in_channels // groups,
            kernel_size[0],
            kernel_size[1],
            kernel_size[2]
        ))
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Reset parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv3d_triton(
            x, 
            self.weight, 
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

# Test code
import math
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)  # Asymmetric kernel size
depth = 16
height = 256
width = 256

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width, device='cuda')
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
# =================== EVOLVE-BLOCK-END ===================