# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def fused_conv_gn_kernel(
    x_ptr,
    conv_weight_ptr,
    gn_weight_ptr,
    gn_bias_ptr,
    output_ptr,
    # Tensor dimensions
    B, C_in, C_out, D, H, W,
    # Group norm parameters
    num_groups,
    eps,
    # Blocking parameters
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Compute 3D position indices
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    pid_group = tl.program_id(4)
    
    # Group handling
    group_size = C_out // num_groups
    c_start = pid_group * group_size
    c_end = c_start + group_size
    
    # Create ranges
    c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
    d_offsets = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    h_offsets = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for boundary checks
    c_mask = c_offsets < c_end
    d_mask = d_offsets < D
    h_mask = h_offsets < H
    w_mask = w_offsets < W
    spatial_mask = d_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]
    
    # Initialize output accumulator
    output = tl.zeros((BLOCK_SIZE_C, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Convolution parameters
    kernel_radius = 1  # kernel_size=3
    for kd in range(3):
        for kh in range(3):
            for kw in range(3):
                # Input position
                d_in = d_offsets - (kd - kernel_radius)
                h_in = h_offsets - (kh - kernel_radius)
                w_in = w_offsets - (kw - kernel_radius)
                
                # Boundary checks for input
                d_in_valid = (d_in >= 0) & (d_in < D)
                h_in_valid = (h_in >= 0) & (h_in < H)
                w_in_valid = (w_in >= 0) & (w_in < W)
                in_mask = d_in_valid & h_in_valid & w_in_valid
                
                # Load input tile
                for c_in in range(0, C_in):
                    x_val = tl.load(
                        x_ptr + pid_b * C_in * D * H * W + 
                        c_in * D * H * W +
                        d_in * H * W +
                        h_in * W +
                        w_in,
                        mask=in_mask & spatial_mask,
                        other=0.0
                    )
                    
                    # Load weight
                    weight = tl.load(
                        conv_weight_ptr + 
                        (c_offsets[:, None, None, None] - c_start) * C_in * 3 * 3 * 3 +
                        c_in * 3 * 3 * 3 +
                        kd * 3 * 3 +
                        kh * 3 +
                        kw,
                        mask=c_mask[:, None, None, None] & spatial_mask,
                        other=0.0
                    )
                    
                    # Accumulate convolution
                    output += x_val * weight
    
    # Apply ReLU
    output = tl.where(output > 0, output, 0.0)
    
    # Group normalization - compute mean and variance
    group_mean = tl.sum(output, axis=[1, 2, 3]) / (D * H * W)
    group_var = tl.sum((output - group_mean[:, None, None, None]) ** 2, axis=[1, 2, 3]) / (D * H * W)
    
    # Normalize and scale
    normalized = (output - group_mean[:, None, None, None]) / tl.sqrt(group_var[:, None, None, None] + eps)
    
    # Load group norm parameters
    gn_scale = tl.load(gn_weight_ptr + c_offsets, mask=c_mask)
    gn_bias_val = tl.load(gn_bias_ptr + c_offsets, mask=c_mask)
    
    # Apply affine transformation
    normalized = normalized * gn_scale[:, None, None, None] + gn_bias_val[:, None, None, None]
    
    # Store results
    for c in range(BLOCK_SIZE_C):
        for d in range(BLOCK_SIZE_D):
            for h in range(BLOCK_SIZE_H):
                for w in range(BLOCK_SIZE_W):
                    if c_mask[c] and d_mask[d] and h_mask[h] and w_mask[w]:
                        out_idx = (
                            pid_b * C_out * D * H * W +
                            (c_start + c) * D * H * W +
                            (pid_d * BLOCK_SIZE_D + d) * H * W +
                            (pid_h * BLOCK_SIZE_H + h) * W +
                            (pid_w * BLOCK_SIZE_W + w)
                        )
                        tl.store(output_ptr + out_idx, normalized[c, d, h, w])

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        
        # Convolution weights
        self.conv_weight = nn.Parameter(torch.empty(
            out_channels, in_channels, kernel_size, kernel_size, kernel_size
        ))
        # Group normalization parameters
        self.gn_weight = nn.Parameter(torch.ones(out_channels))
        self.gn_bias = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize weights
        nn.init.kaiming_normal_(self.conv_weight, nonlinearity='relu')
        
    def forward(self, x):
        B, C_in, D, H, W = x.shape
        C_out = self.out_channels
        
        # Allocate output tensor
        output = torch.empty(B, C_out, D, H, W, device=x.device, dtype=x.dtype)
        
        # Grid configuration
        grid_dim_d = triton.cdiv(D, 8)
        grid_dim_h = triton.cdiv(H, 8)
        grid_dim_w = triton.cdiv(W, 8)
        
        # Launch kernel
        fused_conv_gn_kernel[(
            B, 
            grid_dim_d, 
            grid_dim_h, 
            grid_dim_w, 
            self.groups
        )](
            x, 
            self.conv_weight, 
            self.gn_weight, 
            self.gn_bias, 
            output,
            B, C_in, C_out, D, H, W,
            self.groups,
            1e-5,
            BLOCK_SIZE_C=16,
            BLOCK_SIZE_D=8,
            BLOCK_SIZE_H=8,
            BLOCK_SIZE_W=8,
        )
        
        return output

batch_size = 16
in_channels = 64
out_channels = 128
D, H, W = 8, 16, 16
kernel_size = 3
groups = 8
bias = False

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, bias]
# =================== EVOLVE-BLOCK-END ===================