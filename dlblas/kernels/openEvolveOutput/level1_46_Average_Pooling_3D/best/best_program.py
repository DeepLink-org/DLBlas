# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _avg_pool3d_forward(
    x_ptr, 
    output_ptr,
    # Tensor dimensions
    B, C, D, H, W,
    stride_b, stride_c, stride_d, stride_h, stride_w,
    output_D, output_H, output_W,
    divisor,
    # Constant expressions
    KERNEL_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_spatial = tl.program_id(2)
    
    # Reconstruct spatial indices
    hw_size = output_H * output_W
    d_idx = pid_spatial // hw_size
    hw_idx = pid_spatial % hw_size
    h_idx = hw_idx // output_W
    w_idx = hw_idx % output_W
    
    # Compute window start positions
    start_d = d_idx * STRIDE - PADDING
    start_h = h_idx * STRIDE - PADDING
    start_w = w_idx * STRIDE - PADDING
    
    # Accumulate values
    total = 0.0
    for kd in tl.static_range(KERNEL_SIZE):
        for kh in tl.static_range(KERNEL_SIZE):
            for kw in tl.static_range(KERNEL_SIZE):
                id = start_d + kd
                ih = start_h + kh
                iw = start_w + kw
                
                # Check boundaries
                within_bounds = (id >= 0) & (id < D) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                offset = pid_b * stride_b + pid_c * stride_c + id * stride_d + ih * stride_h + iw * stride_w
                val = tl.load(x_ptr + offset, mask=within_bounds, other=0.0)
                total += val
    
    # Compute average and store result
    out_offset = pid_b * (C * output_D * output_H * output_W) + pid_c * (output_D * output_H * output_W) + d_idx * (output_H * output_W) + h_idx * output_W + w_idx
    tl.store(output_ptr + out_offset, total / divisor)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        D, H, W = x.shape[2:]
        k = self.kernel_size
        s = self.stride
        p = self.padding
        
        # Compute output dimensions
        output_D = (D + 2 * p - k) // s + 1
        output_H = (H + 2 * p - k) // s + 1
        output_W = (W + 2 * p - k) // s + 1
        
        output = torch.empty((x.shape[0], x.shape[1], output_D, output_H, output_W), 
                             device=x.device, dtype=x.dtype)
        
        # Ensure contiguous memory
        x = x.contiguous()
        
        # Get tensor strides
        stride_b, stride_c, stride_d, stride_h, stride_w = x.stride()
        divisor = float(k ** 3)  # Precompute divisor
        
        # Total spatial elements per channel
        total_spatial = output_D * output_H * output_W
        
        # Configure grid (batch, channels, spatial)
        grid = (x.shape[0], x.shape[1], total_spatial)
        
        # Launch optimized kernel
        _avg_pool3d_forward[grid](
            x, output,
            x.shape[0], x.shape[1], D, H, W,
            stride_b, stride_c, stride_d, stride_h, stride_w,
            output_D, output_H, output_W,
            divisor,
            KERNEL_SIZE=k,
            STRIDE=s,
            PADDING=p,
            BLOCK_SIZE=1
        )
        return output

batch_size = 16
channels = 32
depth = 64
height = 64
width = 64
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    x = torch.randn(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]
# =================== EVOLVE-BLOCK-END ===================