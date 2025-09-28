# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def max_pool3d_kernel(
    input_ptr,
    output_ptr,
    # Tensor dimensions
    B, C, D, H, W,
    D_out, H_out, W_out,
    # Input tensor strides
    input_stride_b, input_stride_c, input_stride_d, input_stride_h, input_stride_w,
    # Output tensor strides
    output_stride_b, output_stride_c, output_stride_d, output_stride_h, output_stride_w,
    # Pooling parameters
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    dilation_d, dilation_h, dilation_w,
    # Block size (unused in this implementation but kept for compatibility)
    BLOCK_SIZE: tl.constexpr
):
    # Program indices
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_s = tl.program_id(2)
    
    # Decompose spatial index
    hw_size = H_out * W_out
    d_out = pid_s // hw_size
    hw = pid_s % hw_size
    h_out = hw // W_out
    w_out = hw % W_out

    # Check bounds
    if pid_b >= B or pid_c >= C or d_out >= D_out or h_out >= H_out or w_out >= W_out:
        return

    # Calculate starting indices
    start_d = d_out * stride_d - padding_d
    start_h = h_out * stride_h - padding_h
    start_w = w_out * stride_w - padding_w
    
    # Initialize max value
    max_val = float('-inf')
    
    # Optimized for common kernel size 3x3x3
    if kernel_d == 3 and kernel_h == 3 and kernel_w == 3:
        for kd in range(0, 3):
            d_in = start_d + kd * dilation_d
            for kh in range(0, 3):
                h_in = start_h + kh * dilation_h
                for kw in range(0, 3):
                    w_in = start_w + kw * dilation_w
                    
                    # Check bounds
                    in_bounds = (d_in >= 0) & (d_in < D) & \
                                (h_in >= 0) & (h_in < H) & \
                                (w_in >= 0) & (w_in < W)
                    
                    if in_bounds:
                        # Calculate offset
                        offset = (
                            pid_b * input_stride_b + 
                            pid_c * input_stride_c + 
                            d_in * input_stride_d + 
                            h_in * input_stride_h + 
                            w_in * input_stride_w
                        )
                        val = tl.load(input_ptr + offset)
                    else:
                        val = float('-inf')
                    
                    # Update max value
                    if val > max_val:
                        max_val = val
    else:
        # Generic kernel size handling
        for kd in range(0, kernel_d):
            d_in = start_d + kd * dilation_d
            for kh in range(0, kernel_h):
                h_in = start_h + kh * dilation_h
                for kw in range(0, kernel_w):
                    w_in = start_w + kw * dilation_w
                    
                    # Check bounds
                    in_bounds = (d_in >= 0) & (d_in < D) & \
                                (h_in >= 0) & (h_in < H) & \
                                (w_in >= 0) & (w_in < W)
                    
                    if in_bounds:
                        # Calculate offset
                        offset = (
                            pid_b * input_stride_b + 
                            pid_c * input_stride_c + 
                            d_in * input_stride_d + 
                            h_in * input_stride_h + 
                            w_in * input_stride_w
                        )
                        val = tl.load(input_ptr + offset)
                    else:
                        val = float('-inf')
                    
                    # Update max value
                    if val > max_val:
                        max_val = val
    
    # Calculate output offset
    out_offset = (
        pid_b * output_stride_b + 
        pid_c * output_stride_c + 
        d_out * output_stride_d + 
        h_out * output_stride_h + 
        w_out * output_stride_w
    )
    
    # Store result
    tl.store(output_ptr + out_offset, max_val)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, 
                 dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Get input dimensions
        B, C, D, H, W = x.shape
        
        # Convert parameters to tuples
        kernel_size = (self.kernel_size, self.kernel_size, self.kernel_size)
        stride = (self.stride, self.stride, self.stride)
        padding = (self.padding, self.padding, self.padding)
        dilation = (self.dilation, self.dilation, self.dilation)
        
        # Calculate output dimensions
        def output_dim(dim, k, p, d, s):
            numerator = dim + 2 * p - d * (k - 1) - 1
            if self.ceil_mode:
                return int(math.ceil(numerator / s)) + 1
            return (numerator // s) + 1
            
        D_out = output_dim(D, kernel_size[0], padding[0], dilation[0], stride[0])
        H_out = output_dim(H, kernel_size[1], padding[1], dilation[1], stride[1])
        W_out = output_dim(W, kernel_size[2], padding[2], dilation[2], stride[2])
        
        # Create output tensor
        output = torch.empty((B, C, D_out, H_out, W_out), 
                             device=x.device, dtype=x.dtype)
        
        # Grid configuration
        grid = (B, C, D_out * H_out * W_out)
        
        # Launch kernel
        max_pool3d_kernel[grid](
            x, output,
            B, C, D, H, W,
            D_out, H_out, W_out,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
            kernel_size[0], kernel_size[1], kernel_size[2],
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            dilation[0], dilation[1], dilation[2],
            BLOCK_SIZE=1
        )
        
        return output

batch_size = 16
channels = 32
dim1 = 64
dim2 = 64
dim3 = 64
kernel_size = 3
stride = 2
padding = 1
dilation = 3

def get_inputs():
    x = torch.randn(batch_size, channels, dim1, dim2, dim3)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]
# =================== EVOLVE-BLOCK-END ===================