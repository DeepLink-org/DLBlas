# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import triton
import triton.language as tl

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.out_channels = out_channels
        
        # Initialize weights and bias
        self.weight = torch.nn.Parameter(
            torch.empty(out_channels, in_channels, *kernel_size))
        self.bias = torch.nn.Parameter(
            torch.empty(out_channels), requires_grad=bias) if bias else None
        
        # Reset parameters
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute output dimensions
        _, _, H, W = x.shape
        K_H, K_W = self.kernel_size
        H_out = (H + 2*self.padding[0] - self.dilation[0]*(K_H-1) - 1) // self.stride[0] + 1
        W_out = (W + 2*self.padding[1] - self.dilation[1]*(K_W-1) - 1) // self.stride[1] + 1
        
        # Allocate output tensor
        output = torch.empty((x.size(0), self.out_channels, H_out, W_out), 
                             device=x.device, dtype=x.dtype)
        
        # Launch optimized convolution kernel
        grid = (triton.cdiv(H_out * W_out, 16), x.size(0), self.out_channels)
        conv2d_kernel[grid](
            x, self.weight, output,
            self.bias if self.bias is not None else None,
            K_H, K_W, self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.weight.stride(0), self.weight.stride(1), 
            self.weight.stride(2), self.weight.stride(3),
            output.stride(0), output.stride(1), 
            output.stride(2), output.stride(3),
            H, W, H_out, W_out,
            BLOCK_C=16,
            USE_BIAS=self.bias is not None,
        )
        return output

@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr, bias_ptr,
    K_H, K_W, stride_h, stride_w,
    pad_h, pad_w, dilation_h, dilation_w,
    stride_bx, stride_cx, stride_hx, stride_wx,
    stride_bw, stride_cw, stride_hw, stride_ww,
    stride_bo, stride_co, stride_ho, stride_wo,
    H, W, H_out, W_out,
    BLOCK_C: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    # Parallelize over output positions and channels
    pid_hw = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    # Create block pointers for vectorized access
    block_start = pid_hw * 16
    offsets = block_start + tl.arange(0, 16)
    mask = offsets < (H_out * W_out)
    h_out = offsets // W_out
    w_out = offsets % W_out
    
    # Compute input window start positions
    h_in = tl.maximum(h_out * stride_h - pad_h, 0)
    w_in = tl.maximum(w_out * stride_w - pad_w, 0)
    
    # Initialize accumulator
    acc = tl.zeros((16,), dtype=tl.float32)
    
    # Loop over kernel positions (unrolled for 3x5 kernel)
    for kh in range(K_H):
        for kw in range(K_W):
            # Precompute input positions
            h_idx = h_in + kh * dilation_h
            w_idx = w_in + kw * dilation_w
            
            # Boundary check
            within_bounds = (h_idx < H) & (w_idx < W)
            valid_mask = mask & within_bounds
            
            # Precompute input pointers
            x_ptrs = (
                x_ptr + pid_b * stride_bx +
                h_idx * stride_hx + 
                w_idx * stride_wx
            )
            
            # Precompute weight pointers
            w_ptrs = (
                w_ptr + pid_c * stride_bw +
                kh * stride_hw + 
                kw * stride_ww
            )
            
            # Vectorized load and compute
            for c in range(0, tl.cdiv(w_ptr.shape[1], BLOCK_C)):
                c_mask = c * BLOCK_C + tl.arange(0, BLOCK_C) < w_ptr.shape[1]
                w = tl.load(w_ptrs + c * BLOCK_C * stride_cw, mask=c_mask)
                x_val = tl.load(x_ptrs + c * BLOCK_C * stride_cx, mask=valid_mask & c_mask)
                w = tl.broadcast_to(w, (16, BLOCK_C))
                acc += tl.sum(x_val * w, axis=1)
    
    # Add bias if enabled
    if USE_BIAS:
        bias = tl.load(bias_ptr + pid_c)
        acc += bias
    
    # Store results
    out_ptrs = (
        out_ptr + pid_b * stride_bo +
        pid_c * stride_co +
        h_out * stride_ho + 
        w_out * stride_wo
    )
    tl.store(out_ptrs, acc, mask=mask)

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5)  # Asymmetric kernel
width = 256
height = 256

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
# =================== EVOLVE-BLOCK-END ===================