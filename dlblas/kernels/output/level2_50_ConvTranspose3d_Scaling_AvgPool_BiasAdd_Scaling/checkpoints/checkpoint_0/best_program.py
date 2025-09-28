# ================== EVOLVE-BLOCK-START ==================
import torch
import triton
import triton.language as tl
import math

class ModelNew(torch.nn.Module):
    """
    Optimized model using Triton for fused 3D transposed convolution, scaling, 
    average pooling, bias addition, and scaling operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale1 = torch.nn.Parameter(torch.tensor(scale1))
        self.scale2 = torch.nn.Parameter(torch.tensor(scale2))
        self.bias = torch.nn.Parameter(torch.randn(bias_shape))
        
        # Weight initialization
        self.weight = torch.nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        B, C, D, H, W = x.shape
        # Calculate output dimensions after transposed convolution
        D_out = (D - 1) * self.stride - 2 * self.padding + self.kernel_size
        H_out = (H - 1) * self.stride - 2 * self.padding + self.kernel_size
        W_out = (W - 1) * self.stride - 2 * self.padding + self.kernel_size
        
        # Output after pooling
        pool_out = torch.empty(
            B, self.out_channels, 
            D_out // 2, H_out // 2, W_out // 2,
            device=x.device, dtype=x.dtype
        )
        
        # Grid configuration
        grid = lambda opt: (triton.cdiv(pool_out.numel(), opt['BLOCK_SIZE']),)
        
        # Launch fused kernel
        self.triton_fused_kernel[grid](
            x, self.weight, self.bias,
            pool_out,
            self.scale1, self.scale2,
            B, C, D, H, W,
            D_out, H_out, W_out,
            self.stride, self.padding,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            pool_out.stride(0), pool_out.stride(1), pool_out.stride(2), pool_out.stride(3), pool_out.stride(4),
            BLOCK_SIZE=1024,
        )
        return pool_out

    @triton.jit
    def triton_fused_kernel(
        x_ptr, weight_ptr, bias_ptr, out_ptr,
        scale1, scale2,
        B, C, D, H, W,
        D_out, H_out, W_out,
        stride, padding,
        x_batch_stride, x_channel_stride, x_d_stride, x_h_stride, x_w_stride,
        out_batch_stride, out_channel_stride, out_d_stride, out_h_stride, out_w_stride,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_output_elements = B * (D_out//2) * (H_out//2) * (W_out//2) * (C)
        if pid * BLOCK_SIZE >= num_output_elements:
            return
        
        # Decompose output index
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        b_idx = offsets // ((D_out//2) * (H_out//2) * (W_out//2) * C)
        remainder = offsets % ((D_out//2) * (H_out//2) * (W_out//2) * C)
        c_idx = remainder // ((D_out//2) * (H_out//2) * (W_out//2))
        remainder = remainder % ((D_out//2) * (H_out//2) * (W_out//2))
        d_out_idx = 2 * (remainder // ((H_out//2) * (W_out//2)))
        remainder = remainder % ((H_out//2) * (W_out//2))
        h_out_idx = 2 * (remainder // (W_out//2))
        w_out_idx = 2 * (remainder % (W_out//2))
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        scale1_val = tl.load(scale1)
        scale2_val = tl.load(scale2)
        
        # Loop over kernel and input channels
        for kd in range(3):
            for kh in range(3):
                for kw in range(3):
                    # Calculate input indices
                    d_in = (d_out_idx + kd - padding) // stride
                    h_in = (h_out_idx + kh - padding) // stride
                    w_in = (w_out_idx + kw - padding) // stride
                    
                    # Check if indices are integer and within bounds
                    valid_d = (d_out_idx + kd - padding) % stride == 0
                    valid_h = (h_out_idx + kh - padding) % stride == 0
                    valid_w = (w_out_idx + kw - padding) % stride == 0
                    
                    in_bounds = (
                        (d_in >= 0) & (d_in < D) & 
                        (h_in >= 0) & (h_in < H) & 
                        (w_in >= 0) & (w_in < W)
                    )
                    valid_mask = valid_d & valid_h & valid_w & in_bounds
                    
                    # Load input and weight
                    if tl.sum(valid_mask) > 0:
                        x_offsets = (
                            b_idx * x_batch_stride + 
                            d_in * x_d_stride + 
                            h_in * x_h_stride + 
                            w_in * x_w_stride
                        )
                        x_val = tl.load(
                            x_ptr + x_offsets, 
                            mask=valid_mask, 
                            other=0.0
                        )
                        
                        w_offsets = (
                            c_idx * C * 27 +  # 3x3x3 kernel
                            kd * 9 + kh * 3 + kw
                        )
                        w_val = tl.load(
                            weight_ptr + w_offsets, 
                            mask=valid_mask, 
                            other=0.0
                        )
                        
                        acc += x_val * w_val
        
        # Apply first scaling
        acc = acc * scale1_val
        
        # Average pooling (sum then divide by 8)
        pool_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        count = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        for pd in range(2):
            for ph in range(2):
                for pw in range(2):
                    d = d_out_idx + pd
                    h = h_out_idx + ph
                    w = w_out_idx + pw
                    in_bounds = (d < D_out) & (h < H_out) & (w < W_out)
                    if tl.sum(in_bounds) > 0:
                        pool_acc += tl.where(in_bounds, acc, 0.0)
                        count += tl.where(in_bounds, 1, 0)
        pool_val = pool_acc / tl.maximum(count, 1)
        
        # Add bias and apply second scaling
        bias_val = tl.load(bias_ptr + c_idx)
        result = (pool_val + bias_val) * scale2_val
        
        # Store final result
        out_offsets = (
            b_idx * out_batch_stride + 
            c_idx * out_channel_stride + 
            (d_out_idx//2) * out_d_stride + 
            (h_out_idx//2) * out_h_stride + 
            (w_out_idx//2) * out_w_stride
        )
        tl.store(out_ptr + out_offsets, result)

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale1 = 0.5
scale2 = 1.0
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape]
# =================== EVOLVE-BLOCK-END ===================