# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_ops_kernel(
    input_ptr,
    output_ptr,
    bias_ptr,
    scale1,
    scale2,
    input_d, input_h, input_w,
    pooled_d, pooled_h, pooled_w,
    stride_in_b, stride_in_c, stride_in_d, stride_in_h, stride_in_w,
    stride_out_b, stride_out_c, stride_out_d, stride_out_h, stride_out_w,
    batch, channels,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate total output elements
    num_output_elements = batch * channels * pooled_d * pooled_h * pooled_w
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_output_elements
    
    # Compute dimensions per element
    ch_pool = channels * pooled_d * pooled_h * pooled_w
    b_idx = offsets // ch_pool
    remainder = offsets % ch_pool
    
    pool3d = pooled_d * pooled_h * pooled_w
    c_idx = remainder // pool3d
    remainder = remainder % pool3d
    
    pool2d = pooled_h * pooled_w
    d_idx = remainder // pool2d
    remainder = remainder % pool2d
    
    h_idx = remainder // pooled_w
    w_idx = remainder % pooled_w
    
    # Load scalar parameters
    scale1_val = tl.load(scale1)
    scale2_val = tl.load(scale2)
    
    # Initialize accumulators
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    cnt = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    
    # Process 2x2x2 pooling window
    for kd in range(2):
        d_in = d_idx * 2 + kd
        for kh in range(2):
            h_in = h_idx * 2 + kh
            for kw in range(2):
                w_in = w_idx * 2 + kw
                
                # Check input boundaries
                in_bounds = (d_in < input_d) & (h_in < input_h) & (w_in < input_w)
                valid_mask = in_bounds & mask
                
                # Compute input offset
                input_offset = b_idx * stride_in_b + c_idx * stride_in_c + \
                               d_in * stride_in_d + h_in * stride_in_h + w_in * stride_in_w
                
                # Load input value
                val = tl.load(input_ptr + input_offset, mask=valid_mask, other=0.0)
                
                # Scale and accumulate
                scaled_val = val * scale1_val
                acc = tl.where(valid_mask, acc + scaled_val, acc)
                cnt = tl.where(valid_mask, cnt + 1, cnt)
    
    # Compute average pooling
    avg_val = acc / tl.maximum(cnt, 1)
    
    # Load bias (per channel)
    bias_val = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    # Apply bias and scaling
    result = (avg_val + bias_val) * scale2_val
    
    # Compute output offset
    output_offset = b_idx * stride_out_b + c_idx * stride_out_c + \
                    d_idx * stride_out_d + h_idx * stride_out_h + w_idx
    
    # Store result
    tl.store(output_ptr + output_offset, result, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding
        )
        self.scale1 = nn.Parameter(torch.tensor(scale1, dtype=torch.float32))
        self.scale2 = nn.Parameter(torch.tensor(scale2, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))
        
    def forward(self, x):
        # Compute conv transpose output
        x = self.conv_transpose(x)
        
        # Get input dimensions
        B, C, D, H, W = x.shape
        
        # Calculate pooled dimensions
        pooled_D = (D - 2) // 2 + 1
        pooled_H = (H - 2) // 2 + 1
        pooled_W = (W - 2) // 2 + 1
        
        # Create output tensor
        output = torch.empty(
            (B, C, pooled_D, pooled_H, pooled_W),
            device=x.device, dtype=x.dtype
        )
        
        # Total output elements
        total_output_elements = B * C * pooled_D * pooled_H * pooled_W
        
        # Launch kernel configuration
        BLOCK_SIZE = 128
        grid = (triton.cdiv(total_output_elements, BLOCK_SIZE),)
        
        # Launch fused kernel
        fused_ops_kernel[grid](
            x, output, self.bias,
            self.scale1, self.scale2,
            D, H, W,           # Input dimensions
            pooled_D, pooled_H, pooled_W,  # Pooled dimensions
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
            B, C,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output

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