# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.bias = None

    @triton.jit
    def _depthwise_conv2d_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        in_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        height_out,
        width_out,
        has_bias: tl.constexpr,
        BLOCK_SIZE_H: tl.constexpr,
        BLOCK_SIZE_W: tl.constexpr,
    ):
        # 3D grid: [batch*channels, height_blocks, width_blocks]
        pid0 = tl.program_id(0)
        pid1 = tl.program_id(1)
        pid2 = tl.program_id(2)
        
        # Compute channel and batch indices
        batch = pid0 // in_channels
        channel = pid0 % in_channels
        
        # Output tile starting indices
        start_h = pid1 * BLOCK_SIZE_H
        start_w = pid2 * BLOCK_SIZE_W
        
        # Create indices for output tile
        h_idx = start_h + tl.arange(0, BLOCK_SIZE_H)
        w_idx = start_w + tl.arange(0, BLOCK_SIZE_W)
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
        
        # Precompute weight offset
        weight_offset = channel * kernel_size * kernel_size
        
        # Loop over kernel
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Compute input positions with vectorization
                h_in = h_idx * stride + kh - padding
                w_in = w_idx * stride + kw - padding
                
                # Boundary checks
                mask_h = (h_in >= 0) & (h_in < height)
                mask_w = (w_in >= 0) & (w_in < width)
                mask = mask_h[:, None] & mask_w[None, :]
                
                # Memory offsets with coalesced access
                base = batch * in_channels * height * width + channel * height * width
                offsets = base + h_in[:, None] * width + w_in[None, :]
                
                # Load input with boundary masking
                input_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                
                # Load weight
                weight_val = tl.load(weight_ptr + weight_offset + kh * kernel_size + kw)
                
                # Fused multiply-add
                acc += input_val * weight_val
        
        # Add bias if present
        if has_bias:
            bias_val = tl.load(bias_ptr + channel)
            acc += bias_val
        
        # Compute output offsets
        mask_store = (h_idx < height_out)[:, None] & (w_idx < width_out)[None, :]
        out_base = batch * in_channels * height_out * width_out + channel * height_out * width_out
        out_offsets = out_base + h_idx[:, None] * width_out + w_idx[None, :]
        
        # Store results
        tl.store(output_ptr + out_offsets, acc, mask=mask_store)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        height_out = (height + 2*self.padding - self.kernel_size) // self.stride + 1
        width_out = (width + 2*self.padding - self.kernel_size) // self.stride + 1
        
        output = torch.empty((batch_size, self.in_channels, height_out, width_out), 
                             device=x.device, dtype=x.dtype)
        
        # Prepare kernel parameters
        has_bias = self.bias is not None
        bias_ptr = self.bias.data_ptr() if has_bias else 0
        
        # Configure grid with optimal block sizes
        grid = (
            batch_size * self.in_channels,
            triton.cdiv(height_out, 16),
            triton.cdiv(width_out, 16)
        )
        
        # Launch kernel
        self._depthwise_conv2d_kernel[grid](
            x, self.weight, bias_ptr, output,
            self.in_channels, height, width, self.kernel_size,
            self.stride, self.padding, height_out, width_out,
            has_bias,
            BLOCK_SIZE_H=16, BLOCK_SIZE_W=16
        )
        
        return output

# Test code
batch_size = 16
in_channels = 3
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]
# =================== EVOLVE-BLOCK-END ===================