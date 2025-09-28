# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def max_pool_forward_kernel(
    input_ptr,
    output_ptr,
    input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
    output_batch_stride, output_channel_stride, output_height_stride, output_width_stride,
    height, width, out_height, out_width,
    kernel_size, stride, padding, dilation,
    BLOCK_SIZE: tl.constexpr
):
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    hw_idx = tl.program_id(2)
    
    # Create block of output indices
    hw_offsets = tl.arange(0, BLOCK_SIZE)
    hw_mask = hw_offsets < (out_height * out_width)
    hw_total = hw_offsets + hw_idx * BLOCK_SIZE
    
    # Compute output coordinates
    oh = hw_total // out_width
    ow = hw_total % out_width
    
    # Compute window start positions
    i0 = oh * stride - padding
    j0 = ow * stride - padding
    
    # Initialize max values to -inf
    max_vals = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
    
    # Loop over kernel window
    for di in range(kernel_size):
        for dj in range(kernel_size):
            # Compute input positions with dilation
            i = i0 + di * dilation
            j = j0 + dj * dilation
            
            # Check boundaries and create mask
            in_bounds = (i >= 0) & (i < height) & (j >= 0) & (j < width)
            active_mask = hw_mask & in_bounds
            
            # Calculate memory offsets
            input_offsets = (
                b_idx * input_batch_stride + 
                c_idx * input_channel_stride + 
                i * input_height_stride + 
                j * input_width_stride
            )
            
            # Load values
            vals = tl.load(input_ptr + input_offsets, mask=active_mask, other=float('-inf'))
            
            # Update max values
            max_vals = tl.maximum(max_vals, vals)
    
    # Calculate output offsets
    output_offsets = (
        b_idx * output_batch_stride + 
        c_idx * output_channel_stride + 
        oh * output_height_stride + 
        ow * output_width_stride
    )
    
    # Store results
    tl.store(output_ptr + output_offsets, max_vals, mask=hw_mask)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        
        # Calculate output dimensions
        out_height = ((height + 2 * self.padding - 
                      self.dilation * (self.kernel_size - 1) - 1) 
                     // self.stride) + 1
        out_width = ((width + 2 * self.padding - 
                     self.dilation * (self.kernel_size - 1) - 1) 
                    // self.stride) + 1
        
        # Create output tensor
        output = torch.empty(
            (batch_size, channels, out_height, out_width),
            device=x.device, dtype=x.dtype
        )
        
        if x.is_cuda:
            # Set grid and block dimensions
            BLOCK_SIZE = 128
            grid = (
                batch_size, 
                channels, 
                triton.cdiv(out_height * out_width, BLOCK_SIZE)
            )
            
            # Launch kernel
            max_pool_forward_kernel[grid](
                x, output,
                x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                output.stride(0), output.stride(1), output.stride(2), output.stride(3),
                height, width, out_height, out_width,
                self.kernel_size, self.stride, self.padding, self.dilation,
                BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            # CPU fallback
            output = nn.functional.max_pool2d(
                x, self.kernel_size, self.stride, 
                self.padding, self.dilation
            )
        
        return output

batch_size = 16
channels = 32
height = 128
width = 128
kernel_size = 2
stride = 2
padding = 1
dilation = 3

def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]

# =================== EVOLVE-BLOCK-END ===================