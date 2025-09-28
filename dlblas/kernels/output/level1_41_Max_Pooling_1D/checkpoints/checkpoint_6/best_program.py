# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def max_pool_1d_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    input_batch_stride,
    input_channel_stride,
    input_length_stride,
    output_batch_stride,
    output_channel_stride,
    output_length_stride,
    indices_batch_stride,
    indices_channel_stride,
    indices_length_stride,
    kernel_size,
    stride,
    padding,
    dilation,
    input_length,
    output_length,
    RETURN_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid // BLOCK_SIZE
    channel_idx = pid % BLOCK_SIZE
    
    if batch_idx >= output_batch_stride or channel_idx >= output_channel_stride:
        return
        
    input_base = batch_idx * input_batch_stride + channel_idx * input_channel_stride
    output_base = batch_idx * output_batch_stride + channel_idx * output_channel_stride
    indices_base = batch_idx * indices_batch_stride + channel_idx * indices_channel_stride
    
    for out_idx in tl.range(0, output_length):
        start_idx = out_idx * stride - padding
        max_val = -tl.math.inf
        max_index = -1
        
        for k in range(0, kernel_size):
            pos = start_idx + k * dilation
            if pos >= 0 and pos < input_length:
                offset = input_base + pos * input_length_stride
                val = tl.load(input_ptr + offset)
            else:
                val = -tl.math.inf
                
            if val > max_val:
                max_val = val
                max_index = pos
                
        tl.store(output_ptr + output_base + out_idx * output_length_stride, max_val)
        if RETURN_INDICES:
            tl.store(indices_ptr + indices_base + out_idx * indices_length_stride, max_index)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError("Input must be 3D: (batch, channels, length)")
            
        batch_size, num_channels, input_length = x.shape
        L_out = math.floor(
            (input_length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1
        )
        output = torch.empty((batch_size, num_channels, L_out), device=x.device, dtype=x.dtype)
        
        if self.return_indices:
            indices = torch.empty((batch_size, num_channels, L_out), device=x.device, dtype=torch.int64)
        else:
            indices = torch.empty(0, device=x.device, dtype=torch.int64)
            
        grid = (batch_size * num_channels,)
        max_pool_1d_kernel[grid](
            x,
            output,
            indices,
            x.stride(0), x.stride(1), x.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            input_length,
            L_out,
            self.return_indices,
            num_channels,
        )
        
        if self.return_indices:
            return output, indices
        return output

batch_size = 16
features = 64
sequence_length = 128
kernel_size = 4
stride = 2
padding = 2
dilation = 3
return_indices = False

def get_inputs():
    x = torch.randn(batch_size, features, sequence_length)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]
# =================== EVOLVE-BLOCK-END ===================