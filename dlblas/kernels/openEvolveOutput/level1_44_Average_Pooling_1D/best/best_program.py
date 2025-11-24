# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool1d_kernel(
    x_ptr,
    output_ptr,
    input_length,
    kernel_size,
    stride,
    padding,
    in_channels,
    output_length,
    batch_size,
):
    pid = tl.program_id(0)
    total_operations = batch_size * in_channels * output_length
    if pid >= total_operations:
        return

    batch_stride = in_channels * output_length
    channel_stride = output_length
    
    i_batch = pid // batch_stride
    pid_res = pid % batch_stride
    i_channel = pid_res // channel_stride
    i_out = pid_res % channel_stride

    start_idx = i_out * stride - padding
    base_ptr = i_batch * (in_channels * input_length) + i_channel * input_length

    total = 0.0
    for i in range(0, kernel_size):
        offset = start_idx + i
        if offset >= 0 and offset < input_length:
            val = tl.load(x_ptr + base_ptr + offset)
        else:
            val = 0.0
        total += val

    average = total / kernel_size
    output_index = i_batch * (in_channels * output_length) + i_channel * output_length + i_out
    tl.store(output_ptr + output_index, average)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, input_length = x.shape
        output_length = (input_length + 2*self.padding - self.kernel_size) // self.stride + 1
        
        if output_length <= 0:
            return torch.empty((batch_size, in_channels, 0), device=x.device, dtype=x.dtype)
        
        x = x.contiguous()
        output = torch.empty((batch_size, in_channels, output_length), device=x.device, dtype=x.dtype)
        total_elements = batch_size * in_channels * output_length
        
        if total_elements == 0:
            return output
            
        grid = lambda meta: (total_elements,)
        avg_pool1d_kernel[grid](
            x, output, 
            input_length, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            in_channels, 
            output_length,
            batch_size
        )
        return output

batch_size = 16
in_channels = 32
input_length = 128
kernel_size = 4
stride = 2
padding = 1

def get_inputs():
    x = torch.randn(batch_size, in_channels, input_length)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]
# =================== EVOLVE-BLOCK-END ===================