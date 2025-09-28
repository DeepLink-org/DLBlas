# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def instance_norm_kernel(
    input_ptr, 
    output_ptr, 
    n_elements,
    eps, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * n_elements
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < n_elements

    # Load data with boundary checks
    row = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean and variance using stable online algorithm
    row_sum = tl.sum(row, axis=0)
    mean = row_sum / n_elements
    centered = row - mean
    centered_sq = centered * centered
    variance = tl.sum(centered_sq, axis=0) / n_elements
    std = tl.sqrt(variance + eps)
    
    # Normalize and store
    normalized = centered / std
    tl.store(output_ptr + offsets, normalized, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = divide_by
        self.eps = 1e-5  # Match PyTorch default

    def forward(self, x):
        x = self.conv(x)
        
        # Apply Triton instance normalization
        batch, channels, height, width = x.shape
        n_elements = height * width
        
        if n_elements <= 1024:  # Within Triton block limits
            x_flat = x.reshape(batch * channels, n_elements).contiguous()
            output = torch.empty_like(x_flat)
            block_size = triton.next_power_of_2(n_elements)
            grid = (batch * channels,)
            instance_norm_kernel[grid](
                x_flat, output, n_elements, self.eps, 
                BLOCK_SIZE=block_size
            )
            x = output.reshape(batch, channels, height, width)
        else:
            # Fallback for large tensors
            x = torch.nn.functional.instance_norm(
                x, None, None, eps=self.eps
            )
        
        x = x / self.divide_by
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
divide_by = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]
# =================== EVOLVE-BLOCK-END ===================