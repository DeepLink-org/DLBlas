# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _logsumexp_relu_kernel(
    x_ptr,
    output_ptr,
    out_channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx = pid // spatial_size
    spatial_idx = pid % spatial_size

    # Calculate base pointer position
    base_ptr = x_ptr + batch_idx * out_channels * spatial_size + spatial_idx
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_channels

    # Load channel vector with stride = spatial_size
    ptrs = base_ptr + offsets * spatial_size
    vec = tl.load(ptrs, mask=mask, other=-float('inf'))

    # Compute logsumexp with improved numerical stability
    max_val = tl.max(vec, axis=0)
    safe_vec = vec - max_val
    exp_vec = tl.exp(safe_vec)
    sum_exp = tl.sum(exp_vec, axis=0)
    log_sum_exp = tl.log(sum_exp) + max_val

    # Apply ReLU and store result
    out_val = tl.where(log_sum_exp > 0, log_sum_exp, 0.0)
    tl.store(output_ptr + pid, out_val)

class ModelNew(nn.Module):
    """
    Optimized model using Triton for logsumexp+ReLU fusion
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        
        # Prepare tensors for Triton kernel
        batch_size, out_channels, D, H, W = x.shape
        spatial_size = D * H * W
        x_flat = x.reshape(batch_size, out_channels, spatial_size)
        output_flat = torch.empty(batch_size * spatial_size, device=x.device, dtype=torch.float32)
        
        # Launch Triton kernel
        grid = (batch_size * spatial_size,)
        BLOCK_SIZE = triton.next_power_of_2(out_channels)
        _logsumexp_relu_kernel[grid](
            x_flat, output_flat, 
            out_channels, spatial_size,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Reshape output to original dimensions
        x = output_flat.view(batch_size, 1, D, H, W)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 1
padding = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
# =================== EVOLVE-BLOCK-END ===================