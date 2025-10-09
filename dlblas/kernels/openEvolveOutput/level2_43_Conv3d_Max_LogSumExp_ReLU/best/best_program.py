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
    total_elements,
    BLOCK_SIZE: tl.constexpr,
    SPATIAL_PER_THREAD: tl.constexpr,
):
    pid = tl.program_id(0)
    base_idx = pid * SPATIAL_PER_THREAD
    for i in range(SPATIAL_PER_THREAD):
        idx = base_idx + i
        if idx < total_elements:
            batch_idx = idx // spatial_size
            spatial_idx = idx % spatial_size
            base_ptr = x_ptr + (batch_idx * spatial_size * out_channels) + (spatial_idx * out_channels)
            offsets = tl.arange(0, BLOCK_SIZE)
            mask = offsets < out_channels
            vec = tl.load(base_ptr + offsets, mask=mask, other=-float('inf'))
            
            max_val = tl.max(vec, axis=0)
            safe_vec = vec - max_val
            exp_vec = tl.exp(safe_vec)
            sum_exp = tl.sum(exp_vec, axis=0)
            log_sum_exp = tl.log(sum_exp) + max_val
            out_val = tl.where(log_sum_exp > 0, log_sum_exp, 0.0)
            tl.store(output_ptr + idx, out_val)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        batch_size, _, d_new, h_new, w_new = x.shape
        spatial_size = d_new * h_new * w_new
        total_elements = batch_size * spatial_size
        
        # Reshape to (batch, spatial, channels) and make contiguous
        x_reshaped = x.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, spatial_size, self.out_channels)
        output = torch.empty(total_elements, device=x.device, dtype=x.dtype)
        
        # Optimized kernel launch parameters
        BLOCK_SIZE = triton.next_power_of_2(self.out_channels)
        SPATIAL_PER_THREAD = 4
        grid = (triton.cdiv(total_elements, SPATIAL_PER_THREAD),)
        
        _logsumexp_relu_kernel[grid](
            x_reshaped, output,
            self.out_channels,
            spatial_size,
            total_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            SPATIAL_PER_THREAD=SPATIAL_PER_THREAD
        )
        
        # Reshape output to match original dimensions
        return output.view(batch_size, 1, d_new, h_new, w_new)

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