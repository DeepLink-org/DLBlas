# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_flatten_mean_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # Batch index
    offsets = pid * n_elements + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < n_elements

    # Load and accumulate elements
    chunk = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    chunk_sum = tl.sum(chunk, axis=0)
    chunk_mean = chunk_sum / n_elements

    # Store result for this batch
    tl.store(output_ptr + pid, chunk_mean)

class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, computes the mean
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        
        # Flatten spatial and channel dimensions
        x_flat = x.view(x.shape[0], -1)
        n_elements = x_flat.shape[1]
        output = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)

        # Launch Triton kernel for efficient mean reduction
        grid = (x.shape[0],)
        fused_flatten_mean_kernel[grid](
            x_flat, output, n_elements,
            BLOCK_SIZE=triton.next_power_of_2(n_elements)
        )
        return output

batch_size = 128
in_channels = 3
out_channels = 16
D, H, W = 16, 32, 32
kernel_size = 3
num_groups = 8

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups]
# =================== EVOLVE-BLOCK-END ===================