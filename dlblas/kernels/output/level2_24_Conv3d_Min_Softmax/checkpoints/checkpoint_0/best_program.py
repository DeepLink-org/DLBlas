# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def min_reduction_kernel(
    input_ptr,
    output_ptr,
    reduction_dim_size,
    other_dims_product,
    stride_reduction,
    stride_other,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_other = pid * stride_other + tl.arange(0, BLOCK_SIZE)
    mask_other = tl.arange(0, BLOCK_SIZE) < BLOCK_SIZE
    
    min_val = tl.full((BLOCK_SIZE,), float('inf'), dtype=tl.float32)
    for k in range(0, reduction_dim_size):
        offsets = offsets_other + k * stride_reduction
        current = tl.load(input_ptr + offsets, mask=mask_other, other=float('inf'))
        min_val = tl.minimum(min_val, current)
    
    tl.store(output_ptr + offsets_other, min_val, mask=mask_other)

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    channel_size,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    spatial_idx = pid
    offsets_base = spatial_idx * channel_size + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < channel_size
    
    # Load block of channels for this spatial location
    row = tl.load(input_ptr + offsets_base, mask=mask, other=-float('inf'))
    
    # Compute softmax
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator
    
    tl.store(output_ptr + offsets_base, softmax_out, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        orig_shape = x.shape
        
        # Convert dim to positive index
        if self.dim < 0:
            dim = self.dim + x.dim()
        else:
            dim = self.dim
        
        # Compute min reduction using Triton
        reduction_dim_size = x.size(dim)
        other_dims = [size for i, size in enumerate(x.shape) if i != dim]
        other_dims_product = torch.prod(torch.tensor(other_dims)).item()
        
        # Create output tensor for min reduction
        min_output = torch.empty(other_dims, device=x.device, dtype=x.dtype)
        
        # Compute strides
        stride_reduction = x.stride(dim)
        stride_other = x.stride()[0] if dim != 0 else x.stride()[1]
        
        # Launch min reduction kernel
        grid = (triton.cdiv(other_dims_product, 128),)
        min_reduction_kernel[grid](
            x,
            min_output,
            reduction_dim_size,
            other_dims_product,
            stride_reduction,
            stride_other,
            BLOCK_SIZE=128,
        )
        
        # Apply softmax along channel dimension (dim=1)
        spatial_dims = min_output.shape[2:]
        spatial_size = torch.prod(torch.tensor(spatial_dims)).item()
        channel_size = self.out_channels
        
        # Reshape for softmax kernel
        softmax_input = min_output.view(-1, channel_size, spatial_size)
        softmax_output = torch.empty_like(softmax_input)
        
        # Launch softmax kernel
        grid_softmax = (spatial_size,)
        softmax_kernel[grid_softmax](
            softmax_input,
            softmax_output,
            channel_size,
            spatial_size,
            BLOCK_SIZE=triton.next_power_of_2(channel_size),
        )
        
        return softmax_output.view(min_output.shape)

batch_size = 128
in_channels = 3
out_channels = 16
D, H, W = 16, 32, 32
kernel_size = 3
dim = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]
# =================== EVOLVE-BLOCK-END ===================