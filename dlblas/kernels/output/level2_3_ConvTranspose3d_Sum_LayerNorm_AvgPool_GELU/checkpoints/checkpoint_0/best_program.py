# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def layer_norm_triton(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    N,  # Number of channels (normalization dimension)
    stride_xn, stride_yn,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load input data
    x_ptrs = x_ptr + row_idx * stride_xn + offsets
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / N
    # Compute variance
    x_zm = x - mean
    x_zm_sq = x_zm * x_zm
    variance = tl.sum(x_zm_sq, axis=0) / N

    # Normalize: (x - mean) / sqrt(var + eps)
    inv_std = 1.0 / tl.sqrt(variance + eps)
    x_norm = x_zm * inv_std

    # Apply affine transformation
    w = tl.load(weight_ptr + offsets, mask=mask)
    b = tl.load(bias_ptr + offsets, mask=mask)
    y = x_norm * w + b

    # Store result
    y_ptrs = y_ptr + row_idx * stride_yn + offsets
    tl.store(y_ptrs, y, mask=mask)

class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum, 
    layer normalization (optimized with Triton), average pooling, and GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x + self.sum_weight
        
        # Prepare for Triton layer norm
        original_shape = x.shape
        n_channels = original_shape[1]
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous()  # [N, C, D, H, W] -> [N, D, H, W, C]
        M = x_flat.shape[0] * x_flat.shape[1] * x_flat.shape[2] * x_flat.shape[3]  # Total spatial elements
        N = n_channels
        
        # Create output tensor
        y_flat = torch.empty_like(x_flat)
        
        # Launch Triton kernel
        grid = (M,)
        BLOCK_SIZE = triton.next_power_of_2(N)
        layer_norm_triton[grid](
            x_flat, y_flat, 
            self.norm.weight, self.norm.bias,
            N,
            x_flat.stride(0), y_flat.stride(0),
            eps=self.norm.eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Reshape back to original
        x = y_flat.permute(0, 4, 1, 2, 3).contiguous().view(original_shape)
        
        x = self.avg_pool(x)
        x = self.gelu(x)
        return x

batch_size = 128
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = (1, 1, 1)
output_padding = (1, 1, 1)
sum_weight = 1.0
norm_shape = (out_channels,)
pool_kernel_size = (2, 2, 2)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size]
# =================== EVOLVE-BLOCK-END ===================