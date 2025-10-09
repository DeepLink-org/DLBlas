# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_reduction_activation(
    input_ptr,
    output_ptr,
    bias_value,
    scaling_value,
    C,
    spatial_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = pid // spatial_size
    spatial_id = pid % spatial_size
    
    base_offset = (batch_id * spatial_size + spatial_id) * C
    offsets = base_offset + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < C
    
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(values, axis=0)
    mean_val = total / C
    activated = tl.tanh(mean_val + bias_value)
    scaled = activated * scaling_value
    
    output_offset = batch_id * spatial_size + spatial_id
    tl.store(output_ptr + output_offset, scaled)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        B, C, D, H, W = x.shape
        spatial_size = D * H * W
        
        # Reorder dimensions for efficient channel access
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        output = torch.empty((B, 1, D, H, W), device=x.device, dtype=x.dtype)
        
        # Extract scalar values for kernel
        bias_val = self.bias.item()
        scale_val = self.scaling_factor
        
        # Launch Triton kernel
        grid = (B * spatial_size,)
        fused_reduction_activation[grid](
            x, 
            output.view(B, spatial_size),
            bias_val,
            scale_val,
            C,
            spatial_size,
            BLOCK_SIZE=triton.next_power_of_2(C)
        )
        return output

batch_size = 16
in_channels = 8
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (1, 1, 1, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape, scaling_factor]
# =================== EVOLVE-BLOCK-END ===================