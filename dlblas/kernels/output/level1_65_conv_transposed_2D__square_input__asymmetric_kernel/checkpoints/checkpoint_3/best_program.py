# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_transpose2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_channels, out_channels, 
    input_height, input_width,
    kernel_height, kernel_width,
    stride_h, stride_w,
    padding_h, padding_w,
    output_height, output_width,
    groups,
    total_ow_blocks,
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_OH: tl.constexpr,
    BLOCK_SIZE_OW: tl.constexpr,
):
    # Compute program IDs (3 dimensions max)
    pid_batch = tl.program_id(0)
    pid_oc_block = tl.program_id(1)
    pid_spatial = tl.program_id(2)
    
    # Decompose spatial index into height and width blocks
    pid_oh = pid_spatial // total_ow_blocks
    pid_ow_block = pid_spatial % total_ow_blocks
    
    # Compute output channel range for this block
    oc_offsets = pid_oc_block * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC)
    oc_mask = oc_offsets < out_channels
    
    # Compute output width range for this block
    ow_offsets = pid_ow_block * BLOCK_SIZE_OW + tl.arange(0, BLOCK_SIZE_OW)
    ow_mask = ow_offsets < output_width
    
    # Group processing
    group_size = out_channels // groups
    group_idx = oc_offsets // group_size
    
    # Calculate input channel range per group
    in_channels_per_group = in_channels // groups
    ic_start = group_idx * in_channels_per_group
    
    # Initialize output block accumulator
    output_block = tl.zeros((BLOCK_SIZE_OC, BLOCK_SIZE_OW), dtype=tl.float32)
    
    # Loop over kernel height and width
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Compute corresponding input position
            ih = (pid_oh * stride_h) + kh - padding_h
            iw = (ow_offsets * stride_w) + kw - padding_w
            
            # Check input bounds
            ih_mask = (ih >= 0) & (ih < input_height)
            iw_mask = (iw >= 0) & (iw < input_width) & ow_mask
            valid_mask = ih_mask & iw_mask
            
            # Prepare input pointers
            input_offsets = (pid_batch * in_channels * input_height * input_width +
                             ic_start * input_height * input_width +
                             ih * input_width + iw)
            
            # Load input block
            input_block = tl.load(
                input_ptr + input_offsets,
                mask=valid_mask & oc_mask[:, None],
                other=0.0
            )
            
            # Prepare weight pointers
            weight_offsets = (ic_start * out_channels * kernel_height * kernel_width +
                              oc_offsets * kernel_height * kernel_width +
                              kh * kernel_width + kw)
            
            # Load weight block
            weight_block = tl.load(
                weight_ptr + weight_offsets,
                mask=oc_mask[:, None],
                other=0.0
            )
            
            # Accumulate
            output_block += tl.sum(input_block * weight_block, axis=0)
    
    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + oc_offsets, mask=oc_mask, other=0.0)
        output_block += bias[:, None]
    
    # Compute output offsets
    output_offsets = (pid_batch * out_channels * output_height * output_width +
                      oc_offsets * output_height * output_width +
                      pid_oh * output_width + ow_offsets)
    
    # Store results
    tl.store(
        output_ptr + output_offsets,
        output_block,
        mask=oc_mask[:, None] & ow_mask[None, :]
    )


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weight tensor
        self.weight = nn.Parameter(torch.empty(
            in_channels,
            out_channels // groups,
            kernel_size[0],
            kernel_size[1]
        ))
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, in_height, in_width = x.shape
        
        # Calculate output dimensions
        out_height = (in_height - 1) * self.stride - 2 * self.padding + self.kernel_size[0] + self.output_padding
        out_width = (in_width - 1) * self.stride - 2 * self.padding + self.kernel_size[1] + self.output_padding
        
        # Create output tensor
        output = torch.empty(
            batch_size,
            self.out_channels,
            out_height,
            out_width,
            device=x.device,
            dtype=x.dtype
        )
        
        # Configure kernel launch parameters
        BLOCK_SIZE_OC = 32
        BLOCK_SIZE_OW = 64
        
        # Compute total width blocks and spatial dimension
        total_ow_blocks = triton.cdiv(out_width, BLOCK_SIZE_OW)
        total_spatial_blocks = out_height * total_ow_blocks
        
        grid = (
            batch_size,  # pid_batch
            triton.cdiv(self.out_channels, BLOCK_SIZE_OC),  # pid_oc_block
            total_spatial_blocks,  # pid_spatial (combined height and width blocks)
        )
        
        # Launch kernel
        conv_transpose2d_kernel[grid](
            x, self.weight, self.bias, output,
            self.in_channels, self.out_channels,
            in_height, in_width,
            self.kernel_size[0], self.kernel_size[1],
            self.stride, self.stride,
            self.padding, self.padding,
            out_height, out_width,
            self.groups,
            total_ow_blocks,  # Pass total ow blocks for spatial decomposition
            BLOCK_SIZE_OC=BLOCK_SIZE_OC,
            BLOCK_SIZE_OH=1,
            BLOCK_SIZE_OW=BLOCK_SIZE_OW,
        )
        
        return output

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)  # Asymmetric kernel
width = 128
height = 128

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
# =================== EVOLVE-BLOCK-END ===================