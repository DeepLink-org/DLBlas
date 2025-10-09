# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

# Triton kernel for 1x1 convolution
@triton.jit
def conv1x1_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    in_channels, out_channels, stride_x_b, stride_x_h, stride_x_w, stride_x_c,
    stride_out_b, stride_out_h, stride_out_w, stride_out_c,
    stride_weight_oc, stride_weight_ic,
    height, width, batch_size,
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_IC: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pixels = batch_size * height * width
    if pid >= num_pixels:
        return

    # Calculate spatial indices
    b_idx = pid // (height * width)
    hw = pid % (height * width)
    i_idx = hw // width
    j_idx = hw % width

    # Compute base pointers for current pixel
    base_x = x_ptr + b_idx * stride_x_b + i_idx * stride_x_h + j_idx * stride_x_w
    base_output = output_ptr + b_idx * stride_out_b + i_idx * stride_out_h + j_idx * stride_out_w

    # Process output channels in blocks
    for oc_block_start in range(0, out_channels, BLOCK_SIZE_OC):
        oc_offsets = oc_block_start + tl.arange(0, BLOCK_SIZE_OC)
        oc_mask = oc_offsets < out_channels
        acc = tl.zeros((BLOCK_SIZE_OC,), dtype=tl.float32)

        # Process input channels in blocks
        for ic_block_start in range(0, in_channels, BLOCK_SIZE_IC):
            ic_offsets = ic_block_start + tl.arange(0, BLOCK_SIZE_IC)
            ic_mask = ic_offsets < in_channels

            # Load input block
            x_vals = tl.load(base_x + ic_offsets * stride_x_c, mask=ic_mask, other=0.0)
            
            # Load weight block
            weight_ptrs = weight_ptr + oc_offsets[:, None] * stride_weight_oc + ic_offsets[None, :] * stride_weight_ic
            w_block = tl.load(weight_ptrs, mask=oc_mask[:, None] & ic_mask[None, :], other=0.0)
            
            # Compute partial dot product
            acc += tl.sum(x_vals[None, :] * w_block, axis=1)

        # Load bias and store results
        bias_vals = tl.load(bias_ptr + oc_offsets, mask=oc_mask, other=0.0)
        output_vals = acc + bias_vals
        tl.store(base_output + oc_offsets * stride_out_c, output_vals, mask=oc_mask)

def conv1x1_triton(x, weight, bias):
    batch, in_channels, height, width = x.shape
    out_channels = weight.shape[0]
    output = torch.empty((batch, out_channels, height, width), device=x.device, dtype=x.dtype)
    
    # Ensure contiguous tensors
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Launch kernel
    num_pixels = batch * height * width
    grid = (num_pixels,)
    conv1x1_kernel[grid](
        x, weight, bias, output,
        in_channels, out_channels,
        x.stride(0), x.stride(2), x.stride(3), x.stride(1),
        output.stride(0), output.stride(2), output.stride(3), output.stride(1),
        weight.stride(0), weight.stride(1),
        height, width, batch,
        BLOCK_SIZE_OC=128,
        BLOCK_SIZE_IC=32
    )
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(ModelNew, self).__init__()
        
        # 1x1 convolution branch - replaced with Triton implementation
        self.weight_1x1 = nn.Parameter(torch.Tensor(out_1x1, in_channels, 1, 1))
        self.bias_1x1 = nn.Parameter(torch.Tensor(out_1x1))
        nn.init.kaiming_uniform_(self.weight_1x1, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_1x1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias_1x1, -bound, bound)
        
        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        # 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        # Max pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
    
    def forward(self, x):
        branch1x1 = conv1x1_triton(x, self.weight_1x1.squeeze(-1).squeeze(-1), self.bias_1x1)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

# Test code
in_channels = 480
out_1x1 = 192
reduce_3x3 = 96
out_3x3 = 208
reduce_5x5 = 16
out_5x5 = 48
pool_proj = 64
batch_size = 10
height = 224
width = 224

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj]
# =================== EVOLVE-BLOCK-END ===================