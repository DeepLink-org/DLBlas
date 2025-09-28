# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triton_conv_transpose2d(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    stride, padding, dilation,
    B, IC, OC, H_in, W_in, H_out, W_out, kernel_size,
    stride_x, stride_w, stride_out,
    BLOCK_OH: tl.constexpr, BLOCK_OW: tl.constexpr, BLOCK_IC: tl.constexpr
):
    pid_b_oc = tl.program_id(0)
    pid_oh_block = tl.program_id(1)
    pid_ow_block = tl.program_id(2)
    
    batch_idx = pid_b_oc // OC
    oc_idx = pid_b_oc % OC
    oh_start = pid_oh_block * BLOCK_OH
    ow_start = pid_ow_block * BLOCK_OW
    
    # Initialize accumulator for output block
    acc = tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.float32)
    
    # Process kernel positions
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Calculate base input coordinates for output block
            oh_indices = oh_start + tl.arange(0, BLOCK_OH)
            ow_indices = ow_start + tl.arange(0, BLOCK_OW)
            ih0 = oh_indices[:, None] + padding - kh * dilation
            iw0 = ow_indices[None, :] + padding - kw * dilation
            
            # Check divisibility by stride
            cond_ih = (ih0 % stride == 0)
            cond_iw = (iw0 % stride == 0)
            cond_div = cond_ih & cond_iw
            
            # Compute input indices
            ih = tl.where(cond_div, ih0 // stride, -1)
            iw = tl.where(cond_div, iw0 // stride, -1)
            
            # Check spatial boundaries
            cond_in_bounds = (ih >= 0) & (ih < H_in) & (iw >= 0) & (iw < W_in)
            cond_valid = cond_div & cond_in_bounds
            
            # Process input channels in blocks
            for ic_block in range(0, IC, BLOCK_IC):
                ic_offsets = ic_block + tl.arange(0, BLOCK_IC)
                mask_ic = ic_offsets < IC
                
                # Compute input and weight offsets
                x_offset = batch_idx * stride_x[0] + ih * stride_x[2] + iw * stride_x[3]
                weight_offset = kh * stride_w[2] + kw * stride_w[3] + oc_idx * stride_w[1]
                
                # Vectorized loads for input and weight
                x_vals = tl.load(
                    x_ptr + x_offset[:, :, None] + ic_offsets[None, None, :] * stride_x[1],
                    mask=cond_valid[:, :, None] & mask_ic[None, None, :],
                    other=0.0
                )
                w_vals = tl.load(
                    weight_ptr + weight_offset + ic_offsets[None, None, :] * stride_w[0],
                    mask=mask_ic[None, None, :],
                    other=0.0
                )
                
                # Accumulate products
                acc += tl.sum(x_vals * w_vals, axis=2)
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + oc_idx)
        acc += bias_val
    
    # Store results
    oh_indices = oh_start + tl.arange(0, BLOCK_OH)
    ow_indices = ow_start + tl.arange(0, BLOCK_OW)
    out_offset = (
        batch_idx * stride_out[0] + 
        oc_idx * stride_out[1] + 
        oh_indices[:, None] * stride_out[2] + 
        ow_indices[None, :] * stride_out[3]
    )
    mask = (oh_indices[:, None] < H_out) & (ow_indices[None, :] < W_out)
    tl.store(output_ptr + out_offset, acc, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, IC, H_in, W_in = x.shape
        H_out = (H_in - 1) * self.stride - 2 * self.padding + \
                self.dilation * (self.kernel_size - 1) + 1
        W_out = (W_in - 1) * self.stride - 2 * self.padding + \
                self.dilation * (self.kernel_size - 1) + 1
        
        output = torch.empty((B, self.out_channels, H_out, W_out), 
                            device=x.device, dtype=x.dtype)
        
        grid_b_oc = B * self.out_channels
        grid_oh = triton.cdiv(H_out, 4)
        grid_ow = triton.cdiv(W_out, 4)
        grid = (grid_b_oc, grid_oh, grid_ow)
        
        bias_ptr = self.bias if self.bias is not None else None
        
        triton_conv_transpose2d[grid](
            x, self.weight, bias_ptr, output,
            self.stride, self.padding, self.dilation,
            B, IC, self.out_channels, H_in, W_in, H_out, W_out, self.kernel_size,
            x.stride(), self.weight.stride(), output.stride(),
            BLOCK_OH=4, BLOCK_OW=4, BLOCK_IC=32
        )
        
        return output

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
height_in = 64
width_in = 128
stride = 5
padding = 1
dilation = 2

def get_inputs():
    x = torch.randn(batch_size, in_channels, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
# =================== EVOLVE-BLOCK-END ===================