# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv1x1_relu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    n_rows,
    in_channels,
    out_channels,
    stride_x,
    stride_w,
    stride_b,
    stride_y,
    BLOCK_SIZE_IC: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_oc_block = tl.program_id(1)
    
    oc_offsets = pid_oc_block * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC)
    oc_mask = oc_offsets < out_channels
    
    row_x_ptr = x_ptr + pid_row * stride_x
    row_y_ptr = y_ptr + pid_row * stride_y + oc_offsets
    
    acc = tl.zeros((BLOCK_SIZE_OC,), dtype=tl.float32)
    
    for ic_block in range(0, tl.cdiv(in_channels, BLOCK_SIZE_IC)):
        ic_offsets = ic_block * BLOCK_SIZE_IC + tl.arange(0, BLOCK_SIZE_IC)
        ic_mask = ic_offsets < in_channels
        
        x_vals = tl.load(row_x_ptr + ic_offsets, mask=ic_mask, other=0.0)
        w_vals = tl.load(w_ptr + oc_offsets[:, None] * stride_w + ic_offsets[None, :], 
                         mask=oc_mask[:, None] & ic_mask[None, :], other=0.0)
        
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)
    
    if b_ptr is not None:
        b_vals = tl.load(b_ptr + oc_offsets, mask=oc_mask, other=0.0)
        acc += b_vals
        
    acc = tl.maximum(acc, 0.0)
    tl.store(row_y_ptr, acc, mask=oc_mask)

def conv1x1_relu(x, weight, bias=None):
    n, ic, h, w = x.shape
    oc = weight.shape[0]
    x_flat = x.reshape(n*h*w, ic).contiguous()
    y_flat = torch.empty((n*h*w, oc), device=x.device, dtype=x.dtype)
    
    grid = (n*h*w, triton.cdiv(oc, 64))
    conv1x1_relu_kernel[grid](
        x_flat, weight, bias, y_flat,
        n*h*w, ic, oc,
        x_flat.stride(0), weight.stride(0), 
        bias.stride(0) if bias is not None else 0,
        y_flat.stride(0),
        BLOCK_SIZE_IC=64, BLOCK_SIZE_OC=64
    )
    return y_flat.reshape(n, oc, h, w)

class ModelNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(ModelNew, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = conv1x1_relu(x, self.squeeze.weight, self.squeeze.bias)
        branch1x1 = conv1x1_relu(x, self.expand1x1.weight, self.expand1x1.bias)
        branch3x3 = self.expand3x3_activation(self.expand3x3(x))
        return torch.cat([branch1x1, branch3x3], 1)

# Test code
batch_size = 10
num_input_features = 3
num_output_features = 64
height, width = 224, 224
squeeze_channels = 6
expand1x1_channels = 64
expand3x3_channels = 64

def get_inputs():
    return [torch.randn(batch_size, num_input_features, height, width)]

def get_init_inputs():
    return [num_input_features, squeeze_channels, expand1x1_channels, expand3x3_channels]
# =================== EVOLVE-BLOCK-END ===================