# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    x_ptr, weight_ptr, output_ptr,
    stride_bx, stride_cx, stride_hx, stride_wx,
    stride_ow, stride_cw, stride_hw, stride_ww,
    stride_bo, stride_co, stride_ho, stride_wo,
    H, W, K: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1) * BLOCK_H
    pid_w = tl.program_id(2) * BLOCK_W
    pid_c = tl.program_id(3) * BLOCK_C
    
    # Create pointers
    x_ptrs = x_ptr + pid_b * stride_bx + (pid_h + tl.arange(0, BLOCK_H)[:, None] - 1) * stride_hx + (pid_w + tl.arange(0, BLOCK_W)[None, :] - 1) * stride_wx
    w_ptrs = weight_ptr + pid_c * stride_ow + tl.arange(0, BLOCK_C)[:, None, None] * stride_cw + tl.arange(0, 3)[None, :, None] * stride_hw + tl.arange(0, 3)[None, None, :] * stride_ww
    o_ptrs = output_ptr + pid_b * stride_bo + pid_c * stride_co + pid_h * stride_ho + pid_w * stride_wo
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_C, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Loop over input channels
    for c in range(0, K, BLOCK_C):
        # Load input tile
        mask_x = (pid_h + tl.arange(0, BLOCK_H)[:, None] >= 0) & (pid_h + tl.arange(0, BLOCK_H)[:, None] < H) & \
                 (pid_w + tl.arange(0, BLOCK_W)[None, :] >= 0) & (pid_w + tl.arange(0, BLOCK_W)[None, :] < W)
        x = tl.load(x_ptrs + c * stride_cx, mask=mask_x, other=0.0)
        
        # Load weight tile
        w = tl.load(w_ptrs + c * stride_cw)
        
        # Compute convolution
        for kh in range(3):
            for kw in range(3):
                x_slice = tl.view(x[kh:kh+BLOCK_H, kw:kw+BLOCK_W], (1, BLOCK_H, BLOCK_W))
                w_slice = tl.view(w[:, kh, kw], (BLOCK_C, 1, 1))
                acc += w_slice * x_slice
    
    # Store output
    mask_o = (pid_c + tl.arange(0, BLOCK_C)[:, None, None] < output_ptr.shape[1]) & \
             (pid_h + tl.arange(0, BLOCK_H)[None, :, None] < H) & \
             (pid_w + tl.arange(0, BLOCK_W)[None, None, :] < W)
    tl.store(o_ptrs, acc, mask=mask_o)

class TritonConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.padding = padding
        
    def forward(self, x):
        output = torch.empty(x.shape[0], self.weight.shape[0], x.shape[2], x.shape[3], 
                             device=x.device, dtype=x.dtype)
        
        # Grid configuration
        B, C, H, W = x.shape
        grid = lambda opt: (B, 
                           triton.cdiv(H, opt['BLOCK_H']), 
                           triton.cdiv(W, opt['BLOCK_W']), 
                           triton.cdiv(self.weight.shape[0], opt['BLOCK_C']))
        
        # Launch kernel
        conv2d_kernel[grid](
            x, self.weight, output,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.weight.stride(0), self.weight.stride(1), self.weight.stride(2), self.weight.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            H, W, C,
            BLOCK_H=16, BLOCK_W=16, BLOCK_C=32
        )
        return output

class ModelNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(ModelNew, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            TritonConv2d(in_features, growth_rate),
            nn.Dropout(0.0)
        )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)
        return x
    
batch_size = 10
num_layers = 6
num_input_features = 32
growth_rate = 32
height, width = 224, 224

def get_inputs():
    return [torch.randn(batch_size, num_input_features, height, width)]

def get_init_inputs():
    return [num_layers, num_input_features , growth_rate]
# =================== EVOLVE-BLOCK-END ===================