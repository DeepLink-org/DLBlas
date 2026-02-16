# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _conv2d_3x3(
    x_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    stride_bx, stride_cx, stride_hx, stride_wx,
    stride_oc, stride_ic, stride_kh, stride_kw,
    stride_bo, stride_oc_out, stride_ho, stride_wo,
    BLOCK_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_s_block = tl.program_id(2)
    
    start_idx = pid_s_block * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (height * width)
    
    oh = offsets // width
    ow = offsets % width
    
    # Fixed tl.zeros shape parameter
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for ic in range(in_channels):
        for ky in range(3):
            for kx in range(3):
                ih = oh + ky - 1
                iw = ow + kx - 1
                
                in_bounds = (ih >= 0) & (ih < height) & (iw >= 0) & (iw < width) & mask
                
                x_ptrs = x_ptr + pid_b * stride_bx + ic * stride_cx + ih * stride_hx + iw * stride_wx
                w_ptr_current = weight_ptr + pid_oc * stride_oc + ic * stride_ic + ky * stride_kh + kx * stride_kw
                
                w_val = tl.load(w_ptr_current)
                x_val = tl.load(x_ptrs, mask=in_bounds, other=0.0)
                acc += x_val * w_val
                
    out_ptrs = output_ptr + pid_b * stride_bo + pid_oc * stride_oc_out + oh * stride_ho + ow * stride_wo
    tl.store(out_ptrs, acc, mask=mask)

class TritonConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 3))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        output = torch.empty(batch_size, self.out_channels, height, width, 
                             device=x.device, dtype=x.dtype)
        
        if not x.is_contiguous():
            x = x.contiguous()
        if not self.weight.is_contiguous():
            self.weight.data = self.weight.data.contiguous()
            
        BLOCK_SIZE = 128
        num_spatial_blocks = (height * width + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid = (batch_size, self.out_channels, num_spatial_blocks)
        
        _conv2d_3x3[grid](
            x, self.weight, output,
            batch_size, in_channels, self.out_channels, height, width,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.weight.stride(0), self.weight.stride(1), 
            self.weight.stride(2), self.weight.stride(3),
            output.stride(0), output.stride(1),
            output.stride(2), output.stride(3),
            BLOCK_SIZE=BLOCK_SIZE
        )
        return output

class ModelNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(ModelNew, self).__init__()
        self.num_layers = num_layers
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate
        
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
        batch_size, _, height, width = x.size()
        total_channels = self.num_input_features + self.num_layers * self.growth_rate
        out = torch.zeros(batch_size, total_channels, height, width, device=x.device, dtype=x.dtype)
        out[:, :self.num_input_features] = x
        
        current_channels = self.num_input_features
        features = out[:, :current_channels]
        
        for layer in self.layers:
            new_feature = layer(features)
            end_channel = current_channels + self.growth_rate
            out[:, current_channels:end_channel] = new_feature
            current_channels = end_channel
            features = out[:, :current_channels]
            
        return out
    
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