# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    stride_ow, stride_oc, stride_ic, stride_kh, stride_kw,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    elements_per_program = (height * width + num_pid - 1) // num_pid
    start_idx = pid * elements_per_program
    end_idx = tl.minimum(start_idx + elements_per_program, height * width)

    for idx in range(start_idx, end_idx):
        oh = idx // width
        ow = idx % width
        
        for b in range(batch_size):
            for oc in range(out_channels):
                acc = 0.0
                for ic in range(in_channels):
                    for ky in range(3):
                        for kx in range(3):
                            ih = oh + ky - 1
                            iw = ow + kx - 1
                            if ih >= 0 and ih < height and iw >= 0 and iw < width:
                                x_offset = b * stride_bx + ic * stride_cx + ih * stride_hx + iw * stride_wx
                                w_offset = oc * stride_oc + ic * stride_ic + ky * stride_kh + kx * stride_kw
                                x_val = tl.load(x_ptr + x_offset)
                                w_val = tl.load(weight_ptr + w_offset)
                                acc += x_val * w_val
                
                out_offset = b * out_channels * height * width + oc * height * width + oh * width + ow
                tl.store(output_ptr + out_offset, acc)

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
        
        if x.is_contiguous() is False:
            x = x.contiguous()
        if self.weight.is_contiguous() is False:
            self.weight.data = self.weight.data.contiguous()
            
        total_elements = height * width
        grid = lambda meta: (min(triton.cdiv(total_elements, meta['BLOCK_SIZE']), 1024),)
        
        _conv2d_3x3[grid](
            x, self.weight, output,
            batch_size, in_channels, self.out_channels, height, width,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.weight.stride(0), self.weight.stride(1), 
            self.weight.stride(2), self.weight.stride(3),
            BLOCK_SIZE=128
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