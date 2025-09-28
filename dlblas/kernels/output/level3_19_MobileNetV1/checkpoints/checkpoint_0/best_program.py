# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def global_avg_pool_kernel(
    input_ptr, output_ptr,
    input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
    output_batch_stride, output_channel_stride,
    height, width,
    BLOCK_SIZE: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    
    input_offset = pid_batch * input_batch_stride + pid_channel * input_channel_stride
    spatial_indices = tl.arange(0, BLOCK_SIZE)
    h = spatial_indices // width
    w = spatial_indices % width
    mask = spatial_indices < (height * width)
    
    spatial_offsets = h * input_height_stride + w * input_width_stride
    pointers = input_ptr + input_offset + spatial_offsets
    
    values = tl.load(pointers, mask=mask, other=0.0)
    channel_sum = tl.sum(values)
    channel_avg = channel_sum / (height * width)
    
    output_offset = pid_batch * output_batch_stride + pid_channel * output_channel_stride
    tl.store(output_ptr + output_offset, channel_avg)

class GlobalAvgPool2dTriton(nn.Module):
    def forward(self, x):
        batch_size, channels = x.shape[0], x.shape[1]
        output = torch.empty((batch_size, channels), device=x.device, dtype=x.dtype)
        
        grid = (batch_size, channels)
        spatial_size = x.shape[2] * x.shape[3]
        global_avg_pool_kernel[grid](
            x, output,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            output.stride(0), output.stride(1),
            x.shape[2], x.shape[3],
            BLOCK_SIZE=spatial_size
        )
        return output

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        super(ModelNew, self).__init__()
        
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        
        self.features = nn.Sequential(
            conv_bn(input_channels, int(32 * alpha), 2),
            conv_dw(int(32 * alpha), int(64 * alpha), 1),
            conv_dw(int(64 * alpha), int(128 * alpha), 2),
            conv_dw(int(128 * alpha), int(128 * alpha), 1),
            conv_dw(int(128 * alpha), int(256 * alpha), 2),
            conv_dw(int(256 * alpha), int(256 * alpha), 1),
            conv_dw(int(256 * alpha), int(512 * alpha), 2),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(1024 * alpha), 2),
            conv_dw(int(1024 * alpha), int(1024 * alpha), 1),
        )
        self.avgpool = GlobalAvgPool2dTriton()
        self.fc = nn.Linear(int(1024 * alpha), num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

# Test code
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000
alpha = 1.0

def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [num_classes, input_channels, alpha]
# =================== EVOLVE-BLOCK-END ===================