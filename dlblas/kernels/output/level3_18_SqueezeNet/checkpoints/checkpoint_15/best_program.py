# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def adaptive_avg_pool_kernel(
    input_ptr,
    output_ptr,
    input_batch_stride,
    input_channel_stride,
    input_height_stride,
    input_width_stride,
    output_batch_stride,
    output_channel_stride,
    height,
    width,
    BLOCK_SIZE: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    
    total = 0.0
    num_elements = height * width
    
    for i in range(0, num_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        h_offsets = offsets // width
        w_offsets = offsets % width
        
        mask = offsets < num_elements
        h_mask = h_offsets < height
        w_mask = w_offsets < width
        mask = mask & h_mask & w_mask
        
        input_offset = pid_batch * input_batch_stride + \
                       pid_channel * input_channel_stride + \
                       h_offsets * input_height_stride + \
                       w_offsets * input_width_stride
        values = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
        total += tl.sum(values, axis=0)
    
    avg = total / num_elements
    output_offset = pid_batch * output_batch_stride + pid_channel * output_channel_stride
    tl.store(output_ptr + output_offset, avg)

def adaptive_avg_pool2d_1x1(x: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = x.shape
    output = torch.empty((batch, channels, 1, 1), device=x.device, dtype=x.dtype)
    grid = (batch, channels)
    
    # Use power-of-two block size for efficient reduction
    block_size = triton.next_power_of_2(height * width)
    if block_size > 1024:  # Cap block size for register efficiency
        block_size = 1024
    
    adaptive_avg_pool_kernel[grid](
        x,
        output,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        output.stride(0), output.stride(1),
        height,
        width,
        BLOCK_SIZE=block_size
    )
    return output

class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModule, self).__init__()
        
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(96, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            FireModule(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(256, 32, 128, 128),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(512, 64, 256, 256),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = adaptive_avg_pool2d_1x1(x)
        return torch.flatten(x, 1)

# Test code
batch_size = 1
input_channels = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [num_classes]
# =================== EVOLVE-BLOCK-END ===================