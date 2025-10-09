# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def channel_shuffle_kernel(
    input_ptr,
    output_ptr,
    input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
    output_batch_stride, output_channel_stride, output_height_stride, output_width_stride,
    channels, groups, channels_per_group, height, width,
    BLOCK_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    h = pid_hw // width
    w = pid_hw % width
    
    c_offsets = pid_c * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = c_offsets < channels
    
    group_indices = c_offsets % groups
    index_in_group = c_offsets // groups
    input_c = group_indices * channels_per_group + index_in_group
    
    input_offsets = pid_b * input_batch_stride + input_c * input_channel_stride + h * input_height_stride + w * input_width_stride
    output_offsets = pid_b * output_batch_stride + c_offsets * output_channel_stride + h * output_height_stride + w * output_width_stride
    
    values = tl.load(input_ptr + input_offsets, mask=mask, other=0)
    tl.store(output_ptr + output_offsets, values, mask=mask)

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        
        if self.groups == 1:
            return x
            
        output = torch.empty_like(x)
        
        if x.is_cuda:
            BLOCK_SIZE = 128
            grid = (batch_size, height * width, triton.cdiv(channels, BLOCK_SIZE))
            channel_shuffle_kernel[grid](
                x, output,
                x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                output.stride(0), output.stride(1), output.stride(2), output.stride(3),
                channels, self.groups, channels_per_group, height, width,
                BLOCK_SIZE=BLOCK_SIZE
            )
            return output
        else:
            x = x.view(batch_size, self.groups, channels_per_group, height, width)
            x = x.transpose(1, 2).contiguous()
            x = x.view(batch_size, -1, height, width)
            return x

class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnit, self).__init__()
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shuffle = ChannelShuffle(groups)
        
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shuffle(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out += self.shortcut(x)
        return out

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(3, stages_out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)
        
        self.conv5 = nn.Conv2d(stages_out_channels[3], 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)
        self.fc = nn.Linear(1024, num_classes)
    
    def _make_stage(self, in_channels, out_channels, repeats, groups):
        layers = []
        layers.append(ShuffleNetUnit(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnit(out_channels, out_channels, groups))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Test code
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [num_classes]
# =================== EVOLVE-BLOCK-END ===================