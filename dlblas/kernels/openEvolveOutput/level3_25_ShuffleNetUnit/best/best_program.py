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
    n,
    c,
    h,
    w,
    groups,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = n * c * h * w
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Decompose linear index into tensor indices
    stride_c = h * w
    stride_n = c * stride_c
    stride_h = w
    
    idx = tl.load(input_ptr + offsets, mask=mask)
    off_n = offsets // stride_n
    remainder = offsets % stride_n
    off_c = remainder // stride_c
    remainder2 = remainder % stride_c
    off_h = remainder2 // stride_h
    off_w = remainder2 % stride_h
    
    # Channel shuffle computation
    channels_per_group = c // groups
    group_index = off_c // channels_per_group
    channel_in_group = off_c % channels_per_group
    new_c = channel_in_group * groups + group_index
    
    # Compute output index
    output_idx = off_n * stride_n + new_c * stride_c + off_h * stride_h + off_w
    tl.store(output_ptr + output_idx, idx, mask=mask)

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.contiguous()
        output = torch.empty_like(x)
        total_elements = batch_size * channels * height * width
        
        assert channels % self.groups == 0
        grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
        
        channel_shuffle_kernel[grid](
            x,
            output,
            batch_size,
            channels,
            height,
            width,
            self.groups,
            BLOCK_SIZE=1024
        )
        return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ModelNew, self).__init__()
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

batch_size = 10
input_channels = 240
out_channels = 480
groups = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [input_channels, out_channels, groups]
# =================== EVOLVE-BLOCK-END ===================