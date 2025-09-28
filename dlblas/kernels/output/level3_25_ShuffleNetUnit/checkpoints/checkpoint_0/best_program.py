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
    num_pid_n = tl.cdiv(n * h * w, BLOCK_SIZE)
    pid_n = pid // num_pid_n
    pid_s = pid % num_pid_n
    
    off_s = pid_s * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off_n = off_s // (h * w)
    off_hw = off_s % (h * w)
    off_h = off_hw // w
    off_w = off_hw % w
    
    channels_per_group = c // groups
    group_index = off_n // channels_per_group
    channel_in_group = off_n % channels_per_group
    new_c = channel_in_group * groups + group_index
    
    input_offset = off_n * (h * w * c) + off_c * (h * w) + off_h * w + off_w
    output_offset = off_n * (h * w * c) + new_c * (h * w) + off_h * w + off_w
    
    mask = (off_n < n) & (off_c < c) & (off_h < h) & (off_w < w)
    val = tl.load(input_ptr + input_offset, mask=mask)
    tl.store(output_ptr + output_offset, val, mask=mask)

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.contiguous()
        output = torch.empty_like(x)
        
        assert channels % self.groups == 0
        grid = (triton.cdiv(batch_size * height * width * channels, 256),)
        channel_shuffle_kernel[grid](
            x,
            output,
            batch_size,
            channels,
            height,
            width,
            self.groups,
            BLOCK_SIZE=256,
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