import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def _channel_shuffle_kernel(
    x_ptr,  # *const T
    y_ptr,  # *mut T
    N,      # int32
    C,      # int32
    S,      # int32 (H*W)
    G,      # int32 (groups)
    K,      # int32 (channels_per_group = C // G)
    BLOCK_S: tl.constexpr,
):
    # 2D launch: axis0 over N*C (each program handles a single (n,c_out)),
    # axis1 over tiles of S (H*W)
    pid_nc = tl.program_id(0)
    pid_s = tl.program_id(1)

    n = pid_nc // C
    c_out = pid_nc % C

    # Compute input channel index corresponding to this output channel
    # c_out = k*G + g  -> c_in = g*K + k, where k = c_out // G, g = c_out % G
    g = c_out % G
    k = c_out // G
    c_in = g * K + k

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = offs_s < S
    tl.multiple_of(offs_s, BLOCK_S)

    base_in = (n * C + c_in) * S
    base_out = (n * C + c_out) * S

    vals = tl.load(x_ptr + base_in + offs_s, mask=mask, other=0)
    tl.store(y_ptr + base_out + offs_s, vals, mask=mask)


def _channel_shuffle_triton(x: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return x
    # Preserve original semantics for invalid shapes / CPU path
    B, C, H, W = x.shape
    if (not x.is_cuda) or (C % groups != 0) or x.numel() == 0:
        channels_per_group = C // groups
        y = x.view(B, groups, channels_per_group, H, W)
        y = y.transpose(1, 2).contiguous()
        return y.view(B, -1, H, W)

    x = x.contiguous()
    B, C, H, W = x.shape
    S = H * W
    K = C // groups

    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    # Slightly larger tile for H200; keeps good occupancy while reducing grid dim on S
    BLOCK_S = 2048 if S >= 2048 else 1024
    grid = (B * C, triton.cdiv(S, BLOCK_S))
    _channel_shuffle_kernel[grid](
        x, y,
        B, C, S, groups, K,
        BLOCK_S=BLOCK_S,
        num_warps=8,
        num_stages=4,
    )
    return y


@triton.jit
def _avg_pool2d_1x1_kernel(
    x_ptr,   # *const T (flattened: [NC*S])
    y_ptr,   # *mut float32 (flattened: [NC])
    S,       # int32 (H*W)
    NC,      # int32 (N*C)
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)  # over NC rows
    # accumulate in float32 for numerical stability
    acc = tl.zeros((), dtype=tl.float32)

    offs = tl.arange(0, BLOCK_HW)
    tl.multiple_of(offs, BLOCK_HW)
    base = pid * S
    for start in tl.range(0, S, BLOCK_HW):
        idx = start + offs
        mask = idx < S
        vals = tl.load(x_ptr + base + idx, mask=mask, other=0).to(tl.float32)
        acc += tl.sum(vals, axis=0)

    mean = acc / S
    tl.store(y_ptr + pid, mean)


def adaptive_avg_pool2d_1x1(x: torch.Tensor) -> torch.Tensor:
    # Fallback to PyTorch on CPU or degenerate tensors
    if (not x.is_cuda) or x.numel() == 0:
        return F.adaptive_avg_pool2d(x, (1, 1))
    x = x.contiguous()
    N, C, H, W = x.shape
    S = H * W
    NC = N * C

    x_flat = x.view(NC, S)
    # accumulate in float32, convert back to input dtype at the end
    y_acc = torch.empty((NC,), device=x.device, dtype=torch.float32)

    BLOCK_HW = 2048 if S >= 2048 else 1024
    grid = (NC,)
    _avg_pool2d_1x1_kernel[grid](
        x_flat, y_acc,
        S, NC,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
        num_stages=4,
    )
    y = y_acc.view(N, C, 1, 1)
    if x.dtype != torch.float32:
        y = y.to(dtype=x.dtype)
    return y


class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        """
        ShuffleNet unit implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param groups: Number of groups for group convolution.
        """
        super(ShuffleNetUnit, self).__init__()
        
        # Ensure the output channels are divisible by groups
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        # First 1x1 group convolution
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # Depthwise 3x3 convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # Second 1x1 group convolution
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shuffle operation
        self.shuffle = ChannelShuffle(groups)
        
        # Shortcut connection if input and output channels are the same
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass for ShuffleNet unit.

        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shuffle(out)
        out = F.relu(self.bn3(self.conv3(out)))
        
        out += self.shortcut(x)
        return out


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        """
        Channel shuffle operation.

        :param groups: Number of groups for shuffling.
        """
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        """
        Forward pass for channel shuffle.

        :param x: Input tensor, shape (batch_size, channels, height, width)
        :return: Output tensor, shape (batch_size, channels, height, width)
        """
        return _channel_shuffle_triton(x, self.groups)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        """
        ShuffleNet architecture.

        :param num_classes: Number of output classes.
        :param groups: Number of groups for group convolution.
        :param stages_repeats: List of ints specifying the number of repeats for each stage.
        :param stages_out_channels: List of ints specifying the output channels for each stage.
        """
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
        """
        Helper function to create a stage of ShuffleNet units.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param repeats: Number of ShuffleNet units in the stage.
        :param groups: Number of groups for group convolution.
        :return: nn.Sequential containing the stage.
        """
        layers = []
        layers.append(ShuffleNetUnit(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnit(out_channels, out_channels, groups))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass for ShuffleNet.

        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = adaptive_avg_pool2d_1x1(x)
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