import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _channel_shuffle_kernel(
    in_ptr, out_ptr,
    N, C, H, W, G,
    BLOCK_C: tl.constexpr, BLOCK_PIX: tl.constexpr
):
    pid_pix = tl.program_id(0)
    pid_c = tl.program_id(1)

    S_HW = H * W
    TOT = N * S_HW

    pix_start = pid_pix * BLOCK_PIX
    c_start = pid_c * BLOCK_C

    pix_offsets = pix_start + tl.arange(0, BLOCK_PIX)
    c_offsets = c_start + tl.arange(0, BLOCK_C)

    mask_pix = pix_offsets < TOT
    mask_c = c_offsets < C

    # 2D broadcast for (channels x pixels) tile
    pix_offsets_2d = pix_offsets[None, :]          # (1, BLOCK_PIX)
    c_offsets_2d = c_offsets[:, None]              # (BLOCK_C, 1)
    mask = mask_c[:, None] & mask_pix[None, :]     # (BLOCK_C, BLOCK_PIX)

    # Decompose flattened pixel offsets into (n, s) where
    # n in [0, N), s in [0, H*W)
    n = pix_offsets_2d // S_HW
    s = pix_offsets_2d % S_HW

    # Load input tile using correct NCHW linearization:
    # idx = ((n * C) + c) * (H*W) + s
    in_idx = ((n * C) + c_offsets_2d) * S_HW + s
    vals = tl.load(in_ptr + in_idx, mask=mask, other=0)

    # Compute destination channel index after shuffle:
    # c = g*K + k  ->  c' = k*G + g, where K = C // G
    K = C // G
    k = c_offsets_2d % K
    g = c_offsets_2d // K
    dest_c = k * G + g

    # Store to output with same NCHW layout
    out_idx = ((n * C) + dest_c) * S_HW + s
    tl.store(out_ptr + out_idx, vals, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        """
        ShuffleNet unit implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param groups: Number of groups for group convolution.
        """
        super(ModelNew, self).__init__()

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
        batch_size, channels, height, width = x.size()
        assert channels % self.groups == 0, "Channels must be divisible by groups"
        # Use Triton kernel when running on CUDA for faster, coalesced shuffle
        if x.is_cuda:
            x_c = x.contiguous()
            out = torch.empty_like(x_c)
            N, C, H, W = x_c.shape
            # tile along flattened pixel dimension and channels
            BLOCK_PIX = 256
            BLOCK_C = 32
            grid = (triton.cdiv(N * H * W, BLOCK_PIX), triton.cdiv(C, BLOCK_C))
            _channel_shuffle_kernel[grid](
                x_c, out,
                N, C, H, W, self.groups,
                BLOCK_C=BLOCK_C, BLOCK_PIX=BLOCK_PIX,
                num_warps=8, num_stages=3
            )
            return out
        # CPU fallback (reference semantics)
        channels_per_group = channels // self.groups
        xv = x.view(batch_size, self.groups, channels_per_group, height, width)
        xvt = xv.transpose(1, 2).contiguous()
        return xvt.view(batch_size, -1, height, width)


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