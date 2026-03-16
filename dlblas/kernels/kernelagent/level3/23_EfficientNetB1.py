import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

# Enable cuDNN autotuner to potentially speed up convolutions with channels_last
torch.backends.cudnn.benchmark = True


@triton.jit
def _relu_gap_hw_tile(
    x_ptr,  # *float32 [N, C, H, W]
    y_ptr,  # *float32 [N, C]
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc,
):
    pid = tl.program_id(0)
    nc = N * C
    valid = pid < nc

    # map pid -> (n, c)
    n = tl.where(valid, pid // C, 0)
    c = tl.where(valid, pid % C, 0)

    base = n * stride_xn + c * stride_xc

    # Full HxW tile
    offs_h = tl.arange(0, H)
    offs_w = tl.arange(0, W)
    ptrs = base + offs_h[:, None] * stride_xh + offs_w[None, :] * stride_xw

    # Only guard invalid pid; H and W are exact tile sizes
    vals = tl.load(x_ptr + ptrs, mask=valid, other=0.0)
    # ReLU then accumulate in fp32
    vals = tl.maximum(vals, 0.0).to(tl.float32)
    # Reduce across both spatial dimensions
    tile_sum = tl.sum(vals, axis=1)
    tile_sum = tl.sum(tile_sum, axis=0)

    inv_denom = 1.0 / (H * W)
    out = tile_sum * inv_denom

    y_off = n * stride_yn + c * stride_yc
    tl.store(y_ptr + y_off, out, mask=valid)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB1 architecture implementation.

        :param num_classes: The number of output classes (default is 1000 for ImageNet).
        """
        super(ModelNew, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # MBConv blocks
        self.mbconv1 = self._make_mbconv_block(32, 16, 1, 1)
        self.mbconv2 = self._make_mbconv_block(16, 24, 2, 6)
        self.mbconv3 = self._make_mbconv_block(24, 40, 2, 6)
        self.mbconv4 = self._make_mbconv_block(40, 80, 2, 6)
        self.mbconv5 = self._make_mbconv_block(80, 112, 1, 6)
        self.mbconv6 = self._make_mbconv_block(112, 192, 2, 6)
        self.mbconv7 = self._make_mbconv_block(192, 320, 1, 6)
        
        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
    
    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        """
        Creates a MBConv block.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param stride: Stride of the depthwise convolution.
        :param expand_ratio: Expansion ratio for the hidden layer.
        :return: A sequential MBConv block.
        """
        hidden_dim = round(in_channels * expand_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        """
        Forward pass of the EfficientNetB1 model.

        :param x: Input tensor, shape (batch_size, 3, 240, 240)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        # Use channels_last to speed up convolutions on NVIDIA GPUs
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)

        x = self.bn1(self.conv1(x))
        x = F.relu(x, inplace=True)
        
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)
        
        # Fuse ReLU + Global Average Pool with a Triton kernel on top of BatchNorm output
        x = self.bn2(self.conv2(x))
        N, C, H, W = x.shape

        # Output after pooling is [N, C]
        y = torch.empty((N, C), device=x.device, dtype=x.dtype)

        xs = x.stride()
        ys = y.stride()
        grid = (N * C,)

        _relu_gap_hw_tile[grid](
            x, y,
            N, C, H, W,
            xs[0], xs[1], xs[2], xs[3],
            ys[0], ys[1],
            num_warps=1,  # 1 warp is sufficient for 8x8 tiles and reduces overhead
            num_stages=1,
        )

        # y is already flattened [N, C]
        x = self.fc(y)
        return x


# Test code
batch_size = 10
input_shape = (3, 240, 240)
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [num_classes]