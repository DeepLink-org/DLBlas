import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


@triton.jit
def _gap1x1_row_kernel(
    x_ptr, y_ptr,
    B: tl.constexpr, C: tl.constexpr,
    H, W,
    stride_b, stride_c, stride_h, stride_w,
    out_stride_b, out_stride_c,
    BLOCK_W: tl.constexpr,
):
    # One program per (b, c), iterate rows and vectorize across width to avoid atomics.
    pid = tl.program_id(axis=0)
    b = pid // C
    c = pid % C

    base = b * stride_b + c * stride_c

    offs_w = tl.arange(0, BLOCK_W)
    mask_w = offs_w < W

    acc = tl.zeros((), dtype=tl.float32)
    h = 0
    # Iterate over rows; each iteration loads one full row (vectorized over W)
    while h < H:
        ptrs = x_ptr + base + h * stride_h + offs_w * stride_w
        vals = tl.load(ptrs, mask=mask_w, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)
        h += 1

    denom = tl.full((), 1.0, dtype=tl.float32) * H * W
    mean = acc / denom

    out_ptr = y_ptr + b * out_stride_b + c * out_stride_c
    tl.store(out_ptr, mean)


def adaptive_avg_pool2d_1x1_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Triton-accelerated equivalent of F.adaptive_avg_pool2d(x, (1, 1)).
    Returns tensor of shape [B, C, 1, 1] with the same dtype/device as x.
    Falls back to PyTorch if not CUDA.
    """
    if (not x.is_cuda) or x.numel() == 0:
        return F.adaptive_avg_pool2d(x, (1, 1))

    x = x.contiguous()
    B, C, H, W = x.shape
    y = torch.empty((B, C, 1, 1), device=x.device, dtype=x.dtype)

    sb, sc, sh, sw = x.stride()
    out_sb, out_sc, _, _ = y.stride()

    # Vectorize across width; choose a power-of-2 >= W but cap to keep registers small.
    BLOCK_W = min(_next_pow2(W), 128)

    grid = (B * C,)
    _gap1x1_row_kernel[grid](
        x, y,
        B, C, H, W,
        sb, sc, sh, sw,
        out_sb, out_sc,
        BLOCK_W=BLOCK_W,
        num_warps=1,  # small vector width; many CTAs (B*C) => good occupancy
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB0 architecture implementation in PyTorch.

        :param num_classes: The number of output classes (default is 1000 for ImageNet).
        """
        super(ModelNew, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # MBConv blocks
        self.blocks = nn.Sequential(
            # MBConv1 (32, 16, 1, 1)
            MBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            # MBConv6 (16, 24, 2, 6)
            MBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (24, 24, 1, 6)
            MBConv(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (24, 40, 2, 6)
            MBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (40, 40, 1, 6)
            MBConv(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (40, 80, 2, 6)
            MBConv(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (80, 80, 1, 6)
            MBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (80, 112, 1, 6)
            MBConv(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 112, 1, 6)
            MBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 192, 2, 6)
            MBConv(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 320, 1, 6)
            MBConv(192, 320, kernel_size=3, stride=1, expand_ratio=6)
        )
        
        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        """
        Forward pass of the EfficientNetB0 model.

        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = F.relu(self.bn2(self.conv2(x)))
        # Triton-accelerated global average pooling to (1, 1)
        x = adaptive_avg_pool2d_1x1_triton(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        MBConv block implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Kernel size for the depthwise convolution.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the intermediate channels.
        """
        super(MBConv, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        """
        Forward pass of the MBConv block.

        :param x: The input tensor, shape (batch_size, in_channels, H, W)
        :return: The output tensor, shape (batch_size, out_channels, H', W')
        """
        identity = x
        
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        
        if self.use_residual:
            x += identity
        
        return x

# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]