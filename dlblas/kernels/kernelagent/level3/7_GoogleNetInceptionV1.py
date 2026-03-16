import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

# Enable cuDNN autotuner for potentially faster convolutions on fixed input sizes
torch.backends.cudnn.benchmark = True


@triton.jit
def avg_pool2d_1x1_kernel(
    x_ptr,  # *const T
    y_ptr,  # *T
    N: tl.constexpr,
    C: tl.constexpr,
    H,
    W,
    strideN,
    strideC,
    strideH,  # kept for signature compatibility; we use contiguous path
    strideW,  # kept for signature compatibility; we use contiguous path
    BLOCK_HW: tl.constexpr,
):
    # Each program computes mean over H*W for one (n, c)
    pid = tl.program_id(axis=0)
    NC = N * C
    valid_pid = pid < NC

    # Map pid -> (n, c)
    n = pid // C
    c = pid - n * C

    # Base pointer for start of (n, c, 0, 0)
    base = n * strideN + c * strideC

    # Accumulate sum in fp32 over the entire spatial plane using 1D contiguous indexing
    acc = tl.zeros((), dtype=tl.float32)
    hw_total = H * W

    offs = tl.arange(0, BLOCK_HW)
    k = 0
    while k < hw_total:
        hw_idx = k + offs
        mask = valid_pid & (hw_idx < hw_total)

        # x is ensured contiguous NCHW in the wrapper, so pointers are linear
        ptrs = x_ptr + base + hw_idx
        vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)
        k += BLOCK_HW

    denom = (hw_total).to(tl.float32)
    mean_val = acc / denom

    # Store to y[n, c, 0, 0]; y is allocated contiguous -> linear index pid works
    tl.store(y_ptr + pid, mean_val, mask=valid_pid)


def triton_adaptive_avgpool_1x1(x: torch.Tensor) -> torch.Tensor:
    # Fallback for CPU tensors
    if not x.is_cuda:
        return F.adaptive_avg_pool2d(x, (1, 1))

    # Ensure contiguous NCHW for predictable strides
    x = x.contiguous()
    N, C, H, W = x.shape
    y = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)

    sN, sC, sH, sW = x.stride()
    hw = H * W
    if hw <= 0:
        return torch.zeros_like(y)

    # Next power of two for BLOCK_HW, cap to moderate size to keep register pressure low
    block_hw = 1 << (hw - 1).bit_length()
    block_hw = min(512, max(32, block_hw))

    grid = (N * C,)
    avg_pool2d_1x1_kernel[grid](
        x,
        y,
        N,
        C,
        H,
        W,
        sN,
        sC,
        sH,
        sW,
        BLOCK_HW=block_hw,
        num_warps=1,
        num_stages=1,
    )
    return y


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        """
        :param in_channels: Number of input channels
        :param out_1x1: Number of output channels for the 1x1 convolution
        :param reduce_3x3: Number of output channels for the 1x1 reduction before 3x3 convolution
        :param out_3x3: Number of output channels for the 3x3 convolution
        :param reduce_5x5: Number of output channels for the 1x1 reduction before 5x5 convolution
        :param out_5x5: Number of output channels for the 5x5 convolution
        :param pool_proj: Number of output channels for the pooling projection
        """
        super(InceptionModule, self).__init__()

        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)

        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1),
        )

        # 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2),
        )

        # Max pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: Number of output classes
        """
        super(ModelNew, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        # Use in-place ReLUs to reduce memory traffic
        x = self.maxpool1(F.relu(self.conv1(x), inplace=True))
        x = F.relu(self.conv2(x), inplace=True)
        x = self.maxpool2(F.relu(self.conv3(x), inplace=True))

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        # Triton-optimized AdaptiveAvgPool2d to (1, 1)
        x = triton_adaptive_avgpool_1x1(x)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
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