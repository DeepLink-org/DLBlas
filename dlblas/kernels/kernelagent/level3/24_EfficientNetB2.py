import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def _gap2d_1x1_kernel(
    x_ptr,  # *float32
    y_ptr,  # *float32
    N: tl.constexpr,
    C: tl.constexpr,
    H, W,
    stride_n, stride_c, stride_h, stride_w,
    y_stride_n, y_stride_c,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nc = N * C
    if pid >= nc:
        return
    n = pid // C
    c = pid % C

    # Base pointer to (n, c, 0, 0)
    base = x_ptr + n * stride_n + c * stride_c
    M = H * W

    # Vectorized offsets (constexpr)
    offs = tl.arange(0, BLOCK)
    # Accumulate in fp32 for numerical stability
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    # 2x unrolled loop over the HW plane with linear indexing (input is made contiguous)
    step = BLOCK * 2
    for start in range(0, M, step):
        idx0 = start + offs
        m0 = idx0 < M
        vals0 = tl.load(base + idx0, mask=m0, other=0.0)
        acc += vals0.to(tl.float32)

        idx1 = idx0 + BLOCK
        m1 = idx1 < M
        vals1 = tl.load(base + idx1, mask=m1, other=0.0)
        acc += vals1.to(tl.float32)

    total = tl.sum(acc, axis=0)
    mean = total / (H * W)

    y_ptrs = y_ptr + n * y_stride_n + c * y_stride_c
    tl.store(y_ptrs, mean)


def _adaptive_avg_pool2d_1x1_triton(x: torch.Tensor) -> torch.Tensor:
    # Fallbacks for unsupported cases
    if (not x.is_cuda) or (x.dtype != torch.float32):
        return F.adaptive_avg_pool2d(x, (1, 1))

    N, C, H, W = x.shape
    # Identity when input is already 1x1
    if H == 1 and W == 1:
        return x

    # Ensure NCHW contiguous so that linear indexing along H*W is valid
    x_c = x.contiguous()
    y = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)

    stride_n, stride_c, stride_h, stride_w = x_c.stride()
    y_stride_n, y_stride_c, _, _ = y.stride()

    grid = (N * C,)
    _gap2d_1x1_kernel[grid](
        x_c, y,
        N, C, H, W,
        stride_n, stride_c, stride_h, stride_w,
        y_stride_n, y_stride_c,
        BLOCK=1024,
        num_warps=4,
        num_stages=2,
    )
    return y


class AdaptiveAvgPool2d1x1Triton(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _adaptive_avg_pool2d_1x1_triton(x)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB2 architecture implementation.

        :param num_classes: The number of output classes (default is 1000 for ImageNet).
        """
        super(ModelNew, self).__init__()
        
        # Define the EfficientNetB2 architecture components
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Define the MBConv blocks
        self.mbconv1 = self._make_mbconv_block(32, 96, 1, 3)
        self.mbconv2 = self._make_mbconv_block(96, 144, 2, 6)
        self.mbconv3 = self._make_mbconv_block(144, 192, 2, 6)
        self.mbconv4 = self._make_mbconv_block(192, 288, 2, 6)
        self.mbconv5 = self._make_mbconv_block(288, 384, 1, 6)
        
        # Final layers
        self.conv_final = nn.Conv2d(384, 1408, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_final = nn.BatchNorm2d(1408)
        self.avgpool = AdaptiveAvgPool2d1x1Triton()
        self.fc = nn.Linear(1408, num_classes)

        # Let cuDNN pick optimal kernels for the fixed input size
        torch.backends.cudnn.benchmark = True
    
    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        """
        Helper function to create a MBConv block.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the MBConv block.
        :return: A sequential container of layers forming the MBConv block.
        """
        layers = []
        expanded_channels = in_channels * expand_ratio
        
        # Expansion phase
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(expanded_channels))
            layers.append(nn.ReLU(inplace=True))
        
        # Depthwise convolution
        layers.append(nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False))
        layers.append(nn.BatchNorm2d(expanded_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Squeeze and Excitation (Triton-accelerated 1x1 adaptive avg pooling)
        layers.append(AdaptiveAvgPool2d1x1Triton())
        layers.append(nn.Conv2d(expanded_channels, expanded_channels // 4, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(expanded_channels // 4, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.Sigmoid())
        
        # Output phase
        layers.append(nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass of the EfficientNetB2 model.

        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.relu(self.bn_final(self.conv_final(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Test code
batch_size = 2
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]