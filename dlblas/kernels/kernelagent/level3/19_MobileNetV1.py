import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def _avgpool7x7_nchw_kernel(
    x_ptr,  # *f32
    y_ptr,  # *f32
    B, C, H, W,
    stride_b, stride_c, stride_h, stride_w,
    out_stride_b, out_stride_c,
    BLOCK: tl.constexpr,   # use 8 to allow power-of-two arange
    KERNEL: tl.constexpr,  # 7
):
    pid = tl.program_id(axis=0)
    b = pid // C
    c = pid - b * C
    in_bounds = (b < B) & (c < C)

    # Base pointer for (b, c, 0, 0)
    base = x_ptr + b * stride_b + c * stride_c

    # Load an 8x8 tile and mask to 7x7 to simplify address generation
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    ptrs = base + rows[:, None] * stride_h + cols[None, :] * stride_w

    mask = (rows[:, None] < KERNEL) & (cols[None, :] < KERNEL) & in_bounds
    vals = tl.load(ptrs, mask=mask, other=0.0)

    # Reduce over the 7x7 window
    total = tl.sum(vals, axis=0)        # reduce rows -> (BLOCK,)
    total = tl.sum(total, axis=0)       # reduce cols -> scalar

    inv = tl.full((), 1.0 / (KERNEL * KERNEL), vals.dtype)
    avg = total * inv

    out_off = y_ptr + b * out_stride_b + c * out_stride_c
    tl.store(out_off, avg, mask=in_bounds)


def triton_avgpool7x7_nchw(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, C, 7, 7) on CUDA
    returns: (B, C, 1, 1)
    """
    assert x.is_cuda, "triton_avgpool7x7_nchw expects CUDA tensor"
    B, C, H, W = x.shape
    # Fallback safety: only handle exact 7x7
    if H != 7 or W != 7:
        return F.avg_pool2d(x, kernel_size=7)

    # Allocate output as (B, C)
    y2d = torch.empty((B, C), device=x.device, dtype=x.dtype)

    stride_b, stride_c, stride_h, stride_w = x.stride()
    out_stride_b, out_stride_c = y2d.stride()

    grid = (B * C,)
    _avgpool7x7_nchw_kernel[grid](
        x, y2d,
        B, C, H, W,
        stride_b, stride_c, stride_h, stride_w,
        out_stride_b, out_stride_c,
        BLOCK=8, KERNEL=7,
        num_warps=1, num_stages=1,
    )
    return y2d.view(B, C, 1, 1)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        """
        MobileNetV1 architecture implementation.

        :param num_classes: The number of output classes (default: 1000)
        :param input_channels: The number of input channels (default: 3 for RGB images)
        :param alpha: Width multiplier (default: 1.0)
        """
        super(ModelNew, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        # Keep the original MobileNetV1 structure but move AvgPool2d(7) out of this sequential
        # so we can replace it with a Triton kernel in forward() when possible.
        self.features = nn.Sequential(
            conv_bn(input_channels, int(32 * alpha), 2),
            conv_dw(int(32 * alpha), int(64 * alpha), 1),
            conv_dw(int(64 * alpha), int(128 * alpha), 2),
            conv_dw(int(128 * alpha), int(128 * alpha), 1),
            conv_dw(int(128 * alpha), int(256 * alpha), 2),
            conv_dw(int(256 * alpha), int(256 * alpha), 1),
            conv_dw(int(256 * alpha), int(512 * alpha), 2),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(1024 * alpha), 2),
            conv_dw(int(1024 * alpha), int(1024 * alpha), 1),
        )
        self.avgpool_ksize = 7
        self.fc = nn.Linear(int(1024 * alpha), num_classes)

    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_channels, height, width)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.features(x)
        # Replace AvgPool2d(7) with a Triton kernel when possible; otherwise fallback.
        if x.dim() == 4 and x.shape[-2] == 7 and x.shape[-1] == 7:
            if x.is_cuda:
                x = triton_avgpool7x7_nchw(x)
            else:
                # exact equivalent since there's no padding
                x = x.mean(dim=(2, 3), keepdim=True)
        else:
            x = F.avg_pool2d(x, kernel_size=self.avgpool_ksize)

        x = x.view(x.size(0), -1)
        # Use cuBLAS-backed PyTorch linear for top performance and stability
        x = F.linear(x, self.fc.weight, self.fc.bias)
        return x


# Test code
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000
alpha = 1.0

def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [num_classes, input_channels, alpha]