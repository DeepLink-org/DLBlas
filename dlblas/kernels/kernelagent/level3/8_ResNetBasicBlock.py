import torch
import torch.nn as nn
import torch.nn.functional as F

# Triton import
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


@triton.jit
def _add_relu_fwd(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    y = tl.load(y_ptr + offs, mask=mask, other=0)
    z = x + y
    # ReLU
    z = tl.maximum(z, 0)
    tl.store(out_ptr + offs, z, mask=mask)


def fused_add_relu_(x: torch.Tensor, y: torch.Tensor):
    """
    In-place: x <- relu(x + y)
    Requirements:
      - x, y: same shape, dtype, device
      - contiguous
    Fallbacks to PyTorch ops if requirements aren't met or not CUDA.
    """
    if (
        _TRITON_AVAILABLE
        and x.is_cuda
        and y.is_cuda
        and x.dtype == y.dtype
        and x.is_contiguous()
        and y.is_contiguous()
        and x.numel() == y.numel()
    ):
        n_elements = x.numel()
        BLOCK_SIZE = 4096
        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
        _add_relu_fwd[grid](
            x, y, x, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=4, num_stages=2
        )
        return x
    # Fallback path
    x.add_(y)
    return F.relu_(x)


class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride for the first convolutional layer
        :param downsample: Downsample layer for the shortcut connection
        """
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        # Preserve semantics of inplace=True ReLU
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Fused add + ReLU in a single Triton kernel pass (in-place on 'out')
        fused_add_relu_(out, identity)

        return out
    
# Test code
in_channels = 3
out_channels = 64
stride = 1
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, in_channels, 224, 224)]

def get_init_inputs():
    return [in_channels, out_channels, stride]