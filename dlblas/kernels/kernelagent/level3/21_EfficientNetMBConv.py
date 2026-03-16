import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Enable cuDNN heuristics for better convolution algorithm selection
torch.backends.cudnn.benchmark = True


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 2048}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 4096}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 8192}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 16384}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK": 32768}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK": 65536}, num_warps=8, num_stages=4),
    ],
    key=["n_elements"],
)
@triton.jit
def _relu6_inplace_kernel(x_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    tl.multiple_of(offsets, 16)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # clamp to [0, 6] with branchless min/max
    x = tl.maximum(x, 0.0)
    x = tl.minimum(x, 6.0)
    tl.store(x_ptr + offsets, x, mask=mask)


def relu6_inplace_triton(x: torch.Tensor):
    # Fallback for CPU or non-CUDA tensors to maintain correctness
    if not x.is_cuda:
        return torch.clamp_(x, 0, 6)
    n_elements = x.numel()
    if n_elements == 0:
        return x
    # Autotuned BLOCK size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
    _relu6_inplace_kernel[grid](x, n_elements)
    return x


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        MBConv block implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Kernel size for the depthwise convolution.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the intermediate channels.
        """
        super(ModelNew, self).__init__()

        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        # Expand conv -> BN (ReLU6 applied in forward via Triton for better perf)
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(
                in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.expand_bn = nn.BatchNorm2d(hidden_dim)

        # Depthwise conv -> BN (ReLU6 applied in forward via Triton)
        self.depthwise_conv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=hidden_dim,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)

        # Project conv -> BN
        self.project_conv = nn.Conv2d(
            hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.project_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Forward pass of the MBConv block.

        :param x: The input tensor, shape (batch_size, in_channels, H, W)
        :return: The output tensor, shape (batch_size, out_channels, H', W')
        """
        # Prefer channels_last for faster cuDNN kernels on Hopper/H200
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        identity = x

        # Expand path
        if hasattr(self, "expand_conv"):
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            # Triton-accelerated in-place ReLU6 (preserves ReLU6(inplace=True) semantics)
            relu6_inplace_triton(x)

        # Depthwise path
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        relu6_inplace_triton(x)

        # Project path
        x = self.project_conv(x)
        x = self.project_bn(x)

        if self.use_residual:
            x = x + identity

        return x


# Test code
batch_size = 10
in_channels = 112
out_channels = 192
kernel_size = 5
stride = 2
expand_ratio = 6

def get_inputs():
    return [torch.randn(batch_size, in_channels, 224, 224)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, expand_ratio]