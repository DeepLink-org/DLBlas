import torch
import torch.nn as nn
import torch.nn.functional as F

# Triton-based fast Linear (GEMM) with fused bias
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.autotune(
    configs=[
        # Small-M focused configs
        triton.Config({'BLOCK_M': 1,   'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 2,   'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 4,   'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 8,   'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 1,   'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 2,   'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        # General-purpose configs
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_bias_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # B is expected to be [K, N] (weight.t().contiguous())
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Provide codegen hints
    tl.max_contiguous(offs_m, BLOCK_M)
    tl.max_contiguous(offs_n, BLOCK_N)
    tl.multiple_of(a_ptrs, 16)
    tl.multiple_of(b_ptrs, 16)

    # Double-buffered K loop
    k = 0
    k_mask = (k + offs_k) < K
    a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
    b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
    a_ptrs_next = a_ptrs + BLOCK_K * stride_ak
    b_ptrs_next = b_ptrs + BLOCK_K * stride_bk
    k += BLOCK_K

    while k < K:
        acc += tl.dot(a, b)
        k_mask = (k + offs_k) < K
        a = tl.load(a_ptrs_next, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs_next, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        a_ptrs_next += BLOCK_K * stride_ak
        b_ptrs_next += BLOCK_K * stride_bk
        k += BLOCK_K

    acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _matvec_bias_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program computes one row m, tiled across N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    n_mask = offs_n < N

    # Hints
    tl.max_contiguous(offs_n, BLOCK_N)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Base pointers for this row
    a_ptrs = A_ptr + pid_m * stride_am + offs_k * stride_ak
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Pipeline over K
    k = 0
    k_mask = (k + offs_k) < K
    a = tl.load(a_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K]
    b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)  # [BLOCK_K, BLOCK_N]
    a_ptrs_next = a_ptrs + BLOCK_K * stride_ak
    b_ptrs_next = b_ptrs + BLOCK_K * stride_bk
    k += BLOCK_K

    while k < K:
        acc += tl.sum(a[:, None] * b, axis=0)
        k_mask = (k + offs_k) < K
        a = tl.load(a_ptrs_next, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs_next, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        a_ptrs_next += BLOCK_K * stride_ak
        b_ptrs_next += BLOCK_K * stride_bk
        k += BLOCK_K

    acc += tl.sum(a[:, None] * b, axis=0)

    bias = tl.load(Bias_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias

    c_ptrs = C_ptr + pid_m * stride_cm + offs_n * stride_cn
    tl.store(c_ptrs, acc, mask=n_mask)


def _linear_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # Fallback if Triton unavailable or tensors not on CUDA
    if (not TRITON_AVAILABLE) or (not x.is_cuda):
        return F.linear(x, weight, bias)

    # Ensure dtypes and contiguity; compute in fp32 for numerical parity
    A = x.contiguous().to(torch.float32)  # [M, K]
    # Use W^T contiguous to improve B memory coalescing: [K, N]
    Wt = weight.t().contiguous().to(torch.float32)  # [K, N]
    BIAS = (bias if bias is not None else torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)).contiguous().to(torch.float32)

    M, K = A.shape
    N = Wt.shape[1]
    assert Wt.shape[0] == K, f"Incompatible shapes for Linear: weight(*,{Wt.shape[0]}) vs input(*,{K})"

    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # Strides in elements
    stride_am, stride_ak = A.stride()            # typically (K, 1)
    stride_bk, stride_bn = Wt.stride()           # (N, 1) for contiguous [K, N]
    stride_cm, stride_cn = C.stride()

    # Very small batch (M) benefits from GEMV-style kernel
    if M <= 8:
        BLOCK_N = 128
        BLOCK_K = 64
        grid = (M, triton.cdiv(N, BLOCK_N))
        _matvec_bias_kernel[grid](
            A, Wt, BIAS, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )
    else:
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        _matmul_bias_kernel[grid](
            A, Wt, BIAS, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
        )
    return C.to(x.dtype)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride for the first convolutional layer
        :param downsample: Downsample layer for the shortcut connection
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: Number of output classes
        """
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # Use Triton-accelerated Linear on CUDA for speed; fall back otherwise
        if x.is_cuda and TRITON_AVAILABLE:
            x = _linear_triton(x, self.fc.weight, self.fc.bias)
        else:
            x = self.fc(x)

        return x


# Test code
batch_size = 2
num_classes = 1000
input_shape = (batch_size, 3, 224, 224)

def get_inputs():
    return [torch.randn(input_shape)]

def get_init_inputs():
    return [num_classes]