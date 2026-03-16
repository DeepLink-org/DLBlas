import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def _relu_maxpool3x3s2_nchw_kernel(
    x_ptr,  # *T
    y_ptr,  # *T
    N, C, H, W, OH, OW, NUMEL,  # int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < NUMEL

    # Decode linear index into (n, c, oh, ow)
    ow = offs % OW
    t1 = offs // OW
    oh = t1 % OH
    t2 = t1 // OH
    c = t2 % C
    n = t2 // C

    # Stride 2, kernel 3x3, no padding
    ih = oh * 2
    iw = ow * 2

    # Flattened base index in NCHW
    base = (((n * C) + c) * H + ih) * W + iw

    # Load 3x3 window; masked loads for inactive lanes
    v00 = tl.load(x_ptr + base + 0 * W + 0, mask=mask, other=-float("inf"))
    v01 = tl.load(x_ptr + base + 0 * W + 1, mask=mask, other=-float("inf"))
    v02 = tl.load(x_ptr + base + 0 * W + 2, mask=mask, other=-float("inf"))
    v10 = tl.load(x_ptr + base + 1 * W + 0, mask=mask, other=-float("inf"))
    v11 = tl.load(x_ptr + base + 1 * W + 1, mask=mask, other=-float("inf"))
    v12 = tl.load(x_ptr + base + 1 * W + 2, mask=mask, other=-float("inf"))
    v20 = tl.load(x_ptr + base + 2 * W + 0, mask=mask, other=-float("inf"))
    v21 = tl.load(x_ptr + base + 2 * W + 1, mask=mask, other=-float("inf"))
    v22 = tl.load(x_ptr + base + 2 * W + 2, mask=mask, other=-float("inf"))

    # Max over the 3x3 window
    m0 = tl.maximum(tl.maximum(v00, v01), v02)
    m1 = tl.maximum(tl.maximum(v10, v11), v12)
    m2 = tl.maximum(tl.maximum(v20, v21), v22)
    m = tl.maximum(tl.maximum(m0, m1), m2)

    # Apply ReLU after max-pool (equivalent to maxpool(ReLU(x)))
    m = tl.maximum(m, 0.0)

    # Store
    tl.store(y_ptr + offs, m, mask=mask)


def relu_maxpool3x3s2_triton(x: torch.Tensor) -> torch.Tensor:
    # Fused ReLU + MaxPool2d(kernel=3, stride=2) for NCHW tensors
    if not x.is_cuda:
        # CPU fallback preserves semantics
        return F.max_pool2d(F.relu(x), kernel_size=3, stride=2)

    assert x.dim() == 4, "Expected 4D NCHW tensor"
    N, C, H, W = x.shape
    # Output dims as in PyTorch (ceil_mode=False, padding=0, dilation=1)
    OH = (H - 3) // 2 + 1
    OW = (W - 3) // 2 + 1
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)

    numel = N * C * OH * OW
    if numel == 0:
        return y

    BLOCK_SIZE = 256
    grid = lambda META: (triton.cdiv(numel, META["BLOCK_SIZE"]),)

    # Ensure contiguous memory in NCHW layout
    x_c = x.contiguous()
    y_c = y

    _relu_maxpool3x3s2_nchw_kernel[grid](
        x_c, y_c,
        N, C, H, W, OH, OW, numel,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y_c


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: The number of output classes (default is 1000 for ImageNet)
        """
        super(ModelNew, self).__init__()

        # Enable high-performance backends where safe
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        # Fifth convolutional layer
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.0)

        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.0)

        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # Conv1 -> ReLU -> MaxPool fused
        x = self.conv1(x)
        x = relu_maxpool3x3s2_triton(x)

        # Conv2 -> ReLU -> MaxPool fused
        x = self.conv2(x)
        x = relu_maxpool3x3s2_triton(x)

        # Conv3 -> ReLU
        x = self.conv3(x)
        x = self.relu3(x)

        # Conv4 -> ReLU
        x = self.conv4(x)
        x = self.relu4(x)

        # Conv5 -> ReLU -> MaxPool fused
        x = self.conv5(x)
        x = relu_maxpool3x3s2_triton(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu7(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x


# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]