import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

# Enable cuDNN autotuner for potentially faster convolutions on fixed-size inputs
torch.backends.cudnn.benchmark = True


@triton.jit
def _relu_inplace_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    x = tl.maximum(x, 0)
    tl.store(x_ptr + offs, x, mask=mask)


def relu_inplace_triton(x: torch.Tensor):
    # Fast in-place ReLU using Triton. Fallback for non-CUDA or non-contiguous tensors.
    if (not x.is_cuda) or (not (x.is_contiguous() or x.is_contiguous(memory_format=torch.channels_last))) or x.numel() == 0:
        return F.relu(x, inplace=True)
    n_elems = x.numel()
    # Larger block and more warps to better utilize H200 bandwidth
    BLOCK = 16384
    grid = lambda meta: (triton.cdiv(n_elems, meta["BLOCK_SIZE"]),)
    _relu_inplace_kernel[grid](x, n_elems, BLOCK_SIZE=BLOCK, num_warps=8, num_stages=2)
    return x


@triton.jit
def _relu_mean_hw_singlepass_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    y_stride_n, y_stride_c,
    inv_hw,
    BLOCK_HW: tl.constexpr,
):
    # One Triton program per (n, c); iterate across spatial positions and
    # accumulate mean(ReLU(x[n, c, :, :])) in fp32 without writing back to x.
    pid = tl.program_id(axis=0)
    n = pid // C
    c = pid % C

    HW = H * W
    offs0 = tl.arange(0, BLOCK_HW)

    # Base pointer for this (n, c)
    base = x_ptr + n * stride_n + c * stride_c

    # Accumulation vector; we avoid scalar indexing by using a vector accumulator
    acc_vec = tl.zeros([BLOCK_HW], dtype=tl.float32)

    start = 0
    while start < HW:
        offs = start + offs0
        mask = offs < HW
        h = offs // W
        w = offs % W
        ptrs = base + h * stride_h + w * stride_w
        vals = tl.load(ptrs, mask=mask, other=0)
        relu_vals = tl.maximum(vals, 0).to(tl.float32)
        part = tl.sum(relu_vals, axis=0)  # scalar
        acc_vec += part  # broadcast add
        start += BLOCK_HW

    # Reduce vector accumulator to scalar without indexing
    total = tl.sum(acc_vec, axis=0) * (1.0 / BLOCK_HW)
    out_ptr = y_ptr + n * y_stride_n + c * y_stride_c
    tl.store(out_ptr, total * inv_hw)


def relu_mean_hw_triton(x: torch.Tensor) -> torch.Tensor:
    # Fused ReLU + global average pooling over H, W to produce (N, C) without
    # writing back to x. Falls back to PyTorch when unsuitable.
    if (not x.is_cuda) or (not x.is_contiguous()):
        # Preserve original semantics
        x = F.relu(x, inplace=True)
        return x.mean(dim=(2, 3))

    N, C, H, W = x.shape
    if N == 0 or C == 0 or H == 0 or W == 0:
        return x.new_empty((N, C))

    y = torch.empty((N, C), device=x.device, dtype=x.dtype)

    stride_n, stride_c, stride_h, stride_w = x.stride()
    y_stride_n, y_stride_c = y.stride()

    inv_hw = 1.0 / float(H * W)

    # Use a relatively large tile to cover typical final spatial grids (e.g., 7x7) in one iteration.
    BLOCK_HW = 256
    grid = (N * C,)

    _relu_mean_hw_singlepass_kernel[grid](
        x, y,
        N, C, H, W,
        stride_n, stride_c, stride_h, stride_w,
        y_stride_n, y_stride_c,
        inv_hw,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
        num_stages=2,
    )
    return y


class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        :param num_layers: The number of layers in the dense block
        :param num_input_features: The number of input feature maps
        :param growth_rate: The growth rate for the dense block (new features added per layer)
        """
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        """
        Creates a single layer with BatchNorm, ReLU, Conv2D, and Dropout.
        """
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Concatenated output tensor with shape (batch_size, num_output_features, height, width)
        """
        # Efficient dense connectivity: x already contains all previous features
        for layer in self.layers:
            new_feature = layer(x)
            x = torch.cat((x, new_feature), 1)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Downsampled tensor with reduced number of feature maps
        """
        return self.transition(x)


class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        """
        :param growth_rate: The growth rate of the DenseNet (new features added per layer)
        :param num_classes: The number of output classes for classification
        """
        super(ModelNew, self).__init__()

        # Initial convolution and pooling
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Each dense block is followed by a transition layer, except the last one
        num_features = 64
        block_layers = [6, 12, 24, 16]  # Corresponding layers in DenseNet121

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        # Final batch norm and classifier
        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape (batch_size, 3, height, width)
        :return: Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.final_bn(x)
        # Fused Triton kernel: ReLU + global average pooling over spatial dims to (N, C)
        x = relu_mean_hw_triton(x)
        x = self.classifier(x)
        return x


# Testing the DenseNet121 model
batch_size = 10
num_classes = 10
height, width = 224, 224  # Standard input size for DenseNet

def get_inputs():
    return [torch.randn(batch_size, 3, height, width)]

def get_init_inputs():
    return [32, num_classes]