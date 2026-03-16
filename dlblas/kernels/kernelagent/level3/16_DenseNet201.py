import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: Triton kernel for fast fused ReLU + global average pooling (NCHW -> NC)
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


@triton.jit
def _relu_gavgpool2d_nchw_kernel(
    x_ptr,  # *f32
    y_ptr,  # *f32
    N, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    y_stride_n, y_stride_c,
    BLOCK_W: tl.constexpr,
    UNROLL_W2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    total_nc = N * C
    if pid >= total_nc:
        return

    n = pid // C
    c = pid % C

    base = x_ptr + n * stride_n + c * stride_c

    offs_w = tl.arange(0, BLOCK_W)
    acc_vec = tl.zeros((BLOCK_W,), dtype=tl.float32)

    for h in range(0, H):
        row_ptr = base + h * stride_h
        for w0 in range(0, W, BLOCK_W * UNROLL_W2):
            # chunk 0
            idx0 = w0 + offs_w
            mask0 = idx0 < W
            ptrs0 = row_ptr + idx0 * stride_w
            vals0 = tl.load(ptrs0, mask=mask0, other=0.0)
            vals0 = tl.maximum(vals0, 0)  # ReLU
            acc_vec += vals0.to(tl.float32)

            if UNROLL_W2 > 1:
                idx1 = idx0 + BLOCK_W
                mask1 = idx1 < W
                ptrs1 = row_ptr + idx1 * stride_w
                vals1 = tl.load(ptrs1, mask=mask1, other=0.0)
                vals1 = tl.maximum(vals1, 0)  # ReLU
                acc_vec += vals1.to(tl.float32)

    total = tl.sum(acc_vec, axis=0)
    denom = 1.0 / (H * W)
    avg = total * denom

    out_ptr = y_ptr + n * y_stride_n + c * y_stride_c
    tl.store(out_ptr, avg.to(tl.float32))


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def relu_global_avg_pool2d_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Fused ReLU + Global Average Pooling over HxW.
    Input:  x of shape (N, C, H, W)
    Output: y of shape (N, C) equal to:
            adaptive_avg_pool2d(relu(x), (1,1)).view(N, C)
    Falls back to PyTorch for non-CUDA tensors.
    """
    if (not _TRITON_AVAILABLE) or (not x.is_cuda) or (x.numel() == 0):
        x_relu = F.relu(x, inplace=False)
        return F.adaptive_avg_pool2d(x_relu, (1, 1)).view(x.size(0), -1)

    N, C, H, W = x.shape
    y = torch.empty((N, C), device=x.device, dtype=x.dtype)

    sN, sC, sH, sW = x.stride()
    y_sN, y_sC = y.stride()

    # Tile selection along W for coalesced loads and good occupancy.
    # Use small tiles for small W (e.g., 7) to avoid wasted lanes.
    def next_pow2(v: int) -> int:
        return 1 << (max(v, 1) - 1).bit_length()

    if W >= 512:
        block_w = 128
        unroll = 2
        num_warps = 4
    elif W >= 128:
        block_w = 64
        unroll = 2
        num_warps = 2
    elif W >= 64:
        block_w = 64
        unroll = 1
        num_warps = 2
    else:
        block_w = min(32, next_pow2(W))
        unroll = 1
        num_warps = 1

    grid = (N * C,)
    _relu_gavgpool2d_nchw_kernel[grid](
        x, y,
        N, C, H, W,
        sN, sC, sH, sW,
        y_sN, y_sC,
        BLOCK_W=block_w,
        UNROLL_W2=unroll,
        num_warps=num_warps,
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
        # Efficient iterative concatenation, semantically equivalent to the original implementation.
        for layer in self.layers:
            new_feature = layer(x)
            x = torch.cat((x, new_feature), dim=1)
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
        block_layers = [6, 12, 48, 32]  # Corresponding layers in DenseNet201

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
        # Fused ReLU + global average pooling via Triton for improved performance.
        x = relu_global_avg_pool2d_triton(x)
        x = self.classifier(x)
        return x


# Testing the DenseNet201 model
batch_size = 10
num_classes = 10
height, width = 224, 224  # Standard input size for DenseNet

def get_inputs():
    return [torch.randn(batch_size, 3, height, width)]

def get_init_inputs():
    return [32, num_classes]