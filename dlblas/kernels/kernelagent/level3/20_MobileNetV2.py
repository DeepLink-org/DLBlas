import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _row_mean_kernel(
    x_ptr,        # *ptr to input flattened [NC, S]
    y_ptr,        # *ptr to output [NC] (fp32)
    NC,           # total rows = N*C
    S,            # spatial size = H*W
    BLOCK: tl.constexpr,
):
    # One program per row
    pid = tl.program_id(0)
    in_bounds = pid < NC
    base = x_ptr + pid * S

    acc = tl.zeros((), dtype=tl.float32)
    offs = tl.arange(0, BLOCK)

    # Loop over the row with simple software unrolling by 2 to reduce loop overhead
    i = 0
    while i < S:
        idx0 = i + offs
        m0 = (idx0 < S) & in_bounds
        v0 = tl.load(base + idx0, mask=m0, other=0.0)
        acc += tl.sum(v0.to(tl.float32), axis=0)

        idx1 = idx0 + BLOCK
        m1 = (idx1 < S) & in_bounds
        v1 = tl.load(base + idx1, mask=m1, other=0.0)
        acc += tl.sum(v1.to(tl.float32), axis=0)

        i += 2 * BLOCK

    invS = 1.0 / tl.full((), S, dtype=tl.float32)
    mean_val = acc * invS
    tl.store(y_ptr + pid, mean_val, mask=in_bounds)


def global_avg_pool2d_triton(x: torch.Tensor) -> torch.Tensor:
    # x: [N, C, H, W] on CUDA
    if (not x.is_cuda) or x.numel() == 0:
        return x.mean(dim=(2, 3), keepdim=False)

    N, C, H, W = x.shape
    S = H * W
    # Ensure contiguous NCHW then flatten spatial dims for coalesced loads
    x2d = x.contiguous().view(N * C, S)
    y = torch.empty((N * C,), device=x.device, dtype=torch.float32)

    # Heuristic tiling tuned for small spatial footprints like 7x7 common in MobileNetV2
    if S <= 64:
        BLOCK, WARPS, STAGES = 64, 2, 2
    elif S <= 128:
        BLOCK, WARPS, STAGES = 128, 4, 2
    elif S <= 256:
        BLOCK, WARPS, STAGES = 256, 4, 3
    else:
        BLOCK, WARPS, STAGES = 512, 8, 3

    grid = (N * C,)
    _row_mean_kernel[grid](
        x2d, y,
        N * C, S,
        BLOCK=BLOCK,
        num_warps=WARPS,
        num_stages=STAGES,
    )
    y = y.view(N, C)
    if x.dtype != torch.float32:
        y = y.to(x.dtype)
    return y


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        MobileNetV2 architecture implementation in PyTorch.

        :param num_classes: The number of output classes. Default is 1000.
        """
        super(ModelNew, self).__init__()
        
        def _make_divisible(v, divisor, min_value=None):
            """
            This function ensures that the number of channels is divisible by the divisor.
            """
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _inverted_residual_block(inp, oup, stride, expand_ratio):
            """
            Inverted Residual Block for MobileNetV2.
            """
            hidden_dim = int(inp * expand_ratio)
            use_res_connect = stride == 1 and inp == oup

            layers = []
            if expand_ratio != 1:
                # Pointwise convolution
                layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
                layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.ReLU6(inplace=True))

            layers.extend([
                # Depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Pointwise linear convolution
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])

            if use_res_connect:
                return nn.Sequential(*layers), True
            else:
                return nn.Sequential(*layers), False

        # MobileNetV2 architecture
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Building first layer
        features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(input_channel),
                    nn.ReLU6(inplace=True)]

        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(_inverted_residual_block(input_channel, output_channel, stride, expand_ratio=t)[0])
                input_channel = output_channel

        # Building last several layers (no pooling here; we'll use a Triton kernel in forward)
        features.append(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False))
        features.append(nn.BatchNorm2d(last_channel))
        features.append(nn.ReLU6(inplace=True))

        self.features = nn.Sequential(*features)

        # Linear layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass of the MobileNetV2 model.

        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.features(x)               # [N, C, H, W]
        # Triton-accelerated global average pooling -> [N, C]
        if x.is_cuda:
            x = global_avg_pool2d_triton(x)
        else:
            x = x.mean(dim=(2, 3))
        x = x.view(x.size(0), -1)          # [N, C]
        x = self.classifier(x)
        return x

# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]