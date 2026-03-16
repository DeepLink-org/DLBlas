import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

# Enable cuDNN to pick optimal kernels for fixed shapes
torch.backends.cudnn.benchmark = True


@triton.jit
def _gap2d_reduce_noatomic_kernel(
    x_ptr,                     # *f32 / *f16 / *bf16
    y_ptr,                     # *same dtype as x (output buffer B*C)
    B, C, H, W,                # int32 sizes
    stride_b, stride_c, stride_h, stride_w,  # int64 strides
    BLOCK_M: tl.constexpr,     # tile size along HW
    MAX_TILES: tl.constexpr,   # number of tiles over HW
):
    # One program per (n, c)
    pid = tl.program_id(axis=0)
    n = pid // C
    c = pid % C
    valid_nc = (n < B) & (c < C)

    # Base pointer to (n, c, 0, 0)
    base = x_ptr + n * stride_b + c * stride_c
    M = H * W
    inv_M = 1.0 / (H * W)

    # Accumulator in fp32 for numerical stability
    acc = 0.0
    arange = tl.arange(0, BLOCK_M)

    # Iterate over flattened HW tiles
    for tile_id in tl.static_range(0, MAX_TILES):
        start = tile_id * BLOCK_M
        offs = start + arange
        mask_elems = valid_nc & (offs < M)

        # Map flattened offsets to (h, w) to support any stride layout
        h_idx = offs // W
        w_idx = offs - h_idx * W
        ptrs = base + h_idx * stride_h + w_idx * stride_w

        vals = tl.load(ptrs, mask=mask_elems, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)

    mean = acc * inv_M
    tl.store(y_ptr + (n * C + c), mean, mask=valid_nc)


@triton.jit
def _gap2d_reduce_linear_kernel(
    x_ptr,                     # *f32 / *f16 / *bf16
    y_ptr,                     # *same dtype as x (output buffer B*C)
    B, C, H, W,                # int32 sizes
    stride_b, stride_c, stride_h, stride_w,  # int64 strides (unused in linear path)
    BLOCK_M: tl.constexpr,     # tile size along flattened HW
    MAX_TILES: tl.constexpr,   # number of tiles over flattened HW
):
    # Same PID logic and grid as the fallback kernel
    pid = tl.program_id(axis=0)
    n = pid // C
    c = pid % C
    valid_nc = (n < B) & (c < C)

    base = x_ptr + n * stride_b + c * stride_c
    M = H * W
    inv_M = 1.0 / (H * W)

    acc = 0.0
    arange = tl.arange(0, BLOCK_M)

    # Walk the H*W region linearly; assumes HW is laid out linearly contiguous
    for tile_id in tl.static_range(0, MAX_TILES):
        start = tile_id * BLOCK_M
        offs = start + arange
        mask = valid_nc & (offs < M)
        ptrs = base + offs
        vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)

    mean = acc * inv_M
    tl.store(y_ptr + (n * C + c), mean, mask=valid_nc)


def global_avg_pool2d_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of global average pooling over H and W for NCHW tensors.
    Falls back to PyTorch for non-CUDA tensors or degenerate cases.
    """
    if (not x.is_cuda) or x.numel() == 0 or x.dim() != 4:
        return torch.mean(x, dim=(2, 3))

    B, C, H, W = x.shape
    M = H * W
    if M == 0:
        return torch.zeros((B, C), device=x.device, dtype=x.dtype)

    # Output buffer same dtype as input
    y = torch.empty((B, C), device=x.device, dtype=x.dtype)

    sB, sC, sH, sW = x.stride()
    # Decide if HW is laid out linearly (fast path)
    hw_linear = (sW == 1) and (sH == W) and (sC == H * W)

    grid = (B * C,)

    if hw_linear:
        # Use larger tile for fully contiguous HW
        BLOCK_M = 2048
        num_tiles = (M + BLOCK_M - 1) // BLOCK_M
        # Favor higher warps to maximize memory throughput for contiguous loads
        num_warps = 4 if M <= 4096 else 8
        _gap2d_reduce_linear_kernel[grid](
            x, y,
            B, C, H, W,
            sB, sC, sH, sW,
            BLOCK_M=BLOCK_M,
            MAX_TILES=num_tiles,
            num_warps=num_warps,
            num_stages=2,
        )
    else:
        # Generic path for arbitrary strides (channels_last or others)
        BLOCK_M = 1024
        num_tiles = (M + BLOCK_M - 1) // BLOCK_M
        # More warps help hide non-coalesced latency
        num_warps = 8 if (B * C) >= 512 else 4
        _gap2d_reduce_noatomic_kernel[grid](
            x, y,
            B, C, H, W,
            sB, sC, sH, sW,
            BLOCK_M=BLOCK_M,
            MAX_TILES=num_tiles,
            num_warps=num_warps,
            num_stages=2,
        )
    return y


class ModelNew(nn.Module):
    def __init__(self, input_channels, stages, block_widths, output_classes):
        """
        :param input_channels: int, Number of input channels for the first layer
        :param stages: int, Number of stages in the RegNet architecture
        :param block_widths: List[int], Width (number of channels) for each block in the stages
        :param output_classes: int, Number of output classes for classification
        """
        super(ModelNew, self).__init__()

        self.stages = stages
        self.block_widths = block_widths
        
        layers = []
        current_channels = input_channels
        
        # Construct the stages with their respective blocks
        for i in range(stages):
            layers.append(self._make_stage(current_channels, block_widths[i]))
            current_channels = block_widths[i]
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Final fully connected layer for classification
        self.fc = nn.Linear(block_widths[-1], output_classes)
    
    def _make_stage(self, in_channels, out_channels):
        """
        Creates a simple block for each stage.
        :param in_channels: int, number of input channels
        :param out_channels: int, number of output channels
        :return: nn.Sequential block with convolutional layers
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        Forward pass through the RegNet model.
        :param x: torch.Tensor of shape (batch_size, input_channels, height, width)
        :return: torch.Tensor of shape (batch_size, output_classes)
        """
        # Use channels_last to speed up cuDNN convolutions on modern GPUs
        if x.is_cuda:
            x = x.to(memory_format=torch.channels_last)
        x = self.feature_extractor(x)
        # Global Average Pooling (Triton-accelerated when on CUDA)
        if x.is_cuda:
            x = global_avg_pool2d_triton(x)
        else:
            x = torch.mean(x, dim=[2, 3])
        x = self.fc(x)
        return x

# Test code for the RegNet model
batch_size = 8
input_channels = 3
image_height, image_width = 224, 224
stages = 3
block_widths = [64, 128, 256]
output_classes = 10

def get_inputs():
    """ Generates random input tensor of shape (batch_size, input_channels, height, width) """
    return [torch.randn(batch_size, input_channels, image_height, image_width)]

def get_init_inputs():
    """ Initializes model parameters """
    return [input_channels, stages, block_widths, output_classes]