import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def _relu_maxpool2x2_stride2_nchw_kernel(
    x_ptr,  # *float*
    y_ptr,  # *float*
    N, C, H, W, H_OUT, W_OUT,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * H_OUT * W_OUT
    mask = offs < total

    # Compute index mapping with fewer divisions/mods
    PLANE = H_OUT * W_OUT
    t1 = offs // W_OUT
    wo = offs - t1 * W_OUT
    nc = t1 // H_OUT
    ho = t1 - nc * H_OUT

    # Top-left corner for the 2x2 window (stride=2)
    hi = ho + ho
    wi = wo + wo

    # Flat base offsets for input/output assuming contiguous NCHW
    x_off_base = nc * (H * W) + hi * W + wi
    y_off = offs  # direct linear index into contiguous output

    # Load 2x2 window with robust 'other' to be safe under any mask scenario
    v00 = tl.load(x_ptr + (x_off_base + 0), mask=mask, other=-float("inf"))
    v01 = tl.load(x_ptr + (x_off_base + 1), mask=mask, other=-float("inf"))
    v10 = tl.load(x_ptr + (x_off_base + W + 0), mask=mask, other=-float("inf"))
    v11 = tl.load(x_ptr + (x_off_base + W + 1), mask=mask, other=-float("inf"))

    # MaxPool over 2x2 then ReLU
    m0 = tl.maximum(v00, v01)
    m1 = tl.maximum(v10, v11)
    out = tl.maximum(tl.maximum(m0, m1), 0.0)

    tl.store(y_ptr + y_off, out, mask=mask)


def relu_maxpool2x2_stride2(x: torch.Tensor) -> torch.Tensor:
    """
    Fused ReLU + MaxPool2d (kernel=2, stride=2) for NCHW tensors.
    Falls back to PyTorch path on CPU or small tensors where launch overhead dominates.
    """
    if not x.is_cuda:
        return F.max_pool2d(F.relu(x), kernel_size=2, stride=2)

    x = x.contiguous()
    N, C, H, W = x.shape
    H_OUT = H // 2
    W_OUT = W // 2
    total = N * C * H_OUT * W_OUT

    # For small tensors, the native path is faster due to lower launch overhead.
    # Use a higher threshold to avoid Triton launch overhead on tiny problems.
    if total <= 2048:
        return F.max_pool2d(F.relu(x), kernel_size=2, stride=2)

    y = torch.empty((N, C, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

    def grid(meta):
        blk = meta["BLOCK"]
        return ((total + blk - 1) // blk,)

    _relu_maxpool2x2_stride2_nchw_kernel[grid](
        x, y,
        N, C, H, W, H_OUT, W_OUT,
        BLOCK=256,
        num_warps=1,
        num_stages=1,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, num_classes):
        """
        LeNet-5 architecture implementation in PyTorch.

        :param num_classes: The number of output classes.
        """
        super(ModelNew, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
    
    def forward(self, x):
        """
        Forward pass of the LeNet-5 model.

        :param x: The input tensor, shape (batch_size, 1, 32, 32)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # First convolution followed by fused ReLU+MaxPool2d
        x = self.conv1(x)
        x = relu_maxpool2x2_stride2(x)
        
        # Second convolution followed by fused ReLU+MaxPool2d
        x = self.conv2(x)
        x = relu_maxpool2x2_stride2(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 16*5*5)
        
        # First fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        
        # Second fully connected layer with ReLU activation
        x = F.relu(self.fc2(x))
        
        # Final fully connected layer
        x = self.fc3(x)
        
        return x

# Test code for the LeNet-5 model
batch_size = 1
num_classes = 10

def get_inputs():
    return [torch.randn(batch_size, 1, 32, 32)]

def get_init_inputs():
    return [num_classes]