# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def fused_conv_bn_relu_1x1(
    x_ptr, 
    w_ptr, 
    bn_weight_ptr, 
    bn_bias_ptr, 
    bn_mean_ptr, 
    bn_var_ptr, 
    output_ptr,
    eps: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Get program IDs
    pid_batch = tl.program_id(axis=0)
    pid_hw = tl.program_id(axis=1)
    pid_c = tl.program_id(axis=2)
    
    # Create masks
    c_offsets = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < out_channels
    hw_mask = tl.arange(0, BLOCK_SIZE_HW) < BLOCK_SIZE_HW
    
    # Calculate input offsets
    x_offset = pid_batch * in_channels * BLOCK_SIZE_HW + pid_hw * BLOCK_SIZE_HW
    w_offset = c_offsets[:, None] * in_channels + tl.arange(0, in_channels)[None, :]
    
    # Load weights and input
    w = tl.load(w_ptr + w_offset, mask=c_mask[:, None] & (tl.arange(0, in_channels)[None, :] < in_channels), other=0.0)
    x = tl.load(x_ptr + x_offset + tl.arange(0, in_channels), mask=hw_mask[None, :] & (tl.arange(0, in_channels)[:, None] < in_channels), other=0.0)
    
    # Compute convolution
    conv_out = tl.dot(w, x)
    
    # Load BN parameters
    bn_weight = tl.load(bn_weight_ptr + c_offsets, mask=c_mask, other=0.0)
    bn_bias = tl.load(bn_bias_ptr + c_offsets, mask=c_mask, other=0.0)
    bn_mean = tl.load(bn_mean_ptr + c_offsets, mask=c_mask, other=0.0)
    bn_var = tl.load(bn_var_ptr + c_offsets, mask=c_mask, other=0.0)
    
    # Apply BN
    inv_std = 1.0 / tl.sqrt(bn_var + eps)
    bn_out = (conv_out - bn_mean[:, None]) * inv_std[:, None] * bn_weight[:, None] + bn_bias[:, None]
    
    # Apply ReLU
    relu_out = tl.where(bn_out > 0, bn_out, 0.0)
    
    # Store output
    output_offset = pid_batch * out_channels * BLOCK_SIZE_HW + c_offsets[:, None] * BLOCK_SIZE_HW + pid_hw * BLOCK_SIZE_HW
    tl.store(output_ptr + output_offset, relu_out, mask=c_mask[:, None] & hw_mask[None, :])


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        
        # Fused conv1+bn1+relu with Triton
        B, C, H, W = x.shape
        out = torch.empty(B, self.out_channels, H, W, device=x.device, dtype=x.dtype)
        
        # Calculate grid dimensions
        grid_batch = B
        grid_hw = (H * W + 15) // 16
        grid_c = (self.out_channels + 31) // 32
        
        # Launch Triton kernel
        fused_conv_bn_relu_1x1[ (grid_batch, grid_hw, grid_c) ](
            x,
            self.conv1.weight,
            self.bn1.weight,
            self.bn1.bias,
            self.bn1.running_mean,
            self.bn1.running_var,
            out,
            self.bn1.eps,
            self.in_channels,
            self.out_channels,
            32,  # BLOCK_SIZE_C
            16,  # BLOCK_SIZE_HW
        )
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ModelNew(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = Bottleneck

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        x = self.fc(x)

        return x

# Test code
batch_size = 10
height = 224
width = 224
layers = [3, 4, 23, 3]
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, height, width)]

def get_init_inputs():
    return [layers, num_classes]
# =================== EVOLVE-BLOCK-END ===================