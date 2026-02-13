# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.mbconv1 = self._make_mbconv_block(32, 96, 1, 3)
        self.mbconv2 = self._make_mbconv_block(96, 144, 2, 6)
        self.mbconv3 = self._make_mbconv_block(144, 192, 2, 6)
        self.mbconv4 = self._make_mbconv_block(192, 288, 2, 6)
        self.mbconv5 = self._make_mbconv_block(288, 384, 1, 6)
        self.conv_final = nn.Conv2d(384, 1408, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_final = nn.BatchNorm2d(1408)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1408, num_classes)

    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        layers = []
        expanded_channels = in_channels * expand_ratio
        
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(expanded_channels))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False))
        layers.append(nn.BatchNorm2d(expanded_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Replace sequential SE block with Triton kernel call
        layers.append(SELayerTriton(expanded_channels))
        
        layers.append(nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.relu(self.bn_final(self.conv_final(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Triton-optimized Squeeze-and-Excitation Layer
class SELayerTriton(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.reduction = channels // 4
        self.fc1 = nn.Conv2d(channels, self.reduction, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(self.reduction, channels, kernel_size=1, bias=False)

    @triton.jit
    def _se_kernel(
        input_ptr, output_ptr, channels, reduction, 
        input_stride_b, input_stride_c, input_stride_h, input_stride_w,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        batch_idx = pid // channels
        channel_idx = pid % channels
        
        # Compute spatial average
        total = tl.zeros((1,), tl.float32)
        count = 0
        for h in range(BLOCK_SIZE):
            for w in range(BLOCK_SIZE):
                offset = (
                    batch_idx * input_stride_b +
                    channel_idx * input_stride_c +
                    h * input_stride_h +
                    w * input_stride_w
                )
                val = tl.load(input_ptr + offset, mask=(h < BLOCK_SIZE) & (w < BLOCK_SIZE), other=0.0)
                total += val
                count += 1
        avg = total / count
        
        # First linear layer
        temp = tl.zeros((1,), tl.float32)
        for r in range(reduction):
            weight = tl.load(fc1_ptr + channel_idx * reduction + r)
            temp += avg * weight
        temp = tl.maximum(temp, 0)  # ReLU
        
        # Second linear layer
        out_val = tl.zeros((1,), tl.float32)
        for r in range(reduction):
            weight = tl.load(fc2_ptr + r * channels + channel_idx)
            out_val += temp * weight
        out_val = 1.0 / (1.0 + tl.exp(-out_val))  # Sigmoid
        
        # Store result
        output_offset = batch_idx * channels + channel_idx
        tl.store(output_ptr + output_offset, out_val)

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        output = torch.empty(batch_size, channels, device=x.device, dtype=torch.float32)
        
        # Precompute strides
        strides = (x.stride(0), x.stride(1), x.stride(2), x.stride(3))
        
        # Launch Triton kernel
        grid = lambda opt: (batch_size * channels,)
        self._se_kernel[grid](
            x, output, channels, self.reduction, *strides, h, 
            BLOCK_SIZE=h,  # Assuming square input
            fc1_ptr=self.fc1.weight.data_ptr(),
            fc2_ptr=self.fc2.weight.data_ptr()
        )
        
        return output.view(batch_size, channels, 1, 1)

# Test code
batch_size = 2
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]
# =================== EVOLVE-BLOCK-END ===================