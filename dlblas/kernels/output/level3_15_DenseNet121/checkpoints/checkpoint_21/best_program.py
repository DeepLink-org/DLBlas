# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _adaptive_avg_pool2d_kernel(
    x_ptr,
    output_ptr,
    stride_b,
    stride_c,
    height,
    width,
    output_stride_b,
    output_stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    spatial_size = height * width
    
    base_offset = pid_batch * stride_b + pid_channel * stride_c
    offsets = base_offset + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < spatial_size
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    total = tl.sum(values)
    avg = total / spatial_size
    
    output_index = pid_batch * output_stride_b + pid_channel * output_stride_c
    tl.store(output_ptr + output_index, avg)

def adaptive_avg_pool2d_1x1(x):
    batch, channels, height, width = x.shape
    output = torch.empty((batch, channels), device=x.device, dtype=x.dtype)
    block_size = triton.next_power_of_2(height * width)
    
    grid = (batch, channels)
    _adaptive_avg_pool2d_kernel[grid](
        x, output,
        x.stride(0), x.stride(1),
        height, width,
        output.stride(0), output.stride(1),
        BLOCK_SIZE=block_size,
        num_warps=4
    )
    return output

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
    
    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)

class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)

class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelNew, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_features = 64
        block_layers = [6, 12, 24, 16]

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers, num_features, growth_rate)
            self.dense_blocks.append(block)
            num_features += num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_features, num_features // 2)
                self.transition_layers.append(transition)
                num_features //= 2

        self.final_norm = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)
        
        x = F.relu(self.final_norm(x), inplace=True)
        # Use Triton kernel for efficient pooling
        x = adaptive_avg_pool2d_1x1(x)
        return self.classifier(x)

# Testing the DenseNet121 model
batch_size = 10
num_classes = 10
height, width = 224, 224

def get_inputs():
    return [torch.randn(batch_size, 3, height, width)]

def get_init_inputs():
    return [32, num_classes]

# =================== EVOLVE-BLOCK-END ===================