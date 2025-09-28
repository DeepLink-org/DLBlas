# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Triton-optimized 3x3 convolution kernel with fused ReLU
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8),
    ],
    key=['C_in'],
)
@triton.jit
def conv2d_3x3_kernel(
    # Pointers to matrices
    input_ptr, weight_ptr, output_ptr,
    # Matrix dimensions
    B, C_in, H, W, C_out,
    # Strides
    stride_b, stride_cin, stride_h, stride_w,
    stride_wb, stride_wcin, stride_wh, stride_ww,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Multiple programs in parallel
    pid = tl.program_id(0)
    pid_b = pid // (H * W)
    pid_hw = pid % (H * W)
    pid_h = pid_hw // W
    pid_w = pid_hw % W
    
    # Create pointers to input and weights
    input_offs = pid_b * stride_b + pid_h * stride_h + pid_w * stride_w
    weight_offs = tl.arange(0, C_out)[:, None] * stride_wb
    
    # Block pointers with boundary checks
    input_ptrs = input_ptr + input_offs + tl.arange(0, C_in)[None, :] * stride_cin
    weight_ptrs = weight_ptr + weight_offs + tl.arange(0, C_in)[:, None] * stride_wcin
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over kernel positions
    for kh in range(3):
        for kw in range(3):
            # Calculate input position with padding=1
            h = pid_h + kh - 1
            w = pid_w + kw - 1
            
            # Check boundaries
            within_bounds = (h >= 0) & (h < H) & (w >= 0) & (w < W)
            
            # Load input and weight tiles
            input_val = tl.where(within_bounds, 
                                tl.load(input_ptrs + kh*stride_h + kw*stride_w, 
                                        mask=within_bounds, other=0.0), 
                                0.0)
            weight_val = tl.load(weight_ptrs + kh*stride_wh + kw*stride_ww)
            
            # Compute convolution
            acc += tl.dot(input_val, weight_val)
    
    # Fused ReLU activation
    acc = tl.maximum(acc, 0.0)
    
    # Write back result
    output_ptrs = output_ptr + pid_b * B * C_out + tl.arange(0, C_out)[:, None]
    tl.store(output_ptrs, acc, mask=within_bounds)

class TritonConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight)
        
    def forward(self, x):
        B, C_in, H, W = x.shape
        output = torch.empty(B, self.out_channels, H, W, device=x.device, dtype=x.dtype)
        
        grid = lambda opt: (B * H * W,)
        conv2d_3x3_kernel[grid](
            x, self.weight, output,
            B, C_in, H, W, self.out_channels,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.weight.stride(0), self.weight.stride(1), self.weight.stride(2), self.weight.stride(3),
            BLOCK_M=triton.next_power_of_2(C_in),
            BLOCK_N=triton.next_power_of_2(self.out_channels)
        )
        return output

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            TritonConv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)
        return x

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
        block_layers = [6, 12, 48, 32]

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

        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)
        x = self.final_bn(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Testing the DenseNet201 model
batch_size = 10
num_classes = 10
height, width = 224, 224

def get_inputs():
    return [torch.randn(batch_size, 3, height, width)]

def get_init_inputs():
    return [32, num_classes]

# =================== EVOLVE-BLOCK-END ===================