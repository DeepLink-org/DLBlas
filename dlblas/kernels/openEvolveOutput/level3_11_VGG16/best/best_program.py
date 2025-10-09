# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_features, out_features, batch_size,
    stride_x, stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_batch = pid // out_features
    pid_feature = pid % out_features

    if pid_batch >= batch_size or pid_feature >= out_features:
        return

    output_offset = pid_batch * stride_out + pid_feature
    bias_val = tl.load(bias_ptr + pid_feature)
    acc = bias_val

    for block in range(0, in_features, BLOCK_SIZE):
        offsets = block + tl.arange(0, BLOCK_SIZE)
        mask = offsets < in_features
        
        input_offset = pid_batch * stride_x + offsets
        weight_offset = pid_feature * in_features + offsets
        
        x = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
        w = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
        
        acc += tl.sum(x * w)

    tl.store(output_ptr + output_offset, acc)

class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        output = torch.empty(x.size(0), self.out_features, device=x.device, dtype=x.dtype)
        flat_x = x.view(-1, x.size(-1))
        batch_size, in_features = flat_x.shape

        grid = (batch_size * self.out_features,)
        BLOCK_SIZE = 128
        
        linear_kernel[grid](
            flat_x, self.weight, self.bias, output,
            in_features, self.out_features, batch_size,
            flat_x.stride(0), output.stride(0),
            BLOCK_SIZE=BLOCK_SIZE
        )
        return output

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            TritonLinear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Test code
import math
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]
# =================== EVOLVE-BLOCK-END ===================