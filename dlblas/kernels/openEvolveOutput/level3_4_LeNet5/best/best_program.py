# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_batch, input_channels, input_height, input_width,
    output_channels, stride,
    output_height, output_width,
    BLOCK_SIZE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    hw = pid_hw
    h = hw // output_width
    w = hw % output_width
    
    accumulator = 0.0
    for c in range(input_channels):
        for kh in range(KERNEL_SIZE):
            for kw in range(KERNEL_SIZE):
                h_in = h * stride + kh
                w_in = w * stride + kw
                
                if h_in < input_height and w_in < input_width:
                    input_offset = pid_batch * (input_channels * input_height * input_width) + \
                                 c * (input_height * input_width) + \
                                 h_in * input_width + w_in
                    input_val = tl.load(input_ptr + input_offset)
                    
                    weight_offset = pid_channel * (input_channels * KERNEL_SIZE * KERNEL_SIZE) + \
                                  c * (KERNEL_SIZE * KERNEL_SIZE) + \
                                  kh * KERNEL_SIZE + kw
                    weight_val = tl.load(weight_ptr + weight_offset)
                    
                    accumulator += input_val * weight_val
    
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + pid_channel)
        accumulator += bias_val
    
    output_offset = pid_batch * (output_channels * output_height * output_width) + \
                   pid_channel * (output_height * output_width) + \
                   hw
    tl.store(output_ptr + output_offset, accumulator)

class TritonConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        batch_size, _, input_height, input_width = x.shape
        output_height = (input_height - self.kernel_size) // self.stride + 1
        output_width = (input_width - self.kernel_size) // self.stride + 1
        
        x = x.contiguous()
        output = torch.empty(
            (batch_size, self.out_channels, output_height, output_width),
            device=x.device, dtype=x.dtype
        )
        
        grid = (
            batch_size,
            self.out_channels,
            output_height * output_width
        )
        
        conv2d_kernel[grid](
            x, self.weight, self.bias, output,
            batch_size, self.in_channels, input_height, input_width,
            self.out_channels, self.stride,
            output_height, output_width,
            BLOCK_SIZE=1,
            KERNEL_SIZE=self.kernel_size
        )
        return output

class ModelNew(nn.Module):
    def __init__(self, num_classes):
        super(ModelNew, self).__init__()
        self.conv1 = TritonConv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = TritonConv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

batch_size = 1
num_classes = 10

def get_inputs():
    return [torch.randn(batch_size, 1, 32, 32)]

def get_init_inputs():
    return [num_classes]
# =================== EVOLVE-BLOCK-END ===================