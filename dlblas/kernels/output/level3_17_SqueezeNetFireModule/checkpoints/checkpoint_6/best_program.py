# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

class TritonConv1x1Relu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)  # Fixed: replaced math.sqrt(5) with 5**0.5
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in > 0:
                bound = 1 / (fan_in ** 0.5)  # Fixed: replaced math.sqrt with exponent
                nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        batch, in_channels, height, width = x.shape
        n = height * width
        out_channels = self.out_channels
        
        # Reshape input to (batch, in_channels, n)
        x_2d = x.reshape(batch, in_channels, n)
        output = torch.empty(batch, out_channels, n, device=x.device, dtype=x.dtype)
        
        # Grid configuration
        grid = (batch, out_channels, n)
        
        # Launch kernel
        conv1x1_kernel[grid](
            x_2d, self.weight, self.bias, output,
            batch, in_channels, out_channels, n,
            x_2d.stride(0), x_2d.stride(1), x_2d.stride(2),
            self.weight.stride(0), self.weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_SIZE=32
        )
        
        return output.reshape(batch, out_channels, height, width)

@triton.jit
def conv1x1_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, in_channels, out_channels, n,
    stride_input_b, stride_input_ic, stride_input_n,
    stride_weight_oc, stride_weight_ic,
    stride_output_b, stride_output_oc, stride_output_n,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    if pid_b < batch and pid_oc < out_channels and pid_n < n:
        bias = tl.load(bias_ptr + pid_oc)
        acc = 0.0
        
        # Vectorized accumulation
        for ic in range(0, in_channels, BLOCK_SIZE):
            offs_ic = ic + tl.arange(0, BLOCK_SIZE)
            mask_ic = offs_ic < in_channels
            
            # Load weight block
            w_offs = pid_oc * stride_weight_oc + offs_ic
            weights = tl.load(weight_ptr + w_offs, mask=mask_ic, other=0.0)
            
            # Load input block
            in_offs = (pid_b * stride_input_b + 
                       offs_ic * stride_input_ic + 
                       pid_n * stride_input_n)
            inputs = tl.load(input_ptr + in_offs, mask=mask_ic, other=0.0)
            
            acc += tl.sum(weights * inputs)
        
        # Apply bias and ReLU
        result = tl.maximum(acc + bias, 0.0)
        
        # Store result
        out_offs = (pid_b * stride_output_b + 
                    pid_oc * stride_output_oc + 
                    pid_n * stride_output_n)
        tl.store(output_ptr + out_offs, result)

class ModelNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(ModelNew, self).__init__()
        
        # Use Triton for 1x1 convolutions (squeeze and expand1x1)
        self.squeeze = TritonConv1x1Relu(in_channels, squeeze_channels)
        self.expand1x1 = TritonConv1x1Relu(squeeze_channels, expand1x1_channels)
        
        # Keep original for 3x3 convolution
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat([
            self.expand1x1(x),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

# Test code
batch_size = 10
num_input_features = 3
num_output_features = 64
height, width = 224, 224
squeeze_channels = 6
expand1x1_channels = 64
expand3x3_channels = 64

def get_inputs():
    return [torch.randn(batch_size, num_input_features, height, width)]

def get_init_inputs():
    return [num_input_features, squeeze_channels, expand1x1_channels, expand3x3_channels]
# =================== EVOLVE-BLOCK-END ===================