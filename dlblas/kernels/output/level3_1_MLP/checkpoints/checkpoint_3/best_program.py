# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _linear_kernel(
    x_ptr,  # Input tensor pointer
    w_ptr,  # Weight tensor pointer
    bias_ptr,  # Bias tensor pointer
    output_ptr,  # Output tensor pointer
    in_features,  # Input feature dimension
    out_features,  # Output feature dimension
    stride_x,  # Stride for input tensor (batch, in_features)
    stride_w,  # Stride for weight tensor (out_features, in_features)
    stride_output,  # Stride for output tensor
    BLOCK_SIZE_K: tl.constexpr,  # Block size for reduction dimension
    USE_BIAS: tl.constexpr,  # Whether to use bias
):
    # Extract program IDs
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    
    # Create pointers for batch and feature offsets
    batch_offset = pid_batch * stride_x
    feature_offset = pid_feature * stride_w
    
    # Initialize accumulator
    acc = 0.0
    # Loop over reduction dimension in blocks
    for k in range(0, in_features, BLOCK_SIZE_K):
        # Create range for current block
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        # Create mask to prevent out-of-bounds access
        mask = k_offsets < in_features
        
        # Load input block
        x = tl.load(x_ptr + batch_offset + k_offsets, mask=mask, other=0.0)
        # Load weight block
        w = tl.load(w_ptr + feature_offset + k_offsets, mask=mask, other=0.0)
        # Accumulate dot product
        acc += tl.sum(x * w)
    
    # Add bias if present
    if USE_BIAS:
        bias = tl.load(bias_ptr + pid_feature)
        acc += bias
    
    # Store result
    tl.store(output_ptr + pid_batch * stride_output + pid_feature, acc)

class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        batch_size = x.shape[0]
        output = torch.empty(batch_size, self.out_features, device=x.device, dtype=x.dtype)
        
        # Set grid and block dimensions
        grid = (batch_size, self.out_features)
        BLOCK_SIZE_K = 128  # Optimized for H100 memory access
        
        # Launch kernel
        _linear_kernel[grid](
            x, self.weight, self.bias, output,
            self.in_features, self.out_features,
            x.stride(1), self.weight.stride(1), output.stride(1),
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            USE_BIAS=(self.bias is not None)
        )
        return output

class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(ModelNew, self).__init__()
        layers = []
        current_input_size = input_size
        for layer_size in layer_sizes:
            layers.append(TritonLinear(current_input_size, layer_size))
            layers.append(nn.ReLU())
            current_input_size = layer_size
        layers.append(TritonLinear(current_input_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Test code
batch_size = 1
input_size = 1000
layer_sizes = [400, 800]
output_size = 500

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, layer_sizes, output_size]
# =================== EVOLVE-BLOCK-END ===================