# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Triton kernel for optimized linear layer
@triton.jit
def linear_kernel(
    a_ptr, w_ptr, bias_ptr, output_ptr,
    K, N, 
    stride_ab, stride_ak,
    stride_wk, stride_wn,
    stride_ob, stride_on,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // N
    n = pid % N

    a_ptr_batch = a_ptr + batch_idx * stride_ab
    output_ptr_batch = output_ptr + batch_idx * stride_ob

    acc = 0.0
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        mask = offs_k < K - k
        a_val = tl.load(a_ptr_batch + k + offs_k, mask=mask, other=0.0)
        w_val = tl.load(w_ptr + n * stride_wn + (k + offs_k) * stride_wk, mask=mask, other=0.0)
        acc += tl.sum(a_val * w_val)

    if bias_ptr is not None:
        b = tl.load(bias_ptr + n)
        acc += b

    tl.store(output_ptr_batch + n * stride_on, acc)

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
        output = torch.empty(x.shape[0], self.out_features, device=x.device, dtype=x.dtype)
        batch_size = x.shape[0]
        grid = (batch_size * self.out_features,)
        
        linear_kernel[grid](
            x, self.weight, self.bias, output,
            self.in_features, self.out_features,
            x.stride(0), x.stride(1),
            self.weight.stride(1), self.weight.stride(0),
            output.stride(0), output.stride(1),
            BLOCK_SIZE_K=128
        )
        return output

class ModelNew(nn.Module):
    def __init__(self, num_classes):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = TritonLinear(16*5*5, 120)
        self.fc2 = TritonLinear(120, 84)
        self.fc3 = TritonLinear(84, num_classes)
    
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