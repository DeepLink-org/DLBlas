# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def _per_example_sum(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    stride_x,
    stride_weight,
    stride_bias,
    stride_output,
    input_size: tl.constexpr,
    hidden_size: tl.constexpr,
    batch_size: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    s = 0.0
    for j in range(hidden_size):
        b = tl.load(bias_ptr + j * stride_bias)
        dot = b
        for k in range(input_size):
            x_val = tl.load(x_ptr + pid * stride_x + k)
            w_val = tl.load(weight_ptr + j * stride_weight + k)
            dot += x_val * w_val
        sig = 1.0 / (1.0 + tl.exp(-dot))
        s += sig
    tl.store(output_ptr + pid * stride_output, s)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.contiguous()
        output_per_example = torch.empty(batch_size, device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(batch_size, 1),)
        _per_example_sum[grid](
            x, self.weight, self.bias, output_per_example,
            x.stride(0), self.weight.stride(0), 1, 1,
            self.input_size, self.hidden_size, batch_size
        )
        x = torch.logsumexp(output_per_example, dim=0)
        return x

batch_size = 128
input_size = 10
hidden_size = 20
output_size = 5

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, output_size]
# =================== EVOLVE-BLOCK-END ===================