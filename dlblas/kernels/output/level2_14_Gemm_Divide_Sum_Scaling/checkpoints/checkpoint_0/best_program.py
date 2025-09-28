# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _forward_kernel(
    x_ptr,
    w_sum_ptr,
    output_ptr,
    batch_size,
    input_size: tl.constexpr,
    scaling_factor,
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    
    row_offset = pid * input_size
    acc = 0.0
    for k in range(0, input_size):
        x_val = tl.load(x_ptr + row_offset + k)
        w_val = tl.load(w_sum_ptr + k)
        acc += x_val * w_val
        
    acc = acc * (scaling_factor / 2.0)
    tl.store(output_ptr + pid, acc)

class ModelNew(nn.Module):
    """
    Optimized model using Triton for fused operations.
    Output tensor has shape (batch_size, 1).
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = x.contiguous()
        w_sum = self.weight.sum(dim=0)
        batch_size, input_size = x.shape
        output = torch.empty(batch_size, 1, device=x.device, dtype=x.dtype)
        
        if x.is_cuda:
            grid = (batch_size,)
            _forward_kernel[grid](x, w_sum, output, batch_size, input_size, self.scaling_factor)
        else:
            output = (x @ w_sum).unsqueeze(1) * (self.scaling_factor / 2)
            
        return output


batch_size = 128
input_size = 10
hidden_size = 20
scaling_factor = 1.5

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
# =================== EVOLVE-BLOCK-END ===================