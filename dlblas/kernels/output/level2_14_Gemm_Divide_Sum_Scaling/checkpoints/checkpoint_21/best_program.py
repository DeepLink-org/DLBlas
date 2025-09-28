# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fused_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    scaling_factor,
    input_size,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    total = 0.0
    for j in range(0, hidden_size):
        acc = 0.0
        for i in range(0, input_size):
            x_val = tl.load(x_ptr + pid * input_size + i)
            w_val = tl.load(weight_ptr + j * input_size + i)
            acc += x_val * w_val
        acc = acc / 2.0
        total += acc
        
    total = total * scaling_factor
    tl.store(output_ptr + pid, total)

class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, division, summation, and scaling.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = x.contiguous()
        weight = self.weight.contiguous()
        batch_size = x.size(0)
        output = torch.empty(batch_size, dtype=torch.float32, device=x.device)
        
        grid = (batch_size,)
        _fused_kernel[grid](
            x, weight, output, self.scaling_factor, 
            x.size(1), weight.size(0), 
            BLOCK_SIZE=triton.next_power_of_2(x.size(1))
        )
        return output.unsqueeze(1)


batch_size = 128
input_size = 10
hidden_size = 20
scaling_factor = 1.5

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
# =================== EVOLVE-BLOCK-END ===================