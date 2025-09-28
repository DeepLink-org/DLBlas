# ================== EVOLVE-BLOCK-START ==================
import torch
import triton
import triton.language as tl

@triton.jit
def mish_subtract_kernel(
    input_ptr,
    output_ptr,
    subtract1,
    subtract2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Fused operations: subtract both values then apply Mish
    x = x - subtract1 - subtract2
    # Mish activation: x * tanh(softplus(x))
    softplus = tl.log(1.0 + tl.exp(x))
    mish_out = x * tl.tanh(softplus)
    
    # Store result
    tl.store(output_ptr + offsets, mish_out, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

    def forward(self, x):
        # Convolution using highly optimized PyTorch/cuDNN
        x = self.conv(x)
        
        # Fuse post-conv operations with Triton kernel
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        mish_subtract_kernel[grid](
            x.view(-1), output.view(-1), 
            self.subtract_value_1, self.subtract_value_2,
            n_elements, BLOCK_SIZE=1024
        )
        return output.reshape_as(x)

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]
# =================== EVOLVE-BLOCK-END ===================