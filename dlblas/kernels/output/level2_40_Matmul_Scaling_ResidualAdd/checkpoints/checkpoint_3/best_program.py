# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_scale_residual_kernel(
    y_ptr,
    output_ptr,
    scaling_factor,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    y = tl.load(y_ptr + offsets, mask=mask)
    scaled = y * scaling_factor
    output = scaled + y
    tl.store(output_ptr + offsets, output, mask=mask)

class ScaledResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, scaling_factor):
        ctx.save_for_backward(scaling_factor)
        output = torch.empty_like(y)
        n_elements = y.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        fused_scale_residual_kernel[grid](
            y, output, scaling_factor, n_elements, 
            BLOCK_SIZE=1024
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        scaling_factor, = ctx.saved_tensors
        return grad_output * scaling_factor, None

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.matmul(x)
        return ScaledResidual.apply(x, self.scaling_factor)

batch_size = 128
in_features = 64
out_features = 128
scaling_factor = 0.5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]
# =================== EVOLVE-BLOCK-END ===================