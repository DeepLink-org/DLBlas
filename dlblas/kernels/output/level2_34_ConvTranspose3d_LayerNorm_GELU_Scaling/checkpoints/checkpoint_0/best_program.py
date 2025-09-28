# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_layernorm_gelu_scale(
    input_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    eps,
    scaling_factor,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offsets % weight_ptr.shape[0], mask=mask, other=0.0)
    b = tl.load(bias_ptr + offsets % bias_ptr.shape[0], mask=mask, other=0.0)
    
    # LayerNorm computation
    mean = tl.sum(x, axis=0) / n_elements
    var = tl.sum((x - mean) * (x - mean), axis=0) / n_elements
    rstd = 1.0 / tl.sqrt(var + eps)
    normalized = (x - mean) * rstd * w + b
    
    # GELU approximation
    gelu = normalized * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (normalized + 0.044715 * normalized * normalized * normalized)))
    
    # Scaling
    result = gelu * scaling_factor
    tl.store(output_ptr + offsets, result, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor
        self.eps = eps

    def forward(self, x):
        x = self.conv_transpose(x)
        
        # Save original shape and reshape for fused kernel
        original_shape = x.shape
        n_elements = x.numel()
        x_flat = x.view(-1)
        
        # Prepare output tensor
        output = torch.empty_like(x_flat)
        
        # Launch kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        fused_layernorm_gelu_scale[grid](
            x_flat, output, 
            self.layer_norm.weight, 
            self.layer_norm.bias,
            self.eps,
            self.scaling_factor,
            n_elements,
            BLOCK_SIZE=1024
        )
        
        return output.view(original_shape)

batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 4
stride = 2
padding = 1
bias = True
eps = 1e-5
scaling_factor = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias, eps, scaling_factor]
# =================== EVOLVE-BLOCK-END ===================