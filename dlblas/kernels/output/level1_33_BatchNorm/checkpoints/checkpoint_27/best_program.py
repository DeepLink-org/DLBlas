# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _batchnorm_forward_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    N, C, H, W,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    mean_val = tl.load(mean_ptr + c)
    var_val = tl.load(var_ptr + c)
    weight_val = tl.load(weight_ptr + c)
    bias_val = tl.load(bias_ptr + c)
    
    inv_std = tl.math.rsqrt(var_val + eps)
    scale = weight_val * inv_std
    
    base_idx = n * C * H * W + c * H * W
    base_ptr = x_ptr + base_idx
    out_base_ptr = out_ptr + base_idx
    
    total_elements = H * W
    for start in range(0, total_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements
        x_vals = tl.load(base_ptr + offsets, mask=mask, other=0.0)
        
        centered = x_vals - mean_val
        scaled = centered * scale
        normalized = scaled + bias_val
        
        tl.store(out_base_ptr + offsets, normalized, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = 1e-5
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return torch.nn.functional.batch_norm(
                x, self.running_mean, self.running_var, 
                self.weight, self.bias, training=True, 
                momentum=0, eps=self.eps
            )
        else:
            x = x.contiguous()
            out = torch.empty_like(x)
            N, C, H, W = x.shape
            grid = (N * C,)
            BLOCK_SIZE = 1024
            _batchnorm_forward_kernel[grid](
                x, out, 
                self.running_mean, self.running_var, 
                self.weight, self.bias,
                N, C, H, W, self.eps, 
                BLOCK_SIZE=BLOCK_SIZE
            )
            return out

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]
# =================== EVOLVE-BLOCK-END ===================