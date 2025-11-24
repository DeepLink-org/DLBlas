# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_linear_relu_div(
    x_ptr, w_ptr, b_ptr, output_ptr,
    in_features, out_features, divisor,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Extract program IDs
    pid_batch = tl.program_id(0)
    pid_block = tl.program_id(1)
    
    # Calculate output feature indices
    j = pid_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_j = j < out_features
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Loop over input features in blocks
    for k in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < in_features
        
        # Load input block
        x_chunk = tl.load(
            x_ptr + pid_batch * in_features + offs_k,
            mask=mask_k, other=0.0
        )
        
        # Load weight block
        w_ptrs = w_ptr + j[:, None] * in_features + offs_k[None, :]
        w_chunk = tl.load(
            w_ptrs,
            mask=mask_j[:, None] & mask_k[None, :],
            other=0.0
        )
        
        # Compute partial dot product
        acc += tl.sum(w_chunk * x_chunk, axis=1)
    
    # Load bias and add
    bias = tl.load(b_ptr + j, mask=mask_j, other=0.0)
    acc += bias
    
    # Apply ReLU and scaling
    acc = tl.maximum(acc, 0.0)
    acc = acc / divisor
    
    # Store result
    output_ptrs = output_ptr + pid_batch * out_features + j
    tl.store(output_ptrs, acc, mask=mask_j)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.divisor = divisor
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = x.contiguous()
        output = torch.empty(x.size(0), self.out_features, device=x.device, dtype=x.dtype)
        
        grid = (x.size(0), triton.cdiv(self.out_features, 128))
        fused_linear_relu_div[grid](
            x, self.weight, self.bias, output,
            self.in_features, self.out_features, self.divisor,
            BLOCK_SIZE_N=128, BLOCK_SIZE_K=64
        )
        
        return output

batch_size = 128
in_features = 1024
out_features = 512
divisor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, divisor]
# =================== EVOLVE-BLOCK-END ===================