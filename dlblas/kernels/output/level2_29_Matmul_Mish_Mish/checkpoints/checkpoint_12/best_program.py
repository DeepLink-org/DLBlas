# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_linear_mish_mish_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    in_features,
    out_features,
    stride_x,
    stride_out,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    block_k = tl.arange(0, BLOCK_K)
    mask_k = block_k < in_features

    # Load input row
    x_row = tl.load(x_ptr + pid * stride_x + block_k, mask=mask_k, other=0.0)
    
    # Initialize accumulator
    acc = tl.zeros((out_features,), dtype=tl.float32)
    
    # Compute dot products for all output features
    for j in range(out_features):
        w_ptrs = weight_ptr + j * in_features + block_k
        w_row = tl.load(w_ptrs, mask=mask_k, other=0.0)
        acc = tl.where(tl.arange(0, out_features) == j, 
                      tl.sum(x_row * w_row), 
                      acc)
    
    # Add bias
    bias = tl.load(bias_ptr + tl.arange(0, out_features))
    acc += bias
    
    # Apply Mish activation twice
    softplus1 = tl.log(1.0 + tl.exp(-tl.abs(acc))) + tl.maximum(acc, 0.0)
    tanh1 = tl.tanh(softplus1)
    mish1 = acc * tanh1
    
    softplus2 = tl.log(1.0 + tl.exp(-tl.abs(mish1))) + tl.maximum(mish1, 0.0)
    tanh2 = tl.tanh(softplus2)
    mish2 = mish1 * tanh2
    
    # Store results
    tl.store(output_ptr + pid * stride_out + tl.arange(0, out_features), mish2)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        
    def forward(self, x):
        output = torch.empty(x.shape[0], self.out_features, 
                            device=x.device, dtype=x.dtype)
        if x.is_cuda:
            BLOCK_K = triton.next_power_of_2(self.in_features)
            grid = (x.shape[0],)
            fused_linear_mish_mish_kernel[grid](
                x, 
                self.linear.weight, 
                self.linear.bias,
                output,
                self.in_features,
                self.out_features,
                x.stride(0),
                output.stride(0),
                BLOCK_K=BLOCK_K,
            )
        else:
            # Fallback for CPU
            x = self.linear(x)
            x = torch.nn.functional.mish(x)
            x = torch.nn.functional.mish(x)
            output = x
        return output

batch_size = 128
in_features = 10
out_features = 20

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]
# =================== EVOLVE-BLOCK-END ===================