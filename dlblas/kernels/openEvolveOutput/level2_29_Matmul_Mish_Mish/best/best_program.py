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
):
    pid = tl.program_id(0)
    batch_idx = pid // out_features
    feature_idx = pid % out_features

    # Load input row
    x_row_ptr = x_ptr + batch_idx * stride_x
    weight_row_ptr = weight_ptr + feature_idx * in_features

    # Compute dot product
    acc = 0.0
    for k in range(0, in_features):
        x_val = tl.load(x_row_ptr + k)
        w_val = tl.load(weight_row_ptr + k)
        acc += x_val * w_val

    # Add bias
    bias_val = tl.load(bias_ptr + feature_idx)
    acc += bias_val

    # First Mish activation (inlined)
    abs_acc = tl.abs(acc)
    softplus_acc = tl.log(1.0 + tl.exp(-abs_acc)) + tl.maximum(acc, 0.0)
    tanh_sp_acc = 2 * tl.sigmoid(2 * softplus_acc) - 1
    mish1 = acc * tanh_sp_acc

    # Second Mish activation (inlined)
    abs_mish1 = tl.abs(mish1)
    softplus_mish1 = tl.log(1.0 + tl.exp(-abs_mish1)) + tl.maximum(mish1, 0.0)
    tanh_sp_mish1 = 2 * tl.sigmoid(2 * softplus_mish1) - 1
    mish2 = mish1 * tanh_sp_mish1
    
    # Store result
    tl.store(output_ptr + batch_idx * stride_out + feature_idx, mish2)

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
            x = x.contiguous()
            weight = self.linear.weight.contiguous()
            bias = self.linear.bias.contiguous()
            
            total_elements = x.shape[0] * self.out_features
            grid = (total_elements,)
            fused_linear_mish_mish_kernel[grid](
                x, 
                weight, 
                bias,
                output,
                self.in_features,
                self.out_features,
                x.stride(0),
                output.stride(0),
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