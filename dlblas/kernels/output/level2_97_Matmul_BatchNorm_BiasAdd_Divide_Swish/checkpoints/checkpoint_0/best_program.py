# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_forward_kernel(
    linear_out_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    model_bias_ptr,
    divide_value,
    output_ptr,
    batch_size,
    out_features,
    bn_eps,
    stride_lo0, stride_lo1,
    stride_out0, stride_out1,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_FEATURE: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_feature = tl.program_id(1)
    
    batch_idx = pid_batch * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    feature_idx = pid_feature * BLOCK_SIZE_FEATURE + tl.arange(0, BLOCK_SIZE_FEATURE)
    
    batch_mask = batch_idx < batch_size
    feature_mask = feature_idx < out_features
    block_mask = batch_mask[:, None] & feature_mask[None, :]
    
    # Calculate offsets for linear_out access
    lo_offsets = batch_idx[:, None] * stride_lo0 + feature_idx[None, :] * stride_lo1
    linear_out = tl.load(linear_out_ptr + lo_offsets, mask=block_mask, other=0.0)
    
    # Load batchnorm parameters for the feature block
    bn_weight = tl.load(bn_weight_ptr + feature_idx, mask=feature_mask, other=0.0)
    bn_bias = tl.load(bn_bias_ptr + feature_idx, mask=feature_mask, other=0.0)
    running_mean = tl.load(running_mean_ptr + feature_idx, mask=feature_mask, other=0.0)
    running_var = tl.load(running_var_ptr + feature_idx, mask=feature_mask, other=0.0)
    
    # Compute batchnorm
    scale = bn_weight / tl.sqrt(running_var + bn_eps)
    bn_out = (linear_out - running_mean[None, :]) * scale[None, :] + bn_bias[None, :]
    
    # Add model bias and divide
    model_bias = tl.load(model_bias_ptr)
    normalized = (bn_out + model_bias) / divide_value
    
    # Swish activation
    swish_out = normalized * tl.sigmoid(normalized)
    
    # Store results
    out_offsets = batch_idx[:, None] * stride_out0 + feature_idx[None, :] * stride_out1
    tl.store(output_ptr + out_offsets, swish_out, mask=block_mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value
        
    def forward(self, x):
        linear_out = self.matmul(x)
        
        if self.training:
            x = self.bn(linear_out)
            x = x + self.bias
            x = x / self.divide_value
            return x * torch.sigmoid(x)
        else:
            # Inference path with fused Triton kernel
            batch_size, _ = linear_out.shape
            output = torch.empty_like(linear_out)
            
            grid = lambda meta: (
                triton.cdiv(batch_size, meta['BLOCK_SIZE_BATCH']),
                triton.cdiv(linear_out.size(1), meta['BLOCK_SIZE_FEATURE'])
            )
            
            fused_forward_kernel[grid](
                linear_out,
                self.bn.weight,
                self.bn.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bias,
                self.divide_value,
                output,
                batch_size,
                linear_out.size(1),
                self.bn.eps,
                linear_out.stride(0), linear_out.stride(1),
                output.stride(0), output.stride(1),
                BLOCK_SIZE_BATCH=32,
                BLOCK_SIZE_FEATURE=128
            )
            return output

batch_size = 128
in_features = 1024
out_features = 512
bn_eps = 1e-5
bn_momentum = 0.1
bias_shape = (1,)
divide_value = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, bias_shape, divide_value]
# =================== EVOLVE-BLOCK-END ===================