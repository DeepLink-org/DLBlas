# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fused_forward_kernel(
    x_ptr, weight_ptr, bias_ptr, add_value_ptr, output_ptr,
    batch_size, in_features, out_features,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_FEAT: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_feat = tl.program_id(1)
    
    batch_offsets = pid_batch * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    feat_offsets = pid_feat * BLOCK_SIZE_FEAT + tl.arange(0, BLOCK_SIZE_FEAT)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    batch_mask = batch_offsets < batch_size
    feat_mask = feat_offsets < out_features
    
    accumulator = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_FEAT), dtype=tl.float32)
    
    for k in range(0, in_features, BLOCK_SIZE_K):
        k_mask = k_offsets < (in_features - k)
        input_mask = batch_mask[:, None] & k_mask[None, :]
        weight_mask = feat_mask[:, None] & k_mask[None, :]
        
        x_offset = batch_offsets[:, None] * in_features + (k + k_offsets[None, :])
        w_offset = feat_offsets[:, None] * in_features + (k + k_offsets[None, :])
        
        x_tile = tl.load(x_ptr + x_offset, mask=input_mask, other=0.0)
        w_tile = tl.load(weight_ptr + w_offset, mask=weight_mask, other=0.0)
        
        accumulator += tl.dot(x_tile, w_tile, trans_b=True)
    
    bias_vals = tl.load(bias_ptr + feat_offsets, mask=feat_mask, other=0.0)
    add_vals = tl.load(add_value_ptr + feat_offsets, mask=feat_mask, other=0.0)
    
    result = accumulator + bias_vals[None, :] + add_vals[None, :]
    
    # Fused activations
    swish = tl.sigmoid(result) * result
    tanh = tl.tanh(swish)
    gelu = tanh * 0.5 * (1.0 + tl.erf(tanh * 0.7071067811865475))
    hardtanh = tl.minimum(tl.maximum(gelu, -1.0), 1.0)
    
    output_offset = batch_offsets[:, None] * out_features + feat_offsets[None, :]
    output_mask = batch_mask[:, None] & feat_mask[None, :]
    tl.store(output_ptr + output_offset, hardtanh, mask=output_mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
    
    def forward(self, x):
        weight = self.matmul.weight
        bias = self.matmul.bias
        add_value = self.add_value
        
        batch_size, in_dim = x.shape
        out_dim = weight.shape[0]
        output = torch.empty((batch_size, out_dim), device=x.device, dtype=x.dtype)
        
        BLOCK_SIZE_BATCH = 32
        BLOCK_SIZE_FEAT = 32
        BLOCK_SIZE_K = 64
        
        grid = (
            triton.cdiv(batch_size, BLOCK_SIZE_BATCH),
            triton.cdiv(out_dim, BLOCK_SIZE_FEAT)
        )
        
        _fused_forward_kernel[grid](
            x, weight, bias, add_value, output,
            batch_size, in_dim, out_dim,
            BLOCK_SIZE_BATCH, BLOCK_SIZE_FEAT, BLOCK_SIZE_K
        )
        return output

batch_size = 128
in_features = 1024
out_features = 512
add_value_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, add_value_shape]
# =================== EVOLVE-BLOCK-END ===================