# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _linear_swish_bias(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    in_features,
    out_features,
    stride_x_batch,
    stride_x_feature,
    stride_weight_out,
    stride_weight_feature,
    BLOCK_K: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    # 2D grid: batch index and output feature block index
    batch_idx = tl.program_id(0)
    out_feature_block_idx = tl.program_id(1)
    
    # Compute output feature block offsets
    out_feature_offsets = out_feature_block_idx * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    mask_out = out_feature_offsets < out_features
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
    
    # Process input features in blocks
    for k in range(0, in_features, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < in_features
        
        # Load input block
        x_ptr_row = x_ptr + batch_idx * stride_x_batch + k_offsets * stride_x_feature
        x_val = tl.load(x_ptr_row, mask=mask_k, other=0.0)
        
        # Load weight block
        weight_ptr_block = weight_ptr + out_feature_offsets[:, None] * stride_weight_out + k_offsets[None, :] * stride_weight_feature
        weight_val = tl.load(weight_ptr_block, mask=mask_out[:, None] & mask_k[None, :], other=0.0)
        
        # Compute partial dot product
        acc += tl.sum(weight_val * x_val[None, :], axis=1)
    
    # Load bias values (only linear bias)
    bias_val = tl.load(bias_ptr + out_feature_offsets, mask=mask_out, other=0.0)
    acc += bias_val
    
    # Apply Swish activation
    swish_val = acc * tl.sigmoid(acc)
    
    # Store results
    output_ptr_block = output_ptr + batch_idx * out_features + out_feature_offsets
    tl.store(output_ptr_block, swish_val, mask=mask_out)

class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, applies Swish activation, sums with a bias term, and normalizes with GroupNorm.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Prepare inputs and outputs
        x = x.contiguous()
        batch_size, _ = x.shape
        out_features = self.matmul.out_features
        output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
        
        # Configure kernel launch parameters
        BLOCK_OUT = 128
        BLOCK_K = 64
        num_blocks_out = (out_features + BLOCK_OUT - 1) // BLOCK_OUT
        grid = (batch_size, num_blocks_out)
        
        # Get tensor strides
        stride_x_batch = x.stride(0)
        stride_x_feature = x.stride(1)
        weight = self.matmul.weight
        stride_weight_out = weight.stride(0)
        stride_weight_feature = weight.stride(1)
        
        # Launch optimized kernel with only linear bias
        _linear_swish_bias[grid](
            x, weight, self.matmul.bias, output, 
            in_features, out_features,
            stride_x_batch, stride_x_feature,
            stride_weight_out, stride_weight_feature,
            BLOCK_K, BLOCK_OUT
        )
        
        # Add extra bias after Swish activation
        output = output + self.bias
        
        # Apply group normalization
        output = self.group_norm(output)
        return output

batch_size = 128
in_features = 512
out_features = 1024
num_groups = 32
bias_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
# =================== EVOLVE-BLOCK-END ===================