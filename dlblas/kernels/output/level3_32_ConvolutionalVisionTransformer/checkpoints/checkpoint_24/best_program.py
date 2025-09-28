# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Calculate offsets and mask
    offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_idx * BLOCK_SIZE + offsets < out_features
    row_mask = row_idx * BLOCK_SIZE + offsets < in_features
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over input features in blocks
    for block_start in range(0, in_features, BLOCK_SIZE):
        block_offs = block_start + offsets
        
        # Load x vector
        x_mask = block_offs < in_features
        x_vals = tl.load(
            x_ptr + row_idx * in_features + block_offs,
            mask=x_mask,
            other=0.0
        )
        
        # Load weight block
        w_mask = (block_offs < in_features) & col_mask
        w_vals = tl.load(
            weight_ptr + col_idx * BLOCK_SIZE * in_features + block_offs,
            mask=w_mask,
            other=0.0
        )
        
        # Compute partial dot product
        acc += x_vals * w_vals
    
    # Apply bias if available
    if bias_ptr is not None:
        b_vals = tl.load(
            bias_ptr + col_idx * BLOCK_SIZE + offsets,
            mask=col_mask,
            other=0.0
        )
        acc += b_vals
    
    # Store result
    output_offs = row_idx * out_features + col_idx * BLOCK_SIZE + offsets
    tl.store(
        output_ptr + output_offs,
        acc,
        mask=col_mask
    )

class ModelNew(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6, 
                 mlp_ratio=4.0, patch_size=4, in_channels=3):
        super(ModelNew, self).__init__()
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten()
        
        # Calculate input features for linear layer
        spatial_dim = 32 // patch_size
        self.in_features = embed_dim * spatial_dim * spatial_dim
        self.linear_proj = nn.Linear(self.in_features, embed_dim)
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                       dim_feedforward=int(embed_dim * mlp_ratio), dropout=0.0)
            for _ in range(num_layers)
        ])
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc_out = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv1(x)
        x = self.flatten(x)
        
        # Use Triton for linear projection
        weight = self.linear_proj.weight
        bias = self.linear_proj.bias
        out_features = weight.size(0)
        
        # Allocate output tensor
        x_triton = torch.empty((B, out_features), device=x.device, dtype=x.dtype)
        
        # Configure kernel launch
        BLOCK_SIZE = 128
        grid = (
            triton.cdiv(B, BLOCK_SIZE),
            triton.cdiv(out_features, BLOCK_SIZE)
        )
        
        # Launch Triton kernel
        linear_kernel[grid](
            x, weight, bias, x_triton,
            self.in_features, out_features,
            BLOCK_SIZE
        )
        x = x_triton
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)
        
        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Classify based on cls token
        x = x[:, 0]
        x = self.fc_out(x)
        return x

batch_size = 10
image_size = 32
embed_dim = 128
in_channels = 3
num_heads = 4
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, in_channels, image_size, image_size)]

def get_init_inputs():
    return [num_classes, embed_dim, num_heads]
# =================== EVOLVE-BLOCK-END ===================