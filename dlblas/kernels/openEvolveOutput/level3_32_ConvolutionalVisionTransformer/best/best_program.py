# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def linear_proj_kernel(
    x_ptr, 
    weight_ptr, 
    bias_ptr, 
    output_ptr,
    B, 
    D, 
    embed_dim,
    stride_x_batch,
    stride_w_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    off_x = pid * stride_x_batch
    off_out = pid * embed_dim

    for j in range(0, embed_dim, BLOCK_SIZE):
        j_off = j + tl.arange(0, BLOCK_SIZE)
        j_mask = j_off < embed_dim
        off_w = j_off * stride_w_row
        
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        for i in range(0, D, 64):
            i_off = i + tl.arange(0, 64)
            i_mask = i_off < D
            
            x = tl.load(x_ptr + off_x + i_off, mask=i_mask, other=0.0)
            w_ptrs = weight_ptr + off_w[:, None] + i_off[None, :]
            w = tl.load(w_ptrs, mask=j_mask[:, None] & i_mask[None, :], other=0.0)
            
            acc += tl.sum(w * x[None, :], axis=1)

        if bias_ptr is not None:
            b = tl.load(bias_ptr + j_off, mask=j_mask, other=0.0)
            acc += b
            
        tl.store(output_ptr + off_out + j_off, acc, mask=j_mask)

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
        
        # Replace linear projection with optimized Triton kernel
        D = self.in_features
        embed_dim = self.linear_proj.out_features
        weight = self.linear_proj.weight
        bias = self.linear_proj.bias if self.linear_proj.bias is not None else None
        
        # Precompute strides
        stride_x_batch = x.stride(0)
        stride_w_row = weight.stride(0)
        
        # Allocate output tensor
        output = torch.empty((B, embed_dim), device=x.device, dtype=x.dtype)
        
        # Configure kernel launch
        grid = (B,)
        linear_proj_kernel[grid](
            x, weight, bias, output,
            B, D, embed_dim,
            stride_x_batch, stride_w_row,
            BLOCK_SIZE=128
        )
        x = output
        
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