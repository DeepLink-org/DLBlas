# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def _fused_patch_embedding(
    img_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    img_batch_stride, img_channel_stride, img_height_stride, img_width_stride,
    weight_row_stride, weight_col_stride,
    bias_stride,
    output_batch_stride, output_patch_stride, output_feature_stride,
    channels, height, width, patch_size, embed_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    num_patches = num_patches_h * num_patches_w
    
    batch_idx = pid0
    patch_idx = pid1
    patch_i = patch_idx // num_patches_w
    patch_j = patch_idx % num_patches_w
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for c in range(channels):
        for dx in range(patch_size):
            for dy in range(patch_size):
                # Load image pixel
                h_index = patch_i * patch_size + dx
                w_index = patch_j * patch_size + dy
                img_offset = batch_idx * img_batch_stride + c * img_channel_stride + h_index * img_height_stride + w_index * img_width_stride
                pixel = tl.load(img_ptr + img_offset)
                
                # Compute weight index
                weight_index = c * (patch_size * patch_size) + dx * patch_size + dy
                weight_offset = tl.arange(0, BLOCK_SIZE) * weight_row_stride + weight_index * weight_col_stride
                weights = tl.load(weight_ptr + weight_offset, mask=tl.arange(0, BLOCK_SIZE) < embed_dim)
                
                # Accumulate
                accumulator += pixel * weights
    
    # Add bias
    bias_offset = tl.arange(0, BLOCK_SIZE) * bias_stride
    biases = tl.load(bias_ptr + bias_offset, mask=tl.arange(0, BLOCK_SIZE) < embed_dim)
    accumulator += biases
    
    # Store output
    output_offset = batch_idx * output_batch_stride + patch_idx * output_patch_stride + tl.arange(0, BLOCK_SIZE) * output_feature_stride
    tl.store(output_ptr + output_offset, accumulator, mask=tl.arange(0, BLOCK_SIZE) < embed_dim)

class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super(ModelNew, self).__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )
        
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )
    
    def forward(self, img):
        p = self.patch_size
        batch_size, _, h, w = img.shape
        num_patches = (h // p) * (w // p)
        
        # Prepare output tensor
        x = torch.empty((batch_size, num_patches, dim), device=img.device, dtype=img.dtype)
        
        # Launch Triton kernel for fused patch extraction + embedding
        grid = (batch_size, num_patches)
        _fused_patch_embedding[grid](
            img,
            self.patch_to_embedding.weight,
            self.patch_to_embedding.bias,
            x,
            img.stride(0), img.stride(1), img.stride(2), img.stride(3),
            self.patch_to_embedding.weight.stride(0), self.patch_to_embedding.weight.stride(1),
            self.patch_to_embedding.bias.stride(0),
            x.stride(0), x.stride(1), x.stride(2),
            img.shape[1], h, w, p, dim,
            BLOCK_SIZE=triton.next_power_of_2(dim)
        )
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

# Test code
image_size = 224
patch_size = 16
num_classes = 10
dim = 512
depth = 6
heads = 8
mlp_dim = 2048
channels = 3
dropout = 0.0
emb_dropout = 0.0

def get_inputs():
    return [torch.randn(2, channels, image_size, image_size)]

def get_init_inputs():
    return [image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dropout, emb_dropout]
# =================== EVOLVE-BLOCK-END ===================