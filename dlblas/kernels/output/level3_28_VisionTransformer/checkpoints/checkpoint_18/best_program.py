# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def patch_embedding_kernel(
    img_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    stride_img_b, stride_img_c, stride_img_h, stride_img_w,
    stride_weight_0, stride_weight_1,
    stride_output_b, stride_output_p, stride_output_d,
    C, H, W, p, out_dim, num_patches,
    BLOCK_D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_patch = tl.program_id(1)
    pid_d = tl.program_id(2)
    
    patch_idx = pid_patch
    i = patch_idx // (W // p)
    j = patch_idx % (W // p)
    
    start_i = i * p
    start_j = j * p
    
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < out_dim
    
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    
    # Precompute flattened patch indices
    patch_size_sq = p * p
    total_k = C * patch_size_sq
    
    # Process flattened patch dimension in vectorized blocks
    for k in range(0, total_k, BLOCK_P):
        k_offsets = k + tl.arange(0, BLOCK_P)
        k_mask = k_offsets < total_k
        
        # Compute channel and spatial indices
        c = k_offsets // patch_size_sq
        residual = k_offsets % patch_size_sq
        dx = residual // p
        dy = residual % p
        
        # Vectorized image load
        img_offsets = (
            pid_b * stride_img_b + 
            c * stride_img_c + 
            (start_i + dx) * stride_img_h + 
            (start_j + dy) * stride_img_w
        )
        img_vals = tl.load(img_ptr + img_offsets, mask=k_mask, other=0.0)
        
        # Vectorized weight load
        weight_offsets = d_offsets[:, None] * stride_weight_0 + k_offsets[None, :] * stride_weight_1
        weight_vals = tl.load(weight_ptr + weight_offsets, mask=d_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Efficient matrix-vector product
        acc += tl.sum(weight_vals * img_vals[None, :], axis=1)
    
    # Add bias and store
    bias_vals = tl.load(bias_ptr + d_offsets, mask=d_mask, other=0.0)
    acc += bias_vals
    
    output_offsets = (
        pid_b * stride_output_b + 
        patch_idx * stride_output_p + 
        d_offsets * stride_output_d
    )
    tl.store(output_ptr + output_offsets, acc, mask=d_mask)

class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super(ModelNew, self).__init__()
        
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.dim = dim
        
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
        B, C, H, W = img.shape
        num_patches = self.num_patches
        
        # Optimized block sizes
        BLOCK_D = 32
        BLOCK_P = 64  # Increased for better vectorization
        
        x = torch.empty((B, num_patches, self.dim), device=img.device, dtype=torch.float32)
        
        grid = (B, num_patches, triton.cdiv(self.dim, BLOCK_D))
        
        patch_embedding_kernel[grid](
            img,
            self.patch_to_embedding.weight,
            self.patch_to_embedding.bias,
            x,
            img.stride(0), img.stride(1), img.stride(2), img.stride(3),
            self.patch_to_embedding.weight.stride(0), self.patch_to_embedding.weight.stride(1),
            x.stride(0), x.stride(1), x.stride(2),
            C, H, W, p, self.dim, num_patches,
            BLOCK_D=BLOCK_D,
            BLOCK_P=BLOCK_P
        )
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
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