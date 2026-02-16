# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_patch_embedding_kernel(
    img_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    stride_img_b, stride_img_c, stride_img_h, stride_img_w,
    stride_weight_0, stride_weight_1,
    bias_stride,
    stride_output_b, stride_output_p, stride_output_d,
    C, H, W, P, embed_dim,
    BLOCK_D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_d = tl.program_id(2)
    
    num_patches_w = W // P
    patch_i = pid_p // num_patches_w
    patch_j = pid_p % num_patches_w
    start_i = patch_i * P
    start_j = patch_j * P
    
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < embed_dim
    
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    
    for c in range(0, C):
        for dx in range(0, P, BLOCK_P):
            dx_offsets = dx + tl.arange(0, BLOCK_P)
            dx_mask = dx_offsets < P
            
            for dy in range(0, P, BLOCK_P):
                dy_offsets = dy + tl.arange(0, BLOCK_P)
                dy_mask = dy_offsets < P
                
                img_offsets = (
                    pid_b * stride_img_b + 
                    c * stride_img_c + 
                    (start_i + dx_offsets[:, None]) * stride_img_h + 
                    (start_j + dy_offsets[None, :]) * stride_img_w
                )
                img_vals = tl.load(
                    img_ptr + img_offsets, 
                    mask=dx_mask[:, None] & dy_mask[None, :], 
                    other=0.0
                )
                
                w_index = c * P * P + dx_offsets[None, :, None] * P + dy_offsets[None, None, :]
                weight_offsets = d_offsets[:, None, None] * stride_weight_0 + w_index * stride_weight_1
                weight_vals = tl.load(
                    weight_ptr + weight_offsets, 
                    mask=d_mask[:, None, None] & dx_mask[None, :, None] & dy_mask[None, None, :], 
                    other=0.0
                )
                
                product = img_vals * weight_vals
                partial_sum = tl.sum(product, axis=2)
                full_sum = tl.sum(partial_sum, axis=1)
                acc += full_sum
    
    bias_vals = tl.load(bias_ptr + d_offsets, mask=d_mask, other=0.0)
    acc += bias_vals
    
    output_offsets = (
        pid_b * stride_output_b + 
        pid_p * stride_output_p + 
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
        
        BLOCK_D = 128  # Optimized for H100's 128-bit wide loads
        BLOCK_P = 16   # Process full patch rows at once
        
        x = torch.empty((B, num_patches, self.dim), device=img.device, dtype=img.dtype)
        
        grid = (B, num_patches, triton.cdiv(self.dim, BLOCK_D))
        
        fused_patch_embedding_kernel[grid](
            img,
            self.patch_to_embedding.weight,
            self.patch_to_embedding.bias,
            x,
            img.stride(0), img.stride(1), img.stride(2), img.stride(3),
            self.patch_to_embedding.weight.stride(0), self.patch_to_embedding.weight.stride(1),
            self.patch_to_embedding.bias.stride(0),
            x.stride(0), x.stride(1), x.stride(2),
            C, H, W, p, self.dim,
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