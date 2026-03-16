import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 64}, num_warps=2),
        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 256}, num_warps=8),
    ],
    key=["D"],
)
@triton.jit
def _cat_add_pos_kernel(
    x_ptr,           # float*          [B, N, D]
    pos_ptr,         # float*          [N+1, D] (already sliced pos_embedding[0])
    cls_ptr,         # float*          [D]      (already sliced cls_token[0, 0])
    out_ptr,         # float*          [B, N+1, D]
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    stride_xb, stride_xn, stride_xd,        # strides for x
    stride_posn, stride_posd,               # strides for pos
    stride_clsd,                            # stride for cls token vector
    stride_outb, stride_outn, stride_outd,  # strides for out
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_d = tl.program_id(2)

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    # Load positional embedding row
    pos_ptrs = pos_ptr + pid_n * stride_posn + d_offsets * stride_posd
    pos = tl.load(pos_ptrs, mask=d_mask, other=0.0)

    # Select source: cls token for n==0, else patch embedding from x
    if pid_n == 0:
        src_ptrs = cls_ptr + d_offsets * stride_clsd
        src = tl.load(src_ptrs, mask=d_mask, other=0.0)
    else:
        src_ptrs = x_ptr + pid_b * stride_xb + (pid_n - 1) * stride_xn + d_offsets * stride_xd
        src = tl.load(src_ptrs, mask=d_mask, other=0.0)

    out = src + pos
    out_ptrs = out_ptr + pid_b * stride_outb + pid_n * stride_outn + d_offsets * stride_outd
    tl.store(out_ptrs, out, mask=d_mask)


class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        """
        Vision Transformer (ViT) model.

        :param image_size: The size of the input image (assumed to be square).
        :param patch_size: The size of each patch (assumed to be square).
        :param num_classes: The number of output classes.
        :param dim: The dimensionality of the embedding space.
        :param depth: The number of transformer layers.
        :param heads: The number of attention heads.
        :param mlp_dim: The dimensionality of the MLP (Multi-Layer Perceptron) in the transformer.
        :param channels: The number of channels in the input image (default is 3 for RGB).
        :param dropout: Dropout rate applied in the MLP.
        :param emb_dropout: Dropout rate applied to the embedded patches.
        """
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
        """
        Forward pass of the Vision Transformer.

        :param img: The input image tensor, shape (batch_size, channels, image_size, image_size).
        :return: The output tensor, shape (batch_size, num_classes).
        """
        p = self.patch_size
        
        # Patchify (keep original semantics)
        x = img.unfold(2, p, p).unfold(3, p, p).reshape(img.shape[0], -1, p * p * img.shape[1])
        x = self.patch_to_embedding(x)

        B, N, D = x.shape

        # Fused cat([cls_token]*B, x) + add pos_embedding via Triton
        if x.is_cuda:
            x_contig = x.contiguous()
            pos_row = self.pos_embedding[0].contiguous()       # [N+1, D]
            cls_vec = self.cls_token[0, 0].contiguous()        # [D]
            out = torch.empty((B, N + 1, D), device=x.device, dtype=x.dtype)

            grid = lambda META: (B, N + 1, triton.cdiv(D, META["BLOCK_D"]))
            _cat_add_pos_kernel[grid](
                x_contig, pos_row, cls_vec, out,
                B, N, D,
                x_contig.stride(0), x_contig.stride(1), x_contig.stride(2),
                pos_row.stride(0), pos_row.stride(1),
                cls_vec.stride(0),
                out.stride(0), out.stride(1), out.stride(2),
            )
            x = out
        else:
            cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embedding

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