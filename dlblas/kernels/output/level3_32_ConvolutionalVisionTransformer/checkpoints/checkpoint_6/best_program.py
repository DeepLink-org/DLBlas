# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def linear_proj_kernel(
    x_ptr, w_ptr, b_ptr, output_ptr,
    B, D, embed_dim,
    stride_x_batch, stride_w_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B:
        return

    off_x = pid * stride_x_batch
    off_out = pid * embed_dim

    for j in range(0, embed_dim):
        off_w = j * stride_w_row
        acc = 0.0
        for i in range(0, D, BLOCK_SIZE):
            off = i + tl.arange(0, BLOCK_SIZE)
            mask = off < D
            x = tl.load(x_ptr + off_x + off, mask=mask, other=0.0)
            w = tl.load(w_ptr + off_w + off, mask=mask, other=0.0)
            acc += tl.sum(x * w)
        if b_ptr is not None:
            b = tl.load(b_ptr + j)
            acc += b
        tl.store(output_ptr + off_out + j, acc)

class ModelNew(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6, 
                 mlp_ratio=4.0, patch_size=4, in_channels=3):
        super(ModelNew, self).__init__()
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten()
        self.linear_proj = nn.Linear(embed_dim * (32 // patch_size) * (32 // patch_size), embed_dim)
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
        
        # Replace nn.Linear with Triton kernel
        D = x.size(1)
        weight = self.linear_proj.weight
        bias = self.linear_proj.bias
        output = torch.empty(B, self.linear_proj.out_features, device=x.device, dtype=x.dtype)
        grid = (B,)
        linear_proj_kernel[grid](
            x, weight, bias, output,
            B, D, self.linear_proj.out_features,
            x.stride(0), weight.stride(0),
            BLOCK_SIZE=128
        )
        x = output

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)

        for layer in self.transformer_layers:
            x = layer(x)

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