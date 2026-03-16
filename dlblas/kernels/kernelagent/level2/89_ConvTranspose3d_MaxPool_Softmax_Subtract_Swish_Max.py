import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_C": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_C": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 512, "BLOCK_C": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_C": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 512, "BLOCK_C": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 1024, "BLOCK_C": 16}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 1024, "BLOCK_C": 32}, num_warps=8, num_stages=3),
    ],
    key=["C"],
)
@triton.jit
def _softmax_sub_swish_max_kernel(
    x_ptr,            # *f32, input tensor [B, C, D, H, W]
    sub_ptr,          # *f32, subtract vector [C]
    out_ptr,          # *f32, output tensor [B, D, H, W]
    B, C, D, H, W,    # int32 sizes
    BLOCK_M: tl.constexpr,  # number of (n,d,h,w) positions per program
    BLOCK_C: tl.constexpr,  # channel tile
):
    pid = tl.program_id(axis=0)

    # Total number of spatial positions per sample and altogether
    sC = D * H * W         # elements per channel map (stride along channel)
    M = B * sC             # total positions across batch and spatial dims
    batch_span = C * sC    # number of elements per batch

    # Indices of positions this program handles
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    # Map linear position to (n, inner_m)
    n = offs_m // sC
    inner_m = offs_m % sC
    base = n * batch_span + inner_m  # base offset for channel 0 for each position

    neg_inf = -float('inf')

    # Fast path: if all channels fit into one tile, do everything in a single pass.
    if C <= BLOCK_C:
        c_idx = tl.arange(0, BLOCK_C)
        c_mask = c_idx < C
        offs = base[None, :] + c_idx[:, None] * sC
        load_mask = c_mask[:, None] & m_mask[None, :]
        x_vals = tl.load(x_ptr + offs, mask=load_mask, other=neg_inf, cache_modifier=".ca").to(tl.float32)

        # stable softmax
        m_row = tl.max(x_vals, axis=0)
        ex = tl.exp(x_vals - m_row[None, :])
        s = tl.sum(ex, axis=0)
        inv_s = 1.0 / s

        sub_vals = tl.load(sub_ptr + c_idx, mask=c_mask, other=0.0, cache_modifier=".ca").to(tl.float32)
        y = ex * inv_s[None, :]  # softmax
        z = y - sub_vals[:, None]

        sig = 1.0 / (1.0 + tl.exp(-z))
        s_val = sig * z
        s_val = tl.where(load_mask, s_val, neg_inf)
        out_max = tl.max(s_val, axis=0)
        tl.store(out_ptr + offs_m, out_max, mask=m_mask)
        return

    # Two-pass streaming softmax (general path)
    m = tl.full((BLOCK_M,), neg_inf, dtype=tl.float32)
    s = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Pass 1: compute per-(B,d,h,w) max and normalizer
    for c0 in range(0, C, BLOCK_C):
        c_idx = c0 + tl.arange(0, BLOCK_C)
        c_mask = c_idx < C
        offs = base[None, :] + c_idx[:, None] * sC
        load_mask = c_mask[:, None] & m_mask[None, :]
        x_vals = tl.load(x_ptr + offs, mask=load_mask, other=neg_inf, cache_modifier=".cg").to(tl.float32)
        tile_max = tl.max(x_vals, axis=0)
        m_new = tl.maximum(m, tile_max)
        scale = tl.exp(m - m_new)
        ex_block = tl.exp(x_vals - m_new[None, :])
        s = s * scale + tl.sum(ex_block, axis=0)
        m = m_new

    # Avoid NaNs on masked lanes
    m = tl.where(m_mask, m, 0.0)
    s = tl.where(m_mask, s, 1.0)
    inv_s = 1.0 / s

    # Pass 2: compute fused: softmax -> subtract -> swish; then max over channels
    out_max = tl.full((BLOCK_M,), neg_inf, dtype=tl.float32)
    for c0 in range(0, C, BLOCK_C):
        c_idx = c0 + tl.arange(0, BLOCK_C)
        c_mask = c_idx < C

        offs = base[None, :] + c_idx[:, None] * sC
        load_mask = c_mask[:, None] & m_mask[None, :]
        x_vals = tl.load(x_ptr + offs, mask=load_mask, other=neg_inf, cache_modifier=".cg").to(tl.float32)

        # softmax probs using computed m and inv_s
        y = tl.exp(x_vals - m[None, :]) * inv_s[None, :]

        # load subtract vector for this channel tile
        sub_vals = tl.load(sub_ptr + c_idx, mask=c_mask, other=0.0, cache_modifier=".ca").to(tl.float32)
        z = y - sub_vals[:, None]  # broadcast over positions

        # swish: sigmoid(z) * z
        sig = 1.0 / (1.0 + tl.exp(-z))
        s_val = sig * z

        # mask out invalid lanes to avoid affecting max
        s_val = tl.where(load_mask, s_val, neg_inf)
        tile_max = tl.max(s_val, axis=0)
        out_max = tl.maximum(out_max, tile_max)

    # Store output [B, D, H, W], which is linearized by offs_m
    tl.store(out_ptr + offs_m, out_max, mask=m_mask)


def fused_softmax_sub_swish_max(x: torch.Tensor, sub: torch.Tensor) -> torch.Tensor:
    """
    Fused implementation of:
        y = softmax(x, dim=1)
        z = y - sub.view(1, C, 1, 1, 1)
        s = sigmoid(z) * z
        out = max(s, dim=1).values
    x: [B, C, D, H, W], float32 CUDA tensor
    sub: [C], float32 CUDA tensor
    returns out: [B, D, H, W], float32 CUDA tensor
    """
    # Fallback for non-CUDA or unsupported dtype to guarantee correctness
    if (not x.is_cuda) or (x.dtype != torch.float32) or (sub.dtype != torch.float32):
        y = torch.softmax(x, dim=1)
        z = y - sub.view(1, -1, 1, 1, 1)
        s = torch.sigmoid(z) * z
        return torch.max(s, dim=1)[0]

    B, C, D, H, W = x.shape
    x = x.contiguous()
    sub = sub.contiguous().to(dtype=x.dtype, device=x.device)
    out = torch.empty((B, D, H, W), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(B * D * H * W, meta["BLOCK_M"]),)

    _softmax_sub_swish_max_kernel[grid](
        x, sub, out,
        B, C, D, H, W,
    )
    return out


class ModelNew(nn.Module):
    """
    A model that performs a sequence of operations:
        - ConvTranspose3d
        - MaxPool3d
        - Softmax
        - Subtract
        - Swish
        - Max
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))  # element-wise across channels

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        # Fused: softmax(dim=1) -> subtract -> swish -> max over channels
        x = fused_softmax_sub_swish_max(x, self.subtract)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
pool_stride = 2
pool_padding = 0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding]