import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _pool_gelu_scale_max_kernel(
    x_ptr,               # *[B, F], row-major
    out_ptr,             # *[B]
    stride_x,            # stride between rows in elements
    G,                   # number of pooling groups = F // K
    scale,               # scaling factor (float)
    K: tl.constexpr,     # pool kernel size
    BLOCK_G: tl.constexpr,  # number of groups processed per row (power-of-two >= G)
):
    pid = tl.program_id(0)
    row_ptr = x_ptr + pid * stride_x

    # group offsets
    offs_g = tl.arange(0, BLOCK_G)            # [BLOCK_G]
    mask_g = offs_g < G                       # [BLOCK_G]

    # base pointer for each group's start
    base = row_ptr + offs_g * K

    # compute pooled sums by streaming over K (specialize K==4)
    sums = tl.zeros((BLOCK_G,), dtype=tl.float32)
    if K == 4:
        v0 = tl.load(base + 0, mask=mask_g, other=0.0)
        v1 = tl.load(base + 1, mask=mask_g, other=0.0)
        v2 = tl.load(base + 2, mask=mask_g, other=0.0)
        v3 = tl.load(base + 3, mask=mask_g, other=0.0)
        sums = v0.to(tl.float32) + v1.to(tl.float32) + v2.to(tl.float32) + v3.to(tl.float32)
    else:
        for kk in range(0, K):
            vals = tl.load(base + kk, mask=mask_g, other=0.0)
            sums += vals.to(tl.float32)

    # choose max or min via a single reduction using sign trick
    s = scale
    sign = tl.where(s >= 0, 1.0, -1.0)
    signed = sums * sign
    signed = tl.where(mask_g, signed, -float("inf"))
    best_signed = tl.max(signed, axis=0)
    sel_sum = best_signed * sign

    invK = 1.0 / K
    sel_mean = sel_sum * invK

    # exact GELU via erf on selected mean only
    inv_sqrt2 = 0.7071067811865475
    gelu_sel = 0.5 * sel_mean * (1.0 + tl.math.erf(sel_mean * inv_sqrt2))

    res = gelu_sel * s
    tl.store(out_ptr + pid, res)


class ModelNew(nn.Module):
    """
    A model implementing the pattern "Matmul_AvgPool_GELU_Scale_Max".
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.avg_pool = nn.AvgPool1d(kernel_size=pool_kernel_size)
        self.scale_factor = scale_factor
        # cache kernel size as int
        self._pool_k = int(pool_kernel_size)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        x = self.matmul(x)
        # Fast Triton path on CUDA
        if x.is_cuda:
            x = x.contiguous()
            B, F = x.shape
            K = self._pool_k
            # number of pooling groups (AvgPool1d with stride=kernel and no padding)
            G = F // K
            # handle degenerate case safely via fallback
            if G <= 0:
                x = self.avg_pool(x.unsqueeze(1)).squeeze(1)
                x = torch.nn.functional.gelu(x)
                x = x * self.scale_factor
                x = torch.max(x, dim=1).values
                return x
            # choose BLOCK_G as next power-of-two >= G
            BLOCK_G = 1 << (G - 1).bit_length()
            out = torch.empty((B,), device=x.device, dtype=x.dtype)
            grid = (B,)
            _pool_gelu_scale_max_kernel[grid](
                x, out,
                x.stride(0),
                G,
                float(self.scale_factor),
                K=K,
                BLOCK_G=BLOCK_G,
                num_warps=4,
                num_stages=2,
            )
            return out
        # CPU fallback: original reference pipeline
        x = self.avg_pool(x.unsqueeze(1)).squeeze(1)
        x = torch.nn.functional.gelu(x)
        x = x * self.scale_factor
        x = torch.max(x, dim=1).values
        return x


batch_size = 128
in_features = 512
out_features = 256
pool_kernel_size = 4
scale_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]