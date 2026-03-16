import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _fused_tanh_scale_bias_maxpool2d(
    x_ptr,                # *f32 [B, C, H, W]
    bias_ptr,             # *f32 [C]
    y_ptr,                # *f32 [B, C, Hpo, Wpo]
    B, C, H, W,           # input dims
    HPO, WPO,             # pooled output dims
    STRIDE_B, STRIDE_C, STRIDE_H, STRIDE_W,       # input strides (in elements)
    O_STRIDE_B, O_STRIDE_C, O_STRIDE_H, O_STRIDE_W,  # output strides (in elements)
    scale,                # float scaling factor
    POOL_K: tl.constexpr, # pooling kernel size (assume stride=POOL_K, padding=0, ceil_mode=False)
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    b = pid_bc // C
    c = pid_bc % C

    # Tiled coordinates in pooled space
    oh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]

    mask_hw = (oh < HPO) & (ow < WPO)

    # Base pointers for current (b, c) plane
    x_base = x_ptr + b * STRIDE_B + c * STRIDE_C
    y_base = y_ptr + b * O_STRIDE_B + c * O_STRIDE_C

    # Accumulator for max-pooling
    acc = tl.full((BLOCK_H, BLOCK_W), -float("inf"), tl.float32)

    # Precompute start indices in input feature map for each pooled output position
    ih0 = oh * POOL_K
    iw0 = ow * POOL_K
    ih0s = ih0 * STRIDE_H
    iw0s = iw0 * STRIDE_W

    # Iterate over POOL_K x POOL_K window
    for kh in range(POOL_K):
        ih_off = ih0s + kh * STRIDE_H
        for kw in range(POOL_K):
            ptrs = x_base + ih_off + iw0s + kw * STRIDE_W
            vals = tl.load(ptrs, mask=mask_hw, other=0.0).to(tl.float32)

            # Numerically stable tanh:
            # tanh(x) = sign(x) * (1 - e)/(1 + e), where e = exp(-2*|x|)
            abs_x = tl.abs(vals)
            e = tl.exp(-2.0 * abs_x)
            t = (1.0 - e) / (1.0 + e)
            sign = tl.where(vals >= 0, 1.0, -1.0)
            tanh_x = t * sign

            # Scale then max-reduce for pooling
            v = tanh_x * scale
            acc = tl.maximum(acc, v)

    # Add per-channel bias after max-pooling (equivalent since bias is constant per channel)
    bias_val = tl.load(bias_ptr + c).to(tl.float32)
    acc = acc + bias_val

    # Store results
    out_ptrs = y_base + oh * O_STRIDE_H + ow * O_STRIDE_W
    tl.store(out_ptrs, acc, mask=mask_hw)


class ModelNew(nn.Module):
    """
    A model that performs a convolution, applies tanh, scaling, adds a bias term, and then max-pools.
    Fused Triton kernel computes tanh + scale + bias + max-pooling on CUDA for speed.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = float(scaling_factor)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)
        self._pool_k = pool_kernel_size if isinstance(pool_kernel_size, int) else pool_kernel_size[0]

    def forward(self, x):
        # Convolution
        x = self.conv(x)

        # CUDA fast path: fused tanh + scale + bias + max-pool
        if x.is_cuda:
            B, C, H, W = x.shape
            K = self._pool_k
            # Output dims with stride=K, padding=0, ceil_mode=False
            HPO = H // K
            WPO = W // K

            y = torch.empty((B, C, HPO, WPO), device=x.device, dtype=x.dtype)

            # Flatten bias to [C]
            bias_flat = self.bias.view(C).contiguous()

            # Strides in elements
            sb, sc, sh, sw = x.stride()
            ob, oc, oh, ow = y.stride()

            # Use square tiles to minimize masked compute on small pooled maps
            BLOCK_H = 16
            BLOCK_W = 16
            grid = (B * C, triton.cdiv(HPO, BLOCK_H), triton.cdiv(WPO, BLOCK_W))

            _fused_tanh_scale_bias_maxpool2d[grid](
                x, bias_flat, y,
                B, C, H, W,
                HPO, WPO,
                sb, sc, sh, sw,
                ob, oc, oh, ow,
                self.scaling_factor,
                POOL_K=K,
                BLOCK_H=BLOCK_H,
                BLOCK_W=BLOCK_W,
                num_warps=4,
                num_stages=2,
            )
            return y
        else:
            # CPU fallback: identical semantics
            x = torch.tanh(x)
            x = x * self.scaling_factor
            x = x + self.bias
            x = self.max_pool(x)
            return x


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]