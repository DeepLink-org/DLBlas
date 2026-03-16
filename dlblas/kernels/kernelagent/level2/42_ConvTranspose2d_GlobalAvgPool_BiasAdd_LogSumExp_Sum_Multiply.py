import torch
import torch.nn as nn
import triton
import triton.language as tl


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


@triton.jit
def _fused_mean_bias_lse(
    x_ptr,           # float32[N, C, H, W] - contiguous NCHW
    bias_ptr,        # float32[C, 1, 1]
    out_ptr,         # float32[N]
    N, C, H, W,
    stride_n, stride_c,
    bias_stride_c,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid
    if n >= N:
        return

    HW = H * W
    inv_hw = 1.0 / HW
    n_base = n * stride_n

    NEG_INF = -float("inf")
    m = tl.full((), NEG_INF, dtype=tl.float32)
    s = tl.zeros((), dtype=tl.float32)

    c_arange = tl.arange(0, BLOCK_C)

    for c_start in range(0, C, BLOCK_C):
        c_idx = c_start + c_arange
        c_mask = c_idx < C

        sum_c = tl.zeros((BLOCK_C,), dtype=tl.float32)

        base_c = n_base + c_idx * stride_c
        ptrs_base = base_c[:, None]

        hw_arange = tl.arange(0, BLOCK_HW)
        offs_hw = hw_arange
        hw_mask = offs_hw < HW
        ptrs = ptrs_base + offs_hw[None, :]
        load_mask = c_mask[:, None] & hw_mask[None, :]
        tile = tl.load(x_ptr + ptrs, mask=load_mask, other=0.0, cache_modifier=".cg")

        for hw_start in range(BLOCK_HW, HW, BLOCK_HW):
            sum_c += tl.sum(tile, axis=1)
            offs_hw = hw_start + hw_arange
            hw_mask = offs_hw < HW
            ptrs = ptrs_base + offs_hw[None, :]
            load_mask = c_mask[:, None] & hw_mask[None, :]
            tile = tl.load(x_ptr + ptrs, mask=load_mask, other=0.0, cache_modifier=".cg")

        sum_c += tl.sum(tile, axis=1)

        mean_c = sum_c * inv_hw
        b = tl.load(bias_ptr + c_idx * bias_stride_c, mask=c_mask, other=0.0, cache_modifier=".ca")
        v = mean_c + b
        v = tl.where(c_mask, v, NEG_INF)

        tile_max = tl.max(v, axis=0)
        m2 = tl.maximum(m, tile_max)
        s = s * tl.exp(m - m2) + tl.sum(tl.exp(v - m2), axis=0)
        m = m2

    lse = tl.log(s) + m
    tl.store(out_ptr + n, 10.0 * lse)


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, global average pooling, adds a bias, applies log-sum-exp, sum, and multiplication.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)

        if not x.is_cuda:
            x = torch.mean(x, dim=(2, 3), keepdim=True)
            x = x + self.bias
            x = torch.logsumexp(x, dim=1, keepdim=True)
            x = torch.sum(x, dim=(2, 3))
            x = x * 10.0
            return x

        y = x.contiguous()
        N, C, H, W = y.shape
        HW = H * W

        out = torch.empty((N,), device=y.device, dtype=torch.float32)

        BLOCK_C = min(64, _next_pow2(C))
        BLOCK_HW = min(1024, _next_pow2(HW))

        tile_work = BLOCK_C * BLOCK_HW
        num_warps = 8 if tile_work >= 8192 else 4
        num_stages = 5 if BLOCK_HW >= 512 else 4

        grid = (N,)
        _fused_mean_bias_lse[grid](
            y, self.bias, out,
            N, C, H, W,
            y.stride(0), y.stride(1),
            self.bias.stride(0),
            BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW,
            num_warps=num_warps, num_stages=num_stages,
        )

        return out.view(N, 1)


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]