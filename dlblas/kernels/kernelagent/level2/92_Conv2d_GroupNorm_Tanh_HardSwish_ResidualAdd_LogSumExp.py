import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_tanh_hswish_residual_lse(
    x_conv_ptr,     # *f32 [N, C, H, W] contiguous
    x_norm_ptr,     # *f32 [N, C, H, W] contiguous
    out_ptr,        # *f32 [N, 1, H, W] contiguous (stored flattened as N*H*W)
    N, C, H, W,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    HW = H * W
    total = N * HW

    # Guard: avoid OOB if grid is oversized (safety, though we set exact grid)
    off_out = pid
    mask_pid = pid < total

    # Decode (n, h, w)
    n = pid // HW
    q = pid % HW

    # Base offset for channel loop
    base = n * C * HW + q

    # Pass 1: compute maximum across channels for (n, h, w)
    m = -1.0e30
    c0 = 0
    arange_c = tl.arange(0, BLOCK_C)
    while c0 < C:
        idx = c0 + arange_c
        ch_mask = (idx < C) & mask_pid
        offs = base + idx * HW

        xc = tl.load(x_conv_ptr + offs, mask=ch_mask, other=-1.0e30).to(tl.float32)
        xn = tl.load(x_norm_ptr + offs, mask=ch_mask, other=0.0).to(tl.float32)

        # tanh via sigmoid for Triton (2*sigmoid(2x) - 1)
        t = 2.0 / (1.0 + tl.exp(-2.0 * xn)) - 1.0
        # hardswish: x * clamp(x + 3, 0, 6) / 6
        relu6 = tl.minimum(t + 3.0, 6.0)
        relu6 = tl.maximum(relu6, 0.0)
        hsw = t * (relu6 * (1.0 / 6.0))

        y = xc + hsw
        tile_max = tl.max(y, axis=0)
        m = tl.maximum(m, tile_max)
        c0 += BLOCK_C

    # Pass 2: compute sum(exp(y - m)) across channels
    sum_exp = 0.0
    c0 = 0
    while c0 < C:
        idx = c0 + arange_c
        ch_mask = (idx < C) & mask_pid
        offs = base + idx * HW

        xc = tl.load(x_conv_ptr + offs, mask=ch_mask, other=0.0).to(tl.float32)
        xn = tl.load(x_norm_ptr + offs, mask=ch_mask, other=0.0).to(tl.float32)

        t = 2.0 / (1.0 + tl.exp(-2.0 * xn)) - 1.0
        relu6 = tl.minimum(t + 3.0, 6.0)
        relu6 = tl.maximum(relu6, 0.0)
        hsw = t * (relu6 * (1.0 / 6.0))

        y = xc + hsw
        e = tl.exp(y - m)
        e = tl.where(ch_mask, e, 0.0)
        sum_exp += tl.sum(e, axis=0)
        c0 += BLOCK_C

    lse = tl.log(sum_exp) + m
    tl.store(out_ptr + off_out, lse, mask=mask_pid)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, applies Group Normalization, Tanh, HardSwish, 
    Residual Addition, and LogSumExp.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()

    def forward(self, x):
        # Convolution
        x_conv = self.conv(x)
        # Group Normalization
        x_norm = self.group_norm(x_conv)

        # CPU fallback path
        if not x_conv.is_cuda:
            x_tanh = self.tanh(x_norm)
            x_hard_swish = self.hard_swish(x_tanh)
            x_res = x_conv + x_hard_swish
            x_logsumexp = torch.logsumexp(x_res, dim=1, keepdim=True)
            return x_logsumexp

        # Triton fused kernel: tanh -> hardswish -> residual add -> logsumexp over channels
        N, C, H, W = x_conv.shape
        x_conv_c = x_conv.contiguous()
        x_norm_c = x_norm.contiguous()
        out = torch.empty((N, 1, H, W), device=x_conv.device, dtype=x_conv.dtype)

        # Choose BLOCK_C as next power of two up to 128
        if C <= 1:
            block_c = 1
        else:
            block_c = 1 << (int(math.ceil(math.log2(C))))
            block_c = min(128, max(1, block_c))

        grid = (N * H * W,)
        _fused_tanh_hswish_residual_lse[grid](
            x_conv_c, x_norm_c, out,
            N, C, H, W,
            BLOCK_C=block_c,
            num_warps=2,
            num_stages=2,
        )
        return out

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
groups = 8

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups]