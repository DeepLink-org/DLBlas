import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=4),
        triton.Config({}, num_warps=16, num_stages=2),
    ],
    key=["S", "C"],
)
@triton.jit
def _fused_hswish_relu_softmax_mean_kernel(
    x_ptr,             # *float or *half, shape [N, C, S] (S = D*H*W), contiguous N,C,S layout
    out_ptr,           # *float or *half, shape [N, C]
    N, C, S,           # int32
    stride_n,          # int32, elements between successive n
    stride_c,          # int32, elements between successive c
    stride_out_n,      # int32, elements between successive n in out
    inv_S,             # float32, 1.0 / S
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    if pid_n >= N:
        return

    base_n = pid_n * stride_n
    c_idx = tl.arange(0, BLOCK_C)
    valid_c = c_idx < C

    # Accumulator over spatial positions for each channel
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    inv6 = 1.0 / 6.0
    s_start = 0
    while s_start < S:
        s_idx = s_start + tl.arange(0, BLOCK_S)
        valid_s = s_idx < S
        mask_s = valid_s[None, :]

        # Offsets for 2D tile [C, S_tile]
        offs = base_n + c_idx[:, None] * stride_c + s_idx[None, :]

        # Load in fp32 for stability; mask only along S (C tile == C in launcher)
        x = tl.load(x_ptr + offs, mask=mask_s, other=0.0).to(tl.float32)

        # Fused HardSwish + ReLU: y = max(x, 0) * clamp(x + 3, 0, 6) / 6
        t = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
        y = tl.maximum(x, 0.0) * (t * inv6)

        # Mask invalid spatial lanes with -inf for softmax max-reduction
        y_masked = tl.where(mask_s, y, -float("inf"))

        # Softmax across channels (axis=0) per spatial column
        m = tl.max(y_masked, axis=0)                           # [BLOCK_S]
        expv = tl.exp(y_masked - m[None, :])                   # [BLOCK_C, BLOCK_S]
        sumexp = tl.sum(expv, axis=0)                          # [BLOCK_S]
        p = expv / sumexp[None, :]                             # [BLOCK_C, BLOCK_S]
        p = tl.where(mask_s, p, 0.0)

        # Accumulate probabilities across spatial positions for each channel
        acc += tl.sum(p, axis=1)

        s_start += BLOCK_S

    # Write normalized mean over spatial dims
    out_offs = pid_n * stride_out_n + c_idx
    tl.store(out_ptr + out_offs, acc * inv_S, mask=valid_c)


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def fused_hswish_relu_softmax_mean(x: torch.Tensor) -> torch.Tensor:
    # x: [N, C, D, H, W]
    N, C, D, H, W = x.shape
    S = D * H * W
    x = x.contiguous()

    # Strides in elements for a contiguous [N, C, S] view
    stride_c = S
    stride_n = C * S

    # Output [N, C], same dtype as input
    out = torch.empty((N, C), device=x.device, dtype=x.dtype)
    stride_out_n = C

    # Choose a larger spatial tile to improve arithmetic intensity on Hopper
    if S >= 512:
        BLOCK_S = 512
    elif S >= 256:
        BLOCK_S = 256
    else:
        BLOCK_S = 128
    BLOCK_C = C  # compute softmax across all channels at once

    grid = (N,)
    inv_S = float(1.0 / S)

    _fused_hswish_relu_softmax_mean_kernel[grid](
        x, out,
        N, C, S,
        stride_n, stride_c, stride_out_n,
        inv_S,
        BLOCK_C=BLOCK_C,
        BLOCK_S=BLOCK_S,
    )
    return out


class ModelNew(nn.Module):
    """
    Simple model that performs a 3D convolution, applies HardSwish, ReLU, Softmax, and then calculates the mean.
    Fused with a Triton kernel for post-convolution operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)

    def forward(self, x):
        # Conv stays in PyTorch (highly optimized), post-ops fused in Triton
        x = self.conv(x)
        if x.is_cuda:
            return fused_hswish_relu_softmax_mean(x)
        else:
            # CPU fallback: exact reference ops
            x = torch.nn.functional.hardswish(x)
            x = torch.relu(x)
            x = torch.softmax(x, dim=1)
            x = torch.mean(x, dim=[2, 3, 4])
            return x


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]