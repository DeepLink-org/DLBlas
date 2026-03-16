import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=2, num_stages=4),
        # Extra candidates to better match large M and N=64
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32,  'BLOCK_K': 16}, num_warps=4, num_stages=4),
    ],
    key=["M", "C_out", "C_in"],
)
@triton.jit
def _pw_conv1x1_kernel(
    x_ptr, wt_ptr, bias_ptr, y_ptr,
    B, C_in, H, W, C_out, M,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    stride_wk, stride_wn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)  # tile along M = B*H*W
    pid_n = tl.program_id(axis=1)  # tile along N = C_out

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < C_out

    HW = H * W
    n_idx = offs_m // HW
    hw_idx = offs_m % HW
    h_idx = hw_idx // W
    w_idx = hw_idx % W

    # Base pointers for current (m) rows
    x_row_base = x_ptr + n_idx * stride_xn + h_idx * stride_xh + w_idx * stride_xw
    y_row_base = y_ptr + n_idx * stride_yn + h_idx * stride_yh + w_idx * stride_yw

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over input channels in chunks
    for k0 in range(0, C_in, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < C_in

        # [BLOCK_K, BLOCK_N] weights tile, keep original dtype to enable Tensor Cores on fp16/bf16
        w_ptrs = wt_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w_tile = tl.load(w_ptrs, mask=(k_mask[:, None] & n_mask[None, :]), other=0)

        # [BLOCK_M, BLOCK_K] input tile, keep original dtype to enable Tensor Cores on fp16/bf16
        x_ptrs = x_row_base[:, None] + k_offs[None, :] * stride_xc
        x_tile = tl.load(x_ptrs, mask=(m_mask[:, None] & k_mask[None, :]), other=0)

        # Accumulate; for fp16/bf16 inputs, this uses HMMA with fp32 accumulation
        acc += tl.dot(x_tile, w_tile)

    if HAS_BIAS:
        b = tl.load(bias_ptr + offs_n, mask=n_mask, other=0)
        # Broadcast add in fp32
        acc += b[None, :].to(tl.float32)

    # Store
    y_ptrs = y_row_base[:, None] + offs_n[None, :] * stride_yc
    tl.store(y_ptrs, acc, mask=(m_mask[:, None] & n_mask[None, :]))


class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation (1x1 Conv) using a Triton kernel on CUDA,
    and falls back to PyTorch's nn.Conv2d otherwise. Semantics match nn.Conv2d with kernel_size=1.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback for non-CUDA tensors or unsupported dtypes
        if (not x.is_cuda) or x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            return self.conv1d(x)

        B, C_in, H, W = x.shape
        C_out = self.conv1d.out_channels

        # Output tensor
        y = torch.empty((B, C_out, H, W), device=x.device, dtype=x.dtype)

        # Prepare weights as (K=C_in, N=C_out) contiguous, match input dtype for fast dot
        wt = self.conv1d.weight.view(C_out, C_in).t().contiguous().to(dtype=x.dtype)
        bias = self.conv1d.bias
        has_bias = bias is not None

        # Strides in elements (PyTorch stride is in elements)
        stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
        stride_yn, stride_yc, stride_yh, stride_yw = y.stride()
        stride_wk, stride_wn = wt.stride()

        M = B * H * W

        # Launch configuration
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C_out, meta['BLOCK_N']))

        _pw_conv1x1_kernel[grid](
            x, wt, bias if has_bias else None, y,
            B, C_in, H, W, C_out, M,
            stride_xn, stride_xc, stride_xh, stride_xw,
            stride_yn, stride_yc, stride_yh, stride_yw,
            stride_wk, stride_wn,
            HAS_BIAS=has_bias,
        )
        return y


# Test code
batch_size = 16
in_channels = 3
out_channels = 64
width = 256
height = 256

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels]