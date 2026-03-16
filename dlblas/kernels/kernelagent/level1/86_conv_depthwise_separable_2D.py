import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def _depthwise_conv2d_fwd(
    x_ptr,        # *f32, [N, C, H, W]
    w_ptr,        # *f32, [C, 1, K, K] -> flattened as [C*K*K]
    b_ptr,        # *f32, [C] or dummy
    y_ptr,        # *f32, [N, C, H_OUT, W_OUT]
    N, C, H, W,   # int32
    H_OUT, W_OUT, # int32
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    DILATION: tl.constexpr,
    K: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_tile = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C

    HW_OUT = H_OUT * W_OUT
    offs_hw = pid_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW_OUT

    oh = offs_hw // W_OUT
    ow = offs_hw % W_OUT

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # Convolution as cross-correlation (PyTorch semantics)
    for kh in range(0, K):
        ih = oh * STRIDE - PADDING + kh * DILATION
        in_h_ok = (ih >= 0) & (ih < H)
        for kw in range(0, K):
            iw = ow * STRIDE - PADDING + kw * DILATION
            in_w_ok = (iw >= 0) & (iw < W)
            m = mask_hw & in_h_ok & in_w_ok

            x_offs = (((n * C + c) * H) + ih) * W + iw
            x_vals = tl.load(x_ptr + x_offs, mask=m, other=0.0)

            w_off = c * (K * K) + kh * K + kw
            w_val = tl.load(w_ptr + w_off)
            acc += x_vals * w_val

    if HAS_BIAS:
        b_val = tl.load(b_ptr + c)
        acc += b_val

    y_offs = (((n * C + c) * H_OUT) + oh) * W_OUT + ow
    tl.store(y_ptr + y_offs, acc, mask=mask_hw)


@triton.jit
def _pointwise_1x1_conv_fwd(
    x_ptr,        # *f32, input from depthwise, [N, C_IN, H_OUT, W_OUT]
    w_ptr,        # *f32, weight, [C_OUT, C_IN] row-major
    b_ptr,        # *f32, bias, [C_OUT] or dummy
    y_ptr,        # *f32, output, [N, C_OUT, H_OUT, W_OUT]
    N, C_IN, C_OUT, H_OUT, W_OUT,  # int32
    BM: tl.constexpr,
    BN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    HW = H_OUT * W_OUT
    M = N * HW

    m_offsets = pid_m * BM + tl.arange(0, BM)
    n_offsets = pid_n * BN + tl.arange(0, BN)

    m_mask = m_offsets < M
    n_mask = n_offsets < C_OUT

    # Map linear m_offsets -> (n, hw)
    n_idx = m_offsets // HW
    hw_idx = m_offsets % HW  # equals h * W_OUT + w

    # Base offsets for input/output that exclude channel term
    # NCHW layout: index = n*C*HW + c*HW + hw
    in_base = n_idx * (C_IN * HW) + hw_idx
    out_base = n_idx * (C_OUT * HW) + hw_idx

    acc = tl.zeros([BM, BN], dtype=tl.float32)

    # Small K loop (C_IN)
    for k in range(0, C_IN):
        # Load A: [BM] elements at channel k (stride HW across channel)
        a = tl.load(x_ptr + in_base + k * HW, mask=m_mask, other=0.0)  # [BM]
        # Load B: [BN] weights at channel k across output channels
        b = tl.load(w_ptr + n_offsets * C_IN + k, mask=n_mask, other=0.0)  # [BN]
        acc += a[:, None] * b[None, :]

    if HAS_BIAS:
        bias_tile = tl.load(b_ptr + n_offsets, mask=n_mask, other=0.0)  # [BN]
        acc += bias_tile[None, :]

    # Store: y[n, co, h, w] -> linear index: n*C_OUT*HW + co*HW + hw
    store_offs = out_base[:, None] + n_offsets[None, :] * HW
    tl.store(y_ptr + store_offs, acc, mask=(m_mask[:, None] & n_mask[None, :]))


@triton.jit
def _dw_pw_fused_fwd(
    x_ptr,          # *f32, [N, C_IN, H, W]
    wdw_ptr,        # *f32, [C_IN*K*K] flattened
    bdw_ptr,        # *f32, [C_IN] or dummy
    wpw_ptr,        # *f32, [C_OUT, C_IN] row-major
    bpw_ptr,        # *f32, [C_OUT] or dummy
    y_ptr,          # *f32, [N, C_OUT, H_OUT, W_OUT]
    N, C_OUT, H, W, H_OUT, W_OUT,   # int32
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    DILATION: tl.constexpr,
    K: tl.constexpr,
    C_IN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    HAS_BIAS_DW: tl.constexpr,
    HAS_BIAS_PW: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    HW_OUT = H_OUT * W_OUT
    M = N * HW_OUT

    # tile offsets
    m_offsets = pid_m * BM + tl.arange(0, BM)
    n_offsets = pid_n * BN + tl.arange(0, BN)

    m_mask = m_offsets < M
    n_mask = n_offsets < C_OUT

    # map m -> (n, oh, ow)
    n_idx = m_offsets // HW_OUT
    hw_idx = m_offsets % HW_OUT
    oh = hw_idx // W_OUT
    ow = hw_idx % W_OUT

    # stride-shifted coords
    oh_s = oh * STRIDE - PADDING
    ow_s = ow * STRIDE - PADDING

    acc = tl.zeros([BM, BN], dtype=tl.float32)

    # Precompute panels for B (pointwise weights)
    wpw_panel = wpw_ptr + n_offsets * C_IN

    # For each input channel: compute depthwise conv scalar per lane -> multiply-accumulate into BN outputs
    for ci in tl.static_range(0, C_IN):
        s = tl.zeros([BM], dtype=tl.float32)

        base_in = n_idx * (C_IN * H * W) + ci * (H * W)
        wdw_base = ci * (K * K)

        for kh in tl.static_range(0, K):
            ih = oh_s + kh * DILATION
            in_h_ok = (ih >= 0) & (ih < H)
            ihW = ih * W
            for kw in tl.static_range(0, K):
                iw = ow_s + kw * DILATION
                in_w_ok = (iw >= 0) & (iw < W)
                mask = m_mask & in_h_ok & in_w_ok

                x_offs = base_in + ihW + iw
                x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0)

                w_val = tl.load(wdw_ptr + wdw_base + kh * K + kw)
                s += x_vals * w_val

        if HAS_BIAS_DW:
            s += tl.load(bdw_ptr + ci)

        # Multiply with pointwise weights for channel ci and accumulate over output channels tile
        b_vec = tl.load(wpw_panel + ci, mask=n_mask, other=0.0)  # [BN]
        acc += s[:, None] * b_vec[None, :]

    if HAS_BIAS_PW:
        bias_tile = tl.load(bpw_ptr + n_offsets, mask=n_mask, other=0.0)
        acc += bias_tile[None, :]

    out_base = n_idx * (C_OUT * HW_OUT) + hw_idx
    store_offs = out_base[:, None] + n_offsets[None, :] * HW_OUT
    tl.store(y_ptr + store_offs, acc, mask=(m_mask[:, None] & n_mask[None, :]))


class ModelNew(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation using Triton-accelerated kernels.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Keep nn.Conv2d modules to replicate exact parameter initialization and semantics
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution.
        Falls back to PyTorch ops on non-CUDA tensors; uses Triton kernels on CUDA.
        """
        if not x.is_cuda:
            # CPU or non-CUDA fallback: exact PyTorch semantics
            y = self.depthwise(x)
            y = self.pointwise(y)
            return y

        # Extract parameters
        N, C_in, H, W = x.shape
        K = self.depthwise.kernel_size[0]
        stride = self.depthwise.stride[0]
        padding = self.depthwise.padding[0]
        dilation = self.depthwise.dilation[0]

        # Output spatial dims after depthwise
        H_out = math.floor((H + 2 * padding - dilation * (K - 1) - 1) / stride + 1)
        W_out = math.floor((W + 2 * padding - dilation * (K - 1) - 1) / stride + 1)

        # Prepare buffers and dtypes
        x_c = x.contiguous()
        device = x.device
        compute_dtype = torch.float32

        w_dw = self.depthwise.weight.contiguous()  # [C_in, 1, K, K]
        b_dw = self.depthwise.bias
        has_bias_dw = b_dw is not None
        if has_bias_dw:
            b_dw = b_dw.contiguous()

        # Flatten depthwise weight to [C_in*K*K]
        w_dw_flat = w_dw.view(C_in * K * K).to(compute_dtype)

        # Input to fp32 for compute
        x_fp32 = x_c.to(compute_dtype)
        b_dw_buf = (b_dw.to(compute_dtype) if has_bias_dw else torch.empty(1, device=device, dtype=compute_dtype))

        # Pointwise 1x1 convolution parameters
        C_out = self.pointwise.out_channels
        w_pw = self.pointwise.weight.view(C_out, C_in).contiguous().to(compute_dtype)  # [C_OUT, C_IN]
        b_pw = self.pointwise.bias
        has_bias_pw = b_pw is not None
        b_pw_buf = (b_pw.contiguous().to(compute_dtype) if has_bias_pw else torch.empty(1, device=device, dtype=compute_dtype))

        # Allocate final output (fp32)
        y_out_fp32 = torch.empty((N, C_out, H_out, W_out), device=device, dtype=compute_dtype)

        # Launch fused depthwise + pointwise kernel
        BM, BN = 128, 64
        M = N * H_out * W_out
        grid = (_ceil_div(M, BM), _ceil_div(C_out, BN))
        _dw_pw_fused_fwd[grid](
            x_fp32, w_dw_flat, b_dw_buf, w_pw, b_pw_buf, y_out_fp32,
            N, C_out, H, W, H_out, W_out,
            STRIDE=stride, PADDING=padding, DILATION=dilation,
            K=K, C_IN=C_in, BM=BM, BN=BN,
            HAS_BIAS_DW=has_bias_dw, HAS_BIAS_PW=has_bias_pw,
            num_warps=4, num_stages=2,
        )

        # Cast back to original dtype
        y_out = y_out_fp32.to(dtype=x.dtype)
        return y_out


# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0
dilation = 1

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]