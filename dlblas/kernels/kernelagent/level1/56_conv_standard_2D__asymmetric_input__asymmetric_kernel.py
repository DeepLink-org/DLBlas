import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small-K friendly (common for small kernels and few input channels)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=8, num_stages=3),

        # Balanced tiles
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),

        # Wider N tiles
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),

        # Taller M tiles
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),

        # Large square tiles
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),

        # Skewed tiles
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=3),

        # Deeper K tiles for Hopper/L2 reuse
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=5),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _conv2d_implicit_gemm_kernel(
    x_ptr,        # *[B, Ci, Hi, Wi]
    w_ptr,        # *[Co, Ci, Kh, Kw]
    b_ptr,        # *[Co] (optional, only if HAS_BIAS)
    y_ptr,        # *[B, Co, Ho, Wo]
    B: tl.constexpr, Ci: tl.constexpr, Hi: tl.constexpr, Wi: tl.constexpr,
    Co: tl.constexpr, Kh: tl.constexpr, Kw: tl.constexpr,
    Ho: tl.constexpr, Wo: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    dil_h: tl.constexpr, dil_w: tl.constexpr,
    M, N, K,
    HAS_BIAS: tl.constexpr,
    OUT_FP16: tl.constexpr,  # whether to store in fp16
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Hints for better codegen on Hopper
    tl.multiple_of(offs_k, 16)
    tl.multiple_of(offs_n, 16)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_m_bc = mask_m[:, None]

    # Decode m -> (b, ho, wo)
    hw_total = Ho * Wo
    b_idx = offs_m // hw_total
    hw = offs_m % hw_total
    ho_idx = hw // Wo
    wo_idx = hw % Wo

    # Precompute input base coordinates for each m-lane
    h_base = ho_idx * stride_h - pad_h
    w_base = wo_idx * stride_w - pad_w

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KhKw = Kh * Kw
    HiWi = Hi * Wi

    # Precompute parts independent of k-iteration
    x_off_batch = (b_idx[:, None] * (Ci * HiWi)).to(tl.int32)        # [BM, 1]
    w_off_oc_base = (offs_n[None, :] * (Ci * KhKw)).to(tl.int32)     # [1, BN]

    # Software-pipelined K loop (double-buffered loads)
    k_iter = 0

    # -- Preload first tile --
    k_idx0 = k_iter + offs_k
    mask_k0 = k_idx0 < K

    ci0 = k_idx0 // KhKw
    rem0 = k_idx0 % KhKw
    ky0 = rem0 // Kw
    kx0 = rem0 - ky0 * Kw  # faster than modulo

    h_in0 = h_base[:, None] + ky0[None, :] * dil_h
    w_in0 = w_base[:, None] + kx0[None, :] * dil_w

    in_h0 = (h_in0 >= 0) & (h_in0 < Hi)
    in_w0 = (w_in0 >= 0) & (w_in0 < Wi)
    mask_x0 = (in_h0 & in_w0) & mask_m_bc & mask_k0[None, :]

    x_off_ci0 = (ci0[None, :] * HiWi).to(tl.int32)
    x_off_hw0 = (h_in0 * Wi + w_in0).to(tl.int32)
    x_off0 = x_off_batch + x_off_ci0 + x_off_hw0
    x_tile0 = tl.load(x_ptr + x_off0, mask=mask_x0, other=0.0)

    w_ck0 = (ci0 * KhKw + ky0 * Kw + kx0).to(tl.int32)
    w_off0 = w_off_oc_base + w_ck0[:, None]
    mask_w0 = mask_k0[:, None] & mask_n[None, :]
    w_tile0 = tl.load(w_ptr + w_off0, mask=mask_w0, other=0.0)

    k_iter += BLOCK_K

    while k_iter < K:
        # Preload next tile
        k_idx1 = k_iter + offs_k
        mask_k1 = k_idx1 < K

        ci1 = k_idx1 // KhKw
        rem1 = k_idx1 % KhKw
        ky1 = rem1 // Kw
        kx1 = rem1 - ky1 * Kw

        h_in1 = h_base[:, None] + ky1[None, :] * dil_h
        w_in1 = w_base[:, None] + kx1[None, :] * dil_w

        in_h1 = (h_in1 >= 0) & (h_in1 < Hi)
        in_w1 = (w_in1 >= 0) & (w_in1 < Wi)
        mask_x1 = (in_h1 & in_w1) & mask_m_bc & mask_k1[None, :]

        x_off_ci1 = (ci1[None, :] * HiWi).to(tl.int32)
        x_off_hw1 = (h_in1 * Wi + w_in1).to(tl.int32)
        x_off1 = x_off_batch + x_off_ci1 + x_off_hw1
        x_tile1 = tl.load(x_ptr + x_off1, mask=mask_x1, other=0.0)

        w_ck1 = (ci1 * KhKw + ky1 * Kw + kx1).to(tl.int32)
        w_off1 = w_off_oc_base + w_ck1[:, None]
        mask_w1 = mask_k1[:, None] & mask_n[None, :]
        w_tile1 = tl.load(w_ptr + w_off1, mask=mask_w1, other=0.0)

        # Compute on previously loaded tile while next loads are in flight
        acc += tl.dot(x_tile0, w_tile0, out_dtype=tl.float32)

        # Rotate buffers
        x_tile0 = x_tile1
        w_tile0 = w_tile1

        k_iter += BLOCK_K

    # Final compute for the last preloaded tile
    acc += tl.dot(x_tile0, w_tile0, out_dtype=tl.float32)

    # Add bias if present
    if HAS_BIAS:
        bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc = acc + bias_vals[None, :]

    # Store Y: ((b * Co + oc) * Ho + ho) * Wo + wo
    y_off = ((b_idx[:, None] * Co + offs_n[None, :]) * Ho + ho_idx[:, None]) * Wo + wo_idx[:, None]
    y_off = y_off.to(tl.int32)
    mask_y = mask_m[:, None] & mask_n[None, :]

    if OUT_FP16:
        tl.store(y_ptr + y_off, acc.to(tl.float16), mask=mask_y)
    else:
        tl.store(y_ptr + y_off, acc.to(tl.float32), mask=mask_y)


def _conv2d_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                   stride: tuple[int, int], padding: tuple[int, int], dilation: tuple[int, int]) -> torch.Tensor:
    # Only handle groups=1 path via this Triton kernel; other cases fall back to F.conv2d in the forward method.
    B, Ci, Hi, Wi = x.shape
    Co, Ci_w, Kh, Kw = weight.shape
    assert Ci == Ci_w, "Input channels must match weight's in_channels for groups=1."

    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    # Output dims
    Ho = (Hi + 2 * ph - dh * (Kh - 1) - 1) // sh + 1
    Wo = (Wi + 2 * pw - dw * (Kw - 1) - 1) // sw + 1

    # Ensure contiguous
    x_c = x.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous() if bias is not None else None

    # Allocate output
    y = torch.empty((B, Co, Ho, Wo), device=x.device, dtype=x.dtype)

    # Matmul dims
    M = B * Ho * Wo
    N = Co
    K = Ci * Kh * Kw

    # Kernel launch
    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    out_fp16 = x.dtype == torch.float16
    _conv2d_implicit_gemm_kernel[grid](
        x_c, w_c, (y if b_c is None else b_c), y,
        B, Ci, Hi, Wi, Co, Kh, Kw, Ho, Wo,
        sh, sw, ph, pw, dh, dw,
        M, N, K,
        HAS_BIAS=(b_c is not None),
        OUT_FP16=out_fp16,
    )

    return y


class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with asymmetric input and kernel sizes.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of two integers representing the height and width of the convolution kernel.
        stride (tuple, optional): Tuple of two integers representing the stride in the height and width dimensions. Defaults to (1, 1).
        padding (tuple, optional): Tuple of two integers representing the padding in the height and width dimensions. Defaults to (0, 0).
        dilation (tuple, optional): Tuple of two integers representing the dilation in the height and width dimensions. Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Fast path: custom Triton kernel for common case (CUDA + groups=1)
        if x.is_cuda and self.conv2d.groups == 1 and x.dtype in (torch.float16, torch.float32):
            try:
                return _conv2d_triton(
                    x,
                    self.conv2d.weight,
                    self.conv2d.bias,
                    self.conv2d.stride,
                    self.conv2d.padding,
                    self.conv2d.dilation,
                )
            except Exception:
                # Fallback to PyTorch if anything goes wrong
                return self.conv2d(x)
        else:
            # General fallback (covers CPU, groups>1, unusual dtypes)
            return self.conv2d(x)

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5)  # Asymmetric kernel
height = 256
width = 128  # Asymmetric input dimensions

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization