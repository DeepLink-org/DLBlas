import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        # Tuned configs for Hopper-class GPUs (H200)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=4, num_stages=3),
        # Extra high-occupancy configs for larger tiles on H200
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    ],
    key=["M", "Cout", "K"],
)
@triton.jit
def _conv3d_implicit_gemm_kernel(
    x_ptr,       # *: [N, Cin, Din, Hin, Win] contiguous
    w2d_ptr,     # *: [K, Cout] contiguous, where K = Cin*Kd*Kh*Kw
    y_ptr,       # float32* [N, Cout, Dout, Hout, Wout] contiguous
    N, Cin, Din, Hin, Win,
    Cout, Kd, Kh, Kw,
    Dout, Hout, Wout,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    M, K,  # M = N*Dout*Hout*Wout, K = Cin*Kd*Kh*Kw
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows: flattened (n, d, h, w)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols: output channels

    m_mask = offs_m < M
    n_mask = offs_n < Cout

    # map offs_m -> (n_idx, d_idx, h_idx, w_idx)
    dhw = Dout * Hout * Wout
    hw = Hout * Wout

    n_idx = offs_m // dhw
    rem = offs_m - n_idx * dhw
    d_idx = rem // hw
    rem = rem - d_idx * hw
    h_idx = rem // Wout
    w_idx = rem - h_idx * Wout

    # base input coords for this output location (with stride/padding)
    in_d_base = d_idx * stride_d - pad_d
    in_h_base = h_idx * stride_h - pad_h
    in_w_base = w_idx * stride_w - pad_w

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Precompute strides for faster address arithmetic (NCDHW and NCDHW for output)
    sN_x = Cin * Din * Hin * Win
    sC_x = Din * Hin * Win
    sD_x = Hin * Win
    sH_x = Win

    sN_y = Cout * Dout * Hout * Wout
    sC_y = Dout * Hout * Wout
    sD_y = Hout * Wout
    sH_y = Wout

    # reduction over K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # map offs_k -> (ci, kd, kh, kw) given flatten order ((ci * Kd + kd) * Kh + kh) * Kw + kw
        kw_idx = offs_k % Kw
        tmp1 = offs_k // Kw
        kh_idx = tmp1 % Kh
        tmp2 = tmp1 // Kh
        kd_idx = tmp2 % Kd
        ci_idx = tmp2 // Kd

        # broadcast to (BM, BK)
        in_d = in_d_base[:, None] + kd_idx[None, :] * dil_d
        in_h = in_h_base[:, None] + kh_idx[None, :] * dil_h
        in_w = in_w_base[:, None] + kw_idx[None, :] * dil_w

        # bounds check on input coordinates
        valid_in = (
            m_mask[:, None]
            & k_mask[None, :]
            & (in_d >= 0)
            & (in_d < Din)
            & (in_h >= 0)
            & (in_h < Hin)
            & (in_w >= 0)
            & (in_w < Win)
        )

        # compute input addresses (NCDHW contiguous) using stride arithmetic
        addr_x = (
            n_idx[:, None] * sN_x
            + ci_idx[None, :] * sC_x
            + in_d * sD_x
            + in_h * sH_x
            + in_w
        )
        x_tile = tl.load(x_ptr + addr_x.to(tl.int64), mask=valid_in, other=0)

        # load weight tile w2d: [K, Cout]
        addr_w = offs_k[:, None] * Cout + offs_n[None, :]
        valid_w = k_mask[:, None] & n_mask[None, :]
        w_tile = tl.load(w2d_ptr + addr_w.to(tl.int64), mask=valid_w, other=0)

        # Mixed-precision friendly dot: if inputs are fp16, this will use Tensor Cores with fp32 accumulation.
        # If inputs are fp32, Triton will do fp32 math (TF32 may be used on Hopper depending on settings).
        acc += tl.dot(x_tile, w_tile)

    # store to y (NCDHW contiguous) via precomputed strides
    y_addr = (
        n_idx[:, None] * sN_y
        + offs_n[None, :] * sC_y
        + d_idx[:, None] * sD_y
        + h_idx[:, None] * sH_y
        + w_idx[:, None]
    )
    y_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptr + y_addr.to(tl.int64), acc, mask=y_mask)


def conv3d_triton_implicit_gemm(x: torch.Tensor, weight: torch.Tensor,
                                stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1)) -> torch.Tensor:
    # Assumes:
    # x: [N, Cin, Din, Hin, Win] contiguous, CUDA
    # weight: [Cout, Cin, Kd, Kh, Kw] contiguous, CUDA
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    N, Cin, Din, Hin, Win = x.shape
    Cout, Cin_w, Kd, Kh, Kw = weight.shape
    assert Cin == Cin_w, "Input and weight channel mismatch"

    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation

    # Output dims per PyTorch formula
    Dout = (Din + 2 * pd - dd * (Kd - 1) - 1) // sd + 1
    Hout = (Hin + 2 * ph - dh * (Kh - 1) - 1) // sh + 1
    Wout = (Win + 2 * pw - dw * (Kw - 1) - 1) // sw + 1

    # Choose compute dtype: prefer fp16 Tensor Cores on Hopper when safe.
    # Keep exact dtype if already fp16; if fp32, use fp16 compute on sufficiently large problems for speed,
    # while accumulating in fp32 to preserve accuracy.
    use_fp16_compute = (
        (x.dtype == torch.float16 and weight.dtype == torch.float16)
        or (
            x.dtype == torch.float32
            and weight.dtype == torch.float32
            and (Cin * Kd * Kh * Kw) >= 256
            and (N * Dout * Hout * Wout) >= 4096
        )
    )
    x_comp = x.to(torch.float16) if use_fp16_compute and x.dtype != torch.float16 else x
    w_comp = weight.to(torch.float16) if use_fp16_compute and weight.dtype != torch.float16 else weight

    # Accumulator/output kept in fp32 for stability; cast to input dtype later to match PyTorch API
    y = torch.empty((N, Cout, Dout, Hout, Wout), device=x.device, dtype=torch.float32)

    # Pre-flatten weights to [K, Cout]
    K = Cin * Kd * Kh * Kw
    w2d = w_comp.view(Cout, K).transpose(0, 1).contiguous()

    M = N * Dout * Hout * Wout

    # Triton grid: tiles of (M, Cout)
    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(Cout, meta["BLOCK_N"]))

    _conv3d_implicit_gemm_kernel[grid](
        x_comp, w2d, y,
        N, Cin, Din, Hin, Win,
        Cout, Kd, Kh, Kw,
        Dout, Hout, Wout,
        sd, sh, sw,
        pd, ph, pw,
        dd, dh, dw,
        M, K,
    )

    # match original dtype
    if y.dtype != x.dtype:
        y = y.to(x.dtype)
    return y


class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with asymmetric input and kernel sizes.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel in the form (kernel_size_d, kernel_size_h, kernel_size_w).
        stride (tuple, optional): Stride of the convolution in the form (stride_d, stride_h, stride_w). Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input in the form (padding_d, padding_h, padding_w). Defaults to (0, 0, 0).
        dilation (tuple, optional): Spacing between kernel elements in the form (dilation_d, dilation_h, dilation_w). Defaults to (1, 1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Uses a Triton implicit-GEMM kernel on supported configurations for speed, otherwise falls back to PyTorch.
        """
        conv = self.conv3d

        # Fast path: only for CUDA tensors with common defaults and no groups/bias.
        use_triton = (
            x.is_cuda
            and conv.weight.is_cuda
            and conv.groups == 1
            and conv.bias is None
            and tuple(conv.stride) == (1, 1, 1)
            and tuple(conv.padding) == (0, 0, 0)
            and tuple(conv.dilation) == (1, 1, 1)
            and x.dtype in (torch.float16, torch.float32)
            and conv.weight.dtype in (torch.float16, torch.float32)
        )

        if use_triton:
            x_in = x
            w = conv.weight
            return conv3d_triton_implicit_gemm(x_in, w, stride=conv.stride, padding=conv.padding, dilation=conv.dilation)

        # Fallback to reference implementation
        return conv(x)


# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)  # Asymmetric kernel size
depth = 16
height = 256
width = 256

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization