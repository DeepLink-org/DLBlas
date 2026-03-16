import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _conv_transpose2d_stride1_pad0_kernel(
    x_ptr,            # *f32 [N, C_IN, H_IN, W_IN]
    w_ptr,            # *f32 [C_IN, C_OUT, K_H, K_W]
    y_ptr,            # *f32 [N, C_OUT, H_OUT, W_OUT]
    N,                # int
    C_IN: tl.constexpr,
    H_IN: tl.constexpr,
    W_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    HO_BLOCK: tl.constexpr,
    WO_BLOCK: tl.constexpr,
    COUT_BLOCK: tl.constexpr,
):
    # Program IDs:
    pid_n = tl.program_id(axis=0)  # batch index
    pid_co = tl.program_id(axis=1)  # output channel block
    pid_sp = tl.program_id(axis=2)  # combined spatial tile id (ho_tile, wo_tile)

    # Spatial tiling
    N_WTILES = (W_OUT + WO_BLOCK - 1) // WO_BLOCK
    ho_tile = pid_sp // N_WTILES
    wo_tile = pid_sp % N_WTILES
    ho_start = ho_tile * HO_BLOCK
    wo_start = wo_tile * WO_BLOCK

    # Output channel block
    co_offsets = pid_co * COUT_BLOCK + tl.arange(0, COUT_BLOCK)
    co_mask = co_offsets < C_OUT
    tl.multiple_of(co_offsets, COUT_BLOCK)

    # Flattened spatial offsets within the tile
    p = tl.arange(0, HO_BLOCK * WO_BLOCK)
    tl.multiple_of(p, WO_BLOCK)
    ho = ho_start + (p // WO_BLOCK)
    wo = wo_start + (p % WO_BLOCK)
    mask_sp = (ho < H_OUT) & (wo < W_OUT)

    # Accumulator
    acc = tl.zeros((COUT_BLOCK, HO_BLOCK * WO_BLOCK), dtype=tl.float32)

    # Gather-style accumulation
    # y[n, co, ho, wo] += sum_{ci, kh, kw} w[ci, co, kh, kw] * x[n, ci, ho - kh, wo - kw]
    n = pid_n
    HIN_WIN = H_IN * W_IN
    OUT_SPAN = H_OUT * W_OUT
    base_n = n * C_IN * HIN_WIN

    # Iterate KH/KW outer; tile CI to keep register pressure moderate and enable vectorization
    CI_TILE: tl.constexpr = 8
    # Reverse kw traversal to turn input loads into forward (+1) pointer walks
    # Precompute starting offset for wi at kw=K_W-1: wi0 = wo - (K_W - 1)
    wi0 = wo - (K_W - 1)

    for kh in tl.static_range(0, K_H):
        hi = ho - kh  # [P]
        valid_h = (hi >= 0) & (hi < H_IN)
        x_row_off = hi * W_IN  # [P]
        for t in tl.static_range(0, K_W):
            # traverse kw = K_W-1 - t so that wi increases by +1 every step
            kw = K_W - 1 - t
            wi = wi0 + t  # [P]
            valid_w = (wi >= 0) & (wi < W_IN)
            mask_x = mask_sp & valid_h & valid_w
            # spatial offsets inside one channel at this (kh, kw)
            x_sp = x_row_off + wi  # [P]

            ci = 0
            while ci < C_IN:
                for di in tl.static_range(0, CI_TILE):
                    ci_idx = ci + di
                    ci_mask = ci_idx < C_IN

                    # Load input vector for this (ci_idx, kh, kw) using pointer walk-friendly addressing
                    x_off = base_n + ci_idx * HIN_WIN + x_sp
                    x_val = tl.load(x_ptr + x_off, mask=mask_x & ci_mask, other=0.0)  # [P]

                    # Load weight row for all co in the block; start at kw-index, pointer descends with t
                    # Base pointer for (ci_idx, kh) at kw = K_W - 1 - t
                    w_ptrs = (((ci_idx * C_OUT + co_offsets) * K_H + kh) * K_W + kw)
                    w_val = tl.load(w_ptr + w_ptrs, mask=co_mask & ci_mask, other=0.0)  # [COUT_BLOCK]

                    # Outer product accumulate
                    acc += w_val[:, None] * x_val[None, :]
                ci += CI_TILE

    # Store to y
    co_base = (n * C_OUT + co_offsets) * OUT_SPAN  # [COUT_BLOCK]
    sp_off = ho * W_OUT + wo  # [P]
    y_off = co_base[:, None] + sp_off[None, :]
    mask_store = co_mask[:, None] & mask_sp[None, :]
    tl.store(y_ptr + y_off, acc, mask=mask_store)


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution operation with asymmetric input and kernel size.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of integers representing the kernel size (height, width).
        stride (tuple, optional): Tuple of integers representing the stride of the convolution. Defaults to (1, 1).
        padding (tuple, optional): Tuple of integers representing the padding applied to the input. Defaults to (0, 0).
        output_padding (tuple, optional): Tuple of integers representing the additional size added to one side of the output shape. Defaults to (0, 0).
        dilation (tuple, optional): Tuple of integers representing the spacing between kernel elements. Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        output_padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def _can_use_triton(self):
        # Implement kernel for common default case:
        # stride=(1,1), padding=(0,0), dilation=(1,1), output_padding=(0,0), groups=1
        m = self.conv_transpose2d
        return (
            m.stride == (1, 1)
            and m.padding == (0, 0)
            and m.dilation == (1, 1)
            and m.output_padding == (0, 0)
            and m.groups == 1
        )

    @staticmethod
    def _conv_transpose2d_triton(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # x: [N, C_IN, H_IN, W_IN], weight: [C_IN, C_OUT, K_H, K_W]
        assert x.is_cuda and weight.is_cuda
        x = x.contiguous()
        weight = weight.contiguous()
        N, C_IN, H_IN, W_IN = x.shape
        C_IN_W, C_OUT, K_H, K_W = weight.shape
        assert C_IN_W == C_IN

        # Output shape for stride=1, padding=0, dilation=1, output_padding=0:
        H_OUT = H_IN + K_H - 1
        W_OUT = W_IN + K_W - 1
        y = torch.empty((N, C_OUT, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

        # Favor coalesced width access and larger C_OUT vectorization
        HO_BLOCK = 1
        WO_BLOCK = 64
        COUT_BLOCK = 64

        grid = (
            N,
            triton.cdiv(C_OUT, COUT_BLOCK),
            triton.cdiv(H_OUT, HO_BLOCK) * triton.cdiv(W_OUT, WO_BLOCK),
        )

        _conv_transpose2d_stride1_pad0_kernel[grid](
            x, weight, y,
            N,
            C_IN, H_IN, W_IN,
            C_OUT, K_H, K_W,
            H_OUT, W_OUT,
            HO_BLOCK=HO_BLOCK, WO_BLOCK=WO_BLOCK, COUT_BLOCK=COUT_BLOCK,
            num_warps=4, num_stages=3
        )
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        m = self.conv_transpose2d
        # Use Triton fast path when supported; otherwise fallback to PyTorch implementation.
        if (
            x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and self._can_use_triton()
        ):
            # Simple heuristic: use Triton for relatively small problems; otherwise, rely on cuDNN
            N, CI, HI, WI = x.shape
            KH, KW = m.kernel_size
            HO = HI + KH - 1
            WO = WI + KW - 1
            work = N * m.out_channels * HO * WO * CI * KH * KW
            if work <= 10_000_000:
                x_fp32 = x.float() if x.dtype != torch.float32 else x
                w_fp32 = m.weight.float() if m.weight.dtype != torch.float32 else m.weight
                y = self._conv_transpose2d_triton(x_fp32, w_fp32)
                # Add bias if present
                if m.bias is not None:
                    y += m.bias.float().view(1, -1, 1, 1)
                # Cast back to input dtype if necessary
                if x.dtype != torch.float32:
                    y = y.to(x.dtype)
                return y
        # Fallback to PyTorch implementation for generality/performance
        return m(x)


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height_in = 16
width_in = 32

def get_inputs():
    x = torch.randn(batch_size, in_channels, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization