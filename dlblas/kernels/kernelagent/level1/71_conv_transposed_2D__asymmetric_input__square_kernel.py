import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,   'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,   'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128,  'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128,  'BLOCK_K': 32}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,   'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256,  'BLOCK_K': 32}, num_warps=8, num_stages=4),
        # Added larger tiles and deeper pipelines for H200
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256,  'BLOCK_K': 32}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128,  'BLOCK_K': 32}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256,  'BLOCK_K': 32}, num_warps=8, num_stages=6),
        # Allow a wider K-chunk for larger Cin cases
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128,  'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,   'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['N', 'Cin', 'Cout', 'H_out', 'W_out', 'K'],
)
@triton.jit
def _convtransp2d_stride1_pad0_groups1_kernel(
    x_ptr,         # * (N, Cin, H, W)
    w_ptr,         # * (Cout, Cin, K, K) -- rotated weight: flip(spatial) + permute(out,in,kh,kw)
    bias_ptr,      # * (Cout,) or dummy
    y_ptr,         # * (N, Cout, H_out, W_out)
    N, Cin, H, W,
    Cout,
    K: tl.constexpr,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Tile ids
    pid_m = tl.program_id(0)  # rows: N * H_out * W_out
    pid_n = tl.program_id(1)  # cols: Cout

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    rows = row_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    cols = col_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    total_rows = N * H_out * W_out
    mask_m = rows < total_rows
    mask_n = cols < Cout

    # Hints for codegen
    tl.max_contiguous(rows, BLOCK_M)
    tl.max_contiguous(cols, BLOCK_N)

    # Map rows -> (n, h_out, w_out)
    hw_total = H_out * W_out
    n_idx = rows // hw_total
    hw_idx = rows % hw_total
    h_out_idx = hw_idx // W_out
    w_out_idx = hw_idx % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_range = tl.arange(0, BLOCK_K)
    rc = 0
    while rc < Cin:
        c_idx = rc + k_range  # [BLOCK_K]
        c_mask = c_idx < Cin

        # Iterate over spatial kernel (ConvTranspose2d stride=1,pad=0 equals Conv2d with rotated kernel and padding=K-1)
        for ky in tl.static_range(0, K):
            # input h coordinate for this ky
            h_in = h_out_idx + (ky - (K - 1))
            valid_y = (h_in >= 0) & (h_in < H)
            for kx in tl.static_range(0, K):
                w_in = w_out_idx + (kx - (K - 1))
                valid_x = (w_in >= 0) & (w_in < W)
                vmask = mask_m & valid_y & valid_x

                # Precompute base pointers to reduce integer ops in inner loop
                x_base = (
                    x_ptr
                    + n_idx[:, None] * stride_xn
                    + h_in[:, None] * stride_xh
                    + w_in[:, None] * stride_xw
                )
                x_ptrs = x_base + c_idx[None, :] * stride_xc
                x_mask = vmask[:, None] & c_mask[None, :]
                a = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

                # Load W tile: (BLOCK_K, BLOCK_N) from rotated weight layout [Cout, Cin, K, K]
                w_base = (
                    w_ptr
                    + cols[None, :] * stride_wo
                    + ky * stride_wkh
                    + kx * stride_wkw
                )
                w_ptrs = w_base + c_idx[:, None] * stride_wi
                w_mask = c_mask[:, None] & mask_n[None, :]
                b = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

                acc += tl.dot(a, b)
        rc += BLOCK_K

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + cols, mask=mask_n, other=0.0).to(tl.float32)
        acc = acc + bias_vals[None, :]

    # Store Y tile
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + cols[None, :] * stride_yc
        + h_out_idx[:, None] * stride_yh
        + w_out_idx[:, None] * stride_yw
    )
    y_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def _can_use_triton(self):
        ct = self.conv_transpose2d
        # Support only stride=1, padding=0, dilation=1, output_padding=0, groups=1
        k = ct.kernel_size
        if isinstance(k, tuple):
            if k[0] != k[1]:
                return False, 0
            k = k[0]
        s = ct.stride
        p = ct.padding
        op = ct.output_padding
        d = ct.dilation
        cond = (
            (k > 0)
            and (s == (1, 1) if isinstance(s, tuple) else s == 1)
            and (p == (0, 0) if isinstance(p, tuple) else p == 0)
            and (op == (0, 0) if isinstance(op, tuple) else op == 0)
            and (d == (1, 1) if isinstance(d, tuple) else d == 1)
            and (ct.groups == 1)
        )
        return cond, int(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        use_triton, K = self._can_use_triton()
        # Fallback for unsupported configs/devices/dtypes
        if not (use_triton and x.is_cuda and x.dtype == torch.float32 and self.conv_transpose2d.weight.dtype == torch.float32):
            return self.conv_transpose2d(x)

        # Prepare inputs
        x_c = x.contiguous()
        N, Cin, H, W = x_c.shape
        w = self.conv_transpose2d.weight  # [Cin, Cout, K, K]
        Cout = w.shape[1]

        # Output size for stride=1, padding=0, dilation=1, output_padding=0
        H_out = H + K - 1
        W_out = W + K - 1
        y = torch.empty((N, Cout, H_out, W_out), device=x.device, dtype=x.dtype)

        # Rotate weights for conv2d equivalence: (Cout, Cin, K, K)
        # flip spatial dims and swap in/out channels
        w_rot = w.flip(-1, -2).permute(1, 0, 2, 3).contiguous()

        # Bias (if present)
        bias = self.conv_transpose2d.bias
        has_bias = bias is not None
        bias_c = bias.contiguous() if has_bias else y  # dummy tensor if no bias

        # Strides (elements)
        sxn, sxc, sxh, sxw = x_c.stride()
        swo, swi, swkh, swkw = w_rot.stride()
        syn, syc, syh, syw = y.stride()

        # Launch configuration
        def grid(meta):
            return (
                triton.cdiv(N * H_out * W_out, meta['BLOCK_M']),
                triton.cdiv(Cout, meta['BLOCK_N']),
            )

        _convtransp2d_stride1_pad0_groups1_kernel[grid](
            x_c, w_rot, bias_c, y,
            N, Cin, H, W, Cout, K, H_out, W_out,
            sxn, sxc, sxh, sxw,
            swo, swi, swkh, swkw,
            syn, syc, syh, syw,
            HAS_BIAS=has_bias,
        )
        return y


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
height_in = 128
width_in = 256

def get_inputs():
    x = torch.randn(batch_size, in_channels, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization