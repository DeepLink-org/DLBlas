import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _upsample3d_scatter_kernel(
    in_ptr, out_ptr,
    N, C, Di, Hi, Wi,
    Do, Ho, Wo,
    SD, SH, SW,
    in_stride_n, in_stride_c, in_stride_d, in_stride_h, in_stride_w,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    BLOCK_W: tl.constexpr,
):
    # Simple compile-time sanity
    tl.static_assert(BLOCK_W > 0)
    # program ids
    pid_line = tl.program_id(axis=0)  # over lines (n,c,d,h)
    pid_wblk = tl.program_id(axis=1)  # block along width

    # Decompose pid_line into (n, c, d, h)
    CiHi = C * Di * Hi
    n = pid_line // CiHi
    rem = pid_line % CiHi
    c = rem // (Di * Hi)
    rem2 = rem % (Di * Hi)
    d = rem2 // Hi
    h = rem2 % Hi

    # Vector of width indices to process by this program instance
    w_off = pid_wblk * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_off < Wi

    # Base pointers for input/output lines
    in_base = in_ptr + n * in_stride_n + c * in_stride_c + d * in_stride_d + h * in_stride_h
    out_d_idx = d * SD
    out_h_idx = h * SH
    out_base = out_ptr + n * out_stride_n + c * out_stride_c + out_d_idx * out_stride_d + out_h_idx * out_stride_h

    # Load input values and scatter them into strided output positions
    x = tl.load(in_base + w_off * in_stride_w, mask=w_mask, other=0.0)
    out_w_pos = w_off * SW
    tl.store(out_base + out_w_pos * out_stride_w, x, mask=w_mask)


class ModelNew(nn.Module):
    """
    Performs a 3D transposed convolution operation with asymmetric input and kernel, and optional stride.

    Fast path on CUDA:
      - Insert zeros (upsample) via a Triton kernel.
      - Run torch.nn.functional.conv3d with flipped weights and adjusted padding.
    Falls back to nn.ConvTranspose3d for CPU or unsupported corner cases to ensure exact semantics.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    @staticmethod
    def _weight_to_conv3d(weight: torch.Tensor, groups: int) -> torch.Tensor:
        # Convert ConvTranspose3d weights [Cin, Cout/G, kD, kH, kW]
        # -> Conv3d weights [Cout, Cin/G, kD, kH, kW] with spatial flip.
        G = groups
        Cin, Co_g, kD, kH, kW = weight.shape
        Ci_g = Cin // G
        Co = Co_g * G
        w_flip = weight.flip(dims=(2, 3, 4))  # flip kd,kh,kw
        w_g = w_flip.view(G, Ci_g, Co_g, kD, kH, kW)  # [G, Ci_g, Co_g, kD, kH, kW]
        w_conv = w_g.permute(0, 2, 1, 3, 4, 5).contiguous().view(Co, Ci_g, kD, kH, kW)
        return w_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Fallback on CPU or when dilation != (1,1,1) to preserve exact PyTorch semantics
        if (not x.is_cuda) or any(d != 1 for d in self.conv_transpose3d.dilation):
            return self.conv_transpose3d(x)

        # Read parameters
        sd, sh, sw = self.conv_transpose3d.stride
        pd, ph, pw = self.conv_transpose3d.padding
        od, oh, ow = self.conv_transpose3d.output_padding
        groups = self.conv_transpose3d.groups
        weight = self.conv_transpose3d.weight
        bias = self.conv_transpose3d.bias
        kD, kH, kW = weight.shape[2], weight.shape[3], weight.shape[4]

        N, Cin, Di, Hi, Wi = x.shape

        # Compute upsampled spatial sizes: s*(L-1) + 1 + output_padding
        Du = (Di - 1) * sd + 1 + od
        Hu = (Hi - 1) * sh + 1 + oh
        Wu = (Wi - 1) * sw + 1 + ow

        # Convert weights and compute conv3d padding: pad' = k - 1 - p
        pad_d = kD - 1 - pd
        pad_h = kH - 1 - ph
        pad_w = kW - 1 - pw

        # If any pad' is negative, fallback to PyTorch op to preserve exact output size
        if (pad_d < 0) or (pad_h < 0) or (pad_w < 0):
            return self.conv_transpose3d(x)

        w_conv = self._weight_to_conv3d(weight, groups).contiguous()

        # Allocate zero-initialized upsampled tensor (prefer channels_last_3d for faster conv3d)
        x_up = torch.empty(
            (N, Cin, Du, Hu, Wu),
            dtype=x.dtype,
            device=x.device,
            memory_format=torch.channels_last_3d,
        ).zero_()

        # Launch Triton kernel to scatter x into x_up at strided positions
        in_strides = x.stride()
        out_strides = x_up.stride()

        BLOCK_W = 128
        grid = (N * Cin * Di * Hi, triton.cdiv(Wi, BLOCK_W))
        _upsample3d_scatter_kernel[grid](
            x, x_up,
            N, Cin, Di, Hi, Wi,
            Du, Hu, Wu,
            sd, sh, sw,
            in_strides[0], in_strides[1], in_strides[2], in_strides[3], in_strides[4],
            out_strides[0], out_strides[1], out_strides[2], out_strides[3], out_strides[4],
            BLOCK_W=BLOCK_W,
            num_warps=4,
            num_stages=2,
        )

        # Convolution with converted weights; stride=1, dilation=1, groups preserved
        y = F.conv3d(
            x_up, w_conv, bias=bias, stride=1,
            padding=(pad_d, pad_h, pad_w), dilation=1, groups=groups
        )
        return y


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5, 7)
depth = 16
height = 32
width = 64
stride = (2, 2, 2)
padding = (1, 2, 3)
output_padding = (1, 1, 1)
groups = 4

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, groups]