import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def _permute_flip_weight_kernel(
    src_ptr,  # (in_c, out_c_per_group, kH, kW)
    dst_ptr,  # (out_c, in_c_per_group, kH, kW)
    s_wi, s_wo, s_wh, s_ww,
    s_do, s_di, s_dh, s_dw,
    in_c, out_c, kH, kW, groups, num_kw_tiles,
    BLOCK_KW: tl.constexpr,
):
    # Program IDs
    pid_o = tl.program_id(0)  # out channel (total)
    pid_i = tl.program_id(1)  # in channel within group
    pid_t = tl.program_id(2)  # fused (kh, kw-tile)

    # Decompose fused pid into kh and kw-tile
    tile_idx = pid_t % num_kw_tiles
    kh_idx = pid_t // num_kw_tiles

    out_per_group = out_c // groups
    in_per_group = in_c // groups

    # Compute group for this output channel
    g = pid_o // out_per_group
    o_within = pid_o % out_per_group
    i_total = g * in_per_group + pid_i

    # Offsets along kW
    kw_offsets = tile_idx * BLOCK_KW + tl.arange(0, BLOCK_KW)
    tl.max_contiguous(kw_offsets, BLOCK_KW)
    tl.multiple_of(kw_offsets, 8)
    mask_kw = kw_offsets < kW
    kw_flipped = (kW - 1) - kw_offsets

    # Source pointer (flip along h and w)
    src_ptrs = (
        src_ptr
        + i_total * s_wi
        + o_within * s_wo
        + (kH - 1 - kh_idx) * s_wh
        + kw_flipped * s_ww
    )
    vals = tl.load(src_ptrs, mask=mask_kw, other=0, cache_modifier=".cg")

    # Destination pointer (permute to (out_c, in_c_per_group, kH, kW))
    dst_ptrs = (
        dst_ptr
        + pid_o * s_do
        + pid_i * s_di
        + kh_idx * s_dh
        + kw_offsets * s_dw
    )
    tl.store(dst_ptrs, vals, mask=mask_kw)


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
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
        # Cache for transformed weights for fast path
        self._cached_w2 = None
        self._cached_meta = None  # dict with keys: ptr, version, device, dtype, shape
        # Precompute the conv2d-equivalent weight on CPU; registered as buffer so it moves with .to(device/dtype)
        with torch.no_grad():
            w = self.conv_transpose2d.weight.detach()  # likely on CPU at init
            g = self.conv_transpose2d.groups
            in_c, out_pg, kH, kW = w.shape
            assert in_c % g == 0
            in_pg = in_c // g
            # (G, in_pg, out_pg, kH, kW) -> flip hw -> (G, out_pg, in_pg, kH, kW) -> (out_c, in_pg, kH, kW)
            w2_cpu = (
                w.view(g, in_pg, out_pg, kH, kW)
                 .flip(dims=[3, 4])
                 .permute(0, 2, 1, 3, 4)
                 .contiguous()
                 .view(self.conv_transpose2d.out_channels, in_pg, kH, kW)
                 .contiguous()
            )
        self.register_buffer("_pre_w2", w2_cpu)
        self._pre_w2_version = int(self.conv_transpose2d.weight._version)

    def _fast_path_available(self):
        # Fast path when stride=1, dilation=1, output_padding=0
        s = self.conv_transpose2d.stride
        d = self.conv_transpose2d.dilation
        op = self.conv_transpose2d.output_padding
        if isinstance(s, int):
            s = (s, s)
        if isinstance(d, int):
            d = (d, d)
        if isinstance(op, int):
            op = (op, op)
        return s == (1, 1) and d == (1, 1) and op == (0, 0)

    @staticmethod
    def _select_block_kw(kW: int) -> int:
        # Pick a small power-of-two tile for tiny kernels to avoid masked lanes
        if kW <= 8:
            return 8
        if kW <= 16:
            return 16
        if kW <= 32:
            return 32
        if kW <= 64:
            return 64
        return 128

    def _weight_to_conv2d_cached(self, weight: torch.Tensor) -> torch.Tensor:
        # Convert ConvTranspose2d weight (in_c, out_c_per_group, kH, kW)
        # to Conv2d weight (out_c, in_c_per_group, kH, kW) with 180-degree flip in spatial dims.
        # Cache across forwards to avoid repeated transforms.
        assert weight.is_cuda, "Fast Triton path requires CUDA tensors"

        meta = dict(
            ptr=weight.data_ptr(),
            version=int(weight._version),
            device=weight.device,
            dtype=weight.dtype,
            shape=tuple(weight.shape),
            groups=self.conv_transpose2d.groups,
            out_channels=self.conv_transpose2d.out_channels,
        )
        # If we already have a valid GPU-cached transform, use it
        if (
            self._cached_w2 is not None
            and self._cached_meta is not None
            and all(meta[k] == self._cached_meta.get(k) for k in meta.keys())
        ):
            return self._cached_w2

        # If a precomputed buffer exists and is up-to-date and on the right device/dtype, use it directly
        if (
            hasattr(self, "_pre_w2")
            and self._pre_w2 is not None
            and self._pre_w2_version == int(weight._version)
            and self._pre_w2.dtype == weight.dtype
            and self._pre_w2.device == weight.device
        ):
            self._cached_w2 = self._pre_w2
            self._cached_meta = meta
            return self._cached_w2

        in_c, out_c_per_group, kH, kW = weight.shape
        groups = self.conv_transpose2d.groups
        out_c = self.conv_transpose2d.out_channels
        in_per_group = in_c // groups

        # Heuristic: for small kernels/tensors, let PyTorch handle the transform (often faster than a Triton kernel launch).
        total_elems = in_c * out_c_per_group * kH * kW
        SMALL_THRESH = 64 * 1024  # elements
        if total_elems <= SMALL_THRESH:
            # Reshape by groups -> flip -> permute to (groups, ocpg, icpg, kH, kW) -> merge groups
            wg = weight.view(groups, in_per_group, out_c_per_group, kH, kW)
            wg = wg.flip(dims=[3, 4]).permute(0, 2, 1, 3, 4).contiguous()
            w2 = wg.reshape(out_c, in_per_group, kH, kW).contiguous()
            self._cached_w2 = w2
            self._cached_meta = meta
            return w2

        # Allocate destination tensor for Triton path
        w2 = torch.empty((out_c, in_per_group, kH, kW), dtype=weight.dtype, device=weight.device)

        # Triton kernel launch config
        BLOCK_KW = self._select_block_kw(kW)
        num_kw_tiles = (kW + BLOCK_KW - 1) // BLOCK_KW
        grid = (out_c, in_per_group, kH * max(num_kw_tiles, 1))

        s_wi, s_wo, s_wh, s_ww = weight.stride()
        s_do, s_di, s_dh, s_dw = w2.stride()

        # Choose warps based on tile size to reduce overhead on small kernels
        if BLOCK_KW <= 16:
            num_warps = 1
            num_stages = 1
        elif BLOCK_KW <= 64:
            num_warps = 2
            num_stages = 2
        else:
            num_warps = 4
            num_stages = 2

        # Run kernel to permute + flip spatially
        _permute_flip_weight_kernel[grid](
            weight, w2,
            s_wi, s_wo, s_wh, s_ww,
            s_do, s_di, s_dh, s_dw,
            in_c, out_c, kH, kW, groups, num_kw_tiles,
            BLOCK_KW=BLOCK_KW,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        # Cache
        self._cached_w2 = w2
        self._cached_meta = meta
        return w2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Only enable fast path on CUDA, inference graph (no autograd), and simple stride/dilation/output_padding
        if (x.is_cuda and self._fast_path_available() and not torch.is_grad_enabled()):
            w = self.conv_transpose2d.weight  # (in_c, out_c_per_group, kH, kW)
            bias = self.conv_transpose2d.bias
            p = self.conv_transpose2d.padding
            if isinstance(p, int):
                p = (p, p)
            kH, kW = w.shape[2], w.shape[3]

            # Compute equivalent conv2d padding
            pad_h = kH - 1 - p[0]
            pad_w = kW - 1 - p[1]
            if pad_h < 0 or pad_w < 0:
                return self.conv_transpose2d(x)

            # Prefer using the precomputed buffer (moved to the right device with .to(cuda))
            with torch.no_grad():
                if (
                    hasattr(self, "_pre_w2")
                    and self._pre_w2 is not None
                    and self._pre_w2_version == int(w._version)
                    and self._pre_w2.device == w.device
                    and self._pre_w2.dtype == w.dtype
                ):
                    w2 = self._pre_w2
                else:
                    w2 = self._weight_to_conv2d_cached(w)

            # Use NHWC input to favor fast cudnn kernels on Hopper
            x_nhwc = x.contiguous(memory_format=torch.channels_last)

            # Execute as conv2d (equivalent to transposed conv when stride=1, dilation=1, output_padding=0)
            return F.conv2d(
                x_nhwc, w2, bias=bias, stride=1, padding=(pad_h, pad_w), dilation=1, groups=self.conv_transpose2d.groups
            )

        # Fallback to the standard implementation (ensures gradients and all corner cases)
        # Try to also benefit from channels_last memory format for cudnn speedups
        if x.is_cuda:
            x_opt = x.contiguous(memory_format=torch.channels_last)
            return self.conv_transpose2d(x_opt)
        return self.conv_transpose2d(x)


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)  # Asymmetric kernel
width = 128
height = 128

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization