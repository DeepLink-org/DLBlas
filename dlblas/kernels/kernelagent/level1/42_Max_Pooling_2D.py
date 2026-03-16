import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _maxpool2d_kernel(
    x_ptr,  # *fp*  [N, C, H, W] contiguous
    y_ptr,  # *fp*  [N, C, H_out, W_out] contiguous
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    BLOCK_HO: tl.constexpr,
    BLOCK_WO: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_ho = tl.program_id(1)
    pid_wo = tl.program_id(2)

    # Derive n and c from flattened nc
    n = pid_nc // C
    c = pid_nc % C

    # Tiled output indices
    ho_offsets = pid_ho * BLOCK_HO + tl.arange(0, BLOCK_HO)
    wo_offsets = pid_wo * BLOCK_WO + tl.arange(0, BLOCK_WO)

    ho_mask = ho_offsets < H_out
    wo_mask = wo_offsets < W_out

    HO = ho_offsets[:, None]  # [BH, 1]
    WO = wo_offsets[None, :]  # [1, BW]
    out_mask = ho_mask[:, None] & wo_mask[None, :]  # [BH, BW]

    # Strides for contiguous NCHW layout
    HW = H * W
    HWo = H_out * W_out
    base_x = (n * C + c) * HW
    base_y = (n * C + c) * HWo

    # Precompute output offsets
    y_offs = base_y + HO * W_out + WO

    # Input top-left start for each (ho, wo)
    h_start = HO * STRIDE_H - PAD_H  # [BH, BW]
    w_start = WO * STRIDE_W - PAD_W  # [BH, BW]

    # Detect "interior tiles" where all pooling windows are in-bounds and the tile is fully inside output
    # This allows us to skip masks entirely for the common case.
    tile_full_ho = ((pid_ho + 1) * BLOCK_HO) <= H_out
    tile_full_wo = ((pid_wo + 1) * BLOCK_WO) <= W_out
    hs0 = pid_ho * BLOCK_HO * STRIDE_H - PAD_H
    ws0 = pid_wo * BLOCK_WO * STRIDE_W - PAD_W
    hs_last = hs0 + (BLOCK_HO - 1) * STRIDE_H + (K_H - 1) * DIL_H
    ws_last = ws0 + (BLOCK_WO - 1) * STRIDE_W + (K_W - 1) * DIL_W
    tile_interior = tile_full_ho & tile_full_wo & (hs0 >= 0) & (hs_last < H) & (ws0 >= 0) & (ws_last < W)

    # Fastpath for very common 2x2 kernels
    if (K_H == 2) and (K_W == 2):
        ih0 = h_start
        ih1 = ih0 + DIL_H
        iw0 = w_start
        iw1 = iw0 + DIL_W

        step_h = DIL_H * W
        step_w = DIL_W

        if tile_interior:
            # Fully interior: no masks needed
            row0_base = base_x + ih0 * W
            offs00 = row0_base + iw0
            v00 = tl.load(x_ptr + offs00)
            v01 = tl.load(x_ptr + (offs00 + step_w))

            row1_base = row0_base + step_h
            offs10 = row1_base + iw0
            v10 = tl.load(x_ptr + offs10)
            v11 = tl.load(x_ptr + (offs10 + step_w))

            m0 = tl.maximum(v00, v01)
            m1 = tl.maximum(v10, v11)
            max_val = tl.maximum(m0, m1)
            tl.store(y_ptr + y_offs, max_val)
        else:
            ih0_in = (ih0 >= 0) & (ih0 < H)
            ih1_in = (ih1 >= 0) & (ih1 < H)
            iw0_in = (iw0 >= 0) & (iw0 < W)
            iw1_in = (iw1 >= 0) & (iw1 < W)

            # Row-wise masks to reduce logical ops
            mask_r0 = out_mask & ih0_in
            mask_r1 = out_mask & ih1_in

            row0_base = base_x + ih0 * W
            offs00 = row0_base + iw0

            v00 = tl.load(x_ptr + offs00, mask=(mask_r0 & iw0_in), other=-float("inf"))
            v01 = tl.load(x_ptr + (offs00 + step_w), mask=(mask_r0 & iw1_in), other=-float("inf"))

            row1_base = row0_base + step_h
            offs10 = row1_base + iw0
            v10 = tl.load(x_ptr + offs10, mask=(mask_r1 & iw0_in), other=-float("inf"))
            v11 = tl.load(x_ptr + (offs10 + step_w), mask=(mask_r1 & iw1_in), other=-float("inf"))

            m0 = tl.maximum(v00, v01)
            m1 = tl.maximum(v10, v11)
            max_val = tl.maximum(m0, m1)
            tl.store(y_ptr + y_offs, max_val, mask=out_mask)
    else:
        # General path: initialize accumulator with the first element to avoid dtype upcasts
        iw_list = []
        iw_in_list = []
        for kw in tl.static_range(0, K_W):
            iw_k = w_start + kw * DIL_W
            iw_list.append(iw_k)
            iw_in_list.append((iw_k >= 0) & (iw_k < W))

        # Initialize with (kh=0, kw=0)
        ih0 = h_start + 0 * DIL_H
        ih0_in = (ih0 >= 0) & (ih0 < H)
        offs_init = base_x + ih0 * W + iw_list[0]
        in0 = out_mask & ih0_in & iw_in_list[0]
        max_val = tl.load(x_ptr + offs_init, mask=in0, other=-float("inf"))

        # Iterate remaining kernel elements
        for kh in tl.static_range(0, K_H):
            ih = h_start + kh * DIL_H
            ih_in = (ih >= 0) & (ih < H)
            row_base = base_x + ih * W
            for kw in tl.static_range(0, K_W):
                if not (kh == 0 and kw == 0):
                    in_bounds = out_mask & ih_in & iw_in_list[kw]
                    offs = row_base + iw_list[kw]
                    val = tl.load(x_ptr + offs, mask=in_bounds, other=-float("inf"))
                    max_val = tl.maximum(max_val, val)

        tl.store(y_ptr + y_offs, max_val, mask=out_mask)


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 2D using a Triton kernel on CUDA tensors.
    Falls back to PyTorch implementation on CPU or when Triton is unavailable.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(ModelNew, self).__init__()
        # Store parameters for use in custom kernel
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)

    def _output_dim(self, L: int, k: int, s: int, p: int, d: int) -> int:
        # PyTorch formula with floor (ceil_mode=False)
        eff_k = (k - 1) * d + 1
        return max((L + 2 * p - eff_k) // s + 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 2D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D, shape (batch_size, channels, pooled_height, pooled_width).
        """
        # Fallback to PyTorch on CPU to preserve correctness
        if not x.is_cuda:
            return F.max_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                ceil_mode=False,
                return_indices=False,
            )

        assert x.dim() == 4, "Input must be 4D NCHW tensor"
        N, C, H, W = x.shape

        KH = KW = self.kernel_size
        SH = SW = self.stride
        PH = PW = self.padding
        DH = DW = self.dilation

        H_out = self._output_dim(H, KH, SH, PH, DH)
        W_out = self._output_dim(W, KW, SW, PW, DW)

        # Handle degenerate case
        if H_out == 0 or W_out == 0:
            return x.new_empty((N, C, H_out, W_out))

        # Ensure contiguous memory for simple address math
        x_in = x.contiguous()
        y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

        # Tile sizes tuned for better width coalescing and occupancy
        BLOCK_HO = 8
        BLOCK_WO = 64

        grid = (
            N * C,
            triton.cdiv(H_out, BLOCK_HO),
            triton.cdiv(W_out, BLOCK_WO),
        )

        # Launch kernel; compute in native dtype to reduce casts
        _maxpool2d_kernel[grid](
            x_in, y,
            N, C, H, W,
            H_out, W_out,
            SH, SW, PH, PW, DH, DW, KH, KW,
            BLOCK_HO=BLOCK_HO, BLOCK_WO=BLOCK_WO,
            num_warps=8,
            num_stages=4,
        )

        return y


batch_size = 16
channels = 32
height = 128
width = 128
kernel_size = 2
stride = 2
padding = 1
dilation = 3

def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]