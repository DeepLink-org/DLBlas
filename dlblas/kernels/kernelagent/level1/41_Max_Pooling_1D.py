import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _maxpool1d_forward_kernel(
    x_ptr,                # *T,  input [NC][L_in]
    y_ptr,                # *T,  output [NC][L_out]
    idx_ptr,              # *int64, indices [NC][L_out] (optional)
    L_in,                 # int32
    L_out,                # int32
    STRIDE,               # int32
    PADDING,              # int32
    DILATION,             # int32
    line_stride_x,        # int32 = L_in
    line_stride_y,        # int32 = L_out
    HAS_INDEX: tl.constexpr,  # bool, whether to write indices
    K: tl.constexpr,          # kernel size (compile-time)
    BLOCK: tl.constexpr,      # tile size along output length
):
    pid_nc = tl.program_id(axis=0)          # which (N,C) line
    pid_o_blk = tl.program_id(axis=1)       # which output tile

    o_offsets = pid_o_blk * BLOCK + tl.arange(0, BLOCK)
    mask_o = o_offsets < L_out

    # compute window start per output position
    starts = o_offsets * STRIDE - PADDING  # [BLOCK], int32

    base_x = x_ptr + pid_nc * line_stride_x
    base_y = y_ptr + pid_nc * line_stride_y

    # Vectorized, mostly-unmasked loads with clamped addresses to improve coalescing.
    pos = starts
    valid0 = (pos >= 0) & (pos < L_in) & mask_o
    # clamp to valid range to allow unmasked loads for valid output lanes
    addr0 = tl.minimum(tl.maximum(pos, 0), L_in - 1)
    x0 = tl.load(base_x + addr0, mask=mask_o, other=0)
    y_max = tl.where(valid0, x0, -float("inf"))
    if HAS_INDEX:
        chosen_pos = pos

    # k = 1..K-1
    for _ in tl.static_range(1, K):
        pos = pos + DILATION
        validk = (pos >= 0) & (pos < L_in) & mask_o
        addrk = tl.minimum(tl.maximum(pos, 0), L_in - 1)
        xk = tl.load(base_x + addrk, mask=mask_o, other=0)
        vk = tl.where(validk, xk, -float("inf"))
        better = vk > y_max
        y_max = tl.where(better, vk, y_max)
        if HAS_INDEX:
            chosen_pos = tl.where(better, pos, chosen_pos)

    # Store results
    tl.store(base_y + o_offsets, y_max, mask=mask_o)

    if HAS_INDEX:
        # Clamp to valid input range; out-of-bounds lanes end up clamped to 0.
        chosen_pos = tl.maximum(0, tl.minimum(chosen_pos, L_in - 1))
        base_i = idx_ptr + pid_nc * line_stride_y
        tl.store(base_i + o_offsets, chosen_pos.to(tl.int64), mask=mask_o)


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 1D.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        """
        Initializes the Max Pooling 1D layer.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
            padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(kernel_size if stride is None else stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.return_indices = bool(return_indices)

        # Fallback reference module for non-CUDA or unsupported dtypes
        self._ref = nn.MaxPool1d(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            return_indices=self.return_indices,
        )

    def _out_length(self, L_in: int) -> int:
        # PyTorch formula with ceil_mode=False
        return (L_in + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 1D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 1D applied, shape (batch_size, num_features, output_sequence_length).
        """
        # Fallback if not CUDA or not floating type
        if (not x.is_cuda) or (x.dtype not in (torch.float16, torch.float32, torch.bfloat16)):
            return self._ref(x)

        x = x.contiguous()
        N, C, L_in = x.shape
        L_out = self._out_length(L_in)

        # Guard unusual edge-cases by delegating to reference
        if L_out <= 0:
            return self._ref(x)

        y = torch.empty((N, C, L_out), device=x.device, dtype=x.dtype)
        indices = None
        if self.return_indices:
            indices = torch.empty((N, C, L_out), device=x.device, dtype=torch.int64)

        # Launch kernel: flatten (N, C) into NC lines
        NC = N * C
        BLOCK = 128
        grid = (NC, triton.cdiv(L_out, BLOCK))

        _maxpool1d_forward_kernel[grid](
            x, y, indices if self.return_indices else torch.empty(0, device=x.device, dtype=torch.int64),
            L_in, L_out, self.stride, self.padding, self.dilation,
            L_in, L_out,
            HAS_INDEX=self.return_indices,
            K=self.kernel_size,
            BLOCK=BLOCK,
            num_warps=4,
            num_stages=4,
        )

        if self.return_indices:
            return y, indices
        return y


batch_size = 16
features = 64
sequence_length = 128
kernel_size = 4
stride = 2
padding = 2
dilation = 3
return_indices = False

def get_inputs():
    x = torch.randn(batch_size, features, sequence_length)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]