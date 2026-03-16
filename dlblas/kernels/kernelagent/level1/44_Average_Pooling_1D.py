import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def avgpool1d_forward_kernel(
    x_ptr,  # *[N_ROWS, L_IN]
    y_ptr,  # *[N_ROWS, L_OUT]
    N_ROWS: tl.constexpr,
    L_IN: tl.constexpr,
    L_OUT: tl.constexpr,
    stride_x_row: tl.constexpr,
    stride_y_row: tl.constexpr,
    STRIDE: tl.int32,
    PADDING: tl.int32,
    KERNEL_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(axis=0)
    col_block = tl.program_id(axis=1)

    # Offsets for the BLOCK of output elements this program handles
    offs = col_block * BLOCK + tl.arange(0, BLOCK)
    mask_o = offs < L_OUT

    # Row base pointers
    x_row_ptr = x_ptr + row_id * stride_x_row
    y_row_ptr = y_ptr + row_id * stride_y_row

    # Compute the start index in input for each output position
    j = offs * STRIDE - PADDING  # [BLOCK], int32

    # Accumulate in fp32 for numerical stability
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    # Unrolled reduction across kernel window
    # Keep clamped addresses for safety while using L2-friendly cache modifier
    for k in tl.static_range(0, KERNEL_SIZE):
        pos = j + k  # [BLOCK]
        in_bounds = (pos >= 0) & (pos < L_IN)
        mask_k = in_bounds & mask_o
        pos_safe = tl.maximum(tl.minimum(pos, L_IN - 1), 0)
        vals_k = tl.load(
            x_row_ptr + pos_safe,
            mask=mask_k,
            other=0.0,
            cache_modifier=".cg",
        )
        acc += vals_k.to(tl.float32)

    # Average with count_include_pad=True => divide by KERNEL_SIZE
    invK = 1.0 / float(KERNEL_SIZE)
    out = acc * invK

    # Store back to output with proper mask
    tl.store(y_row_ptr + offs, out, mask=mask_o)


class ModelNew(nn.Module):
    """
    Simple model that performs 1D Average Pooling using a Triton kernel (CUDA) with
    a CPU fallback to PyTorch's functional implementation. Semantics match nn.AvgPool1d
    with count_include_pad=True and ceil_mode=False.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch for non-CUDA tensors
        if x.device.type != "cuda":
            return F.avg_pool1d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                ceil_mode=False,
                count_include_pad=True,
            )

        # Ensure contiguous memory for predictable strides
        x = x.contiguous()
        B, C, L_in = x.shape

        # Output length following PyTorch formula (ceil_mode=False)
        L_out = (L_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        if L_out <= 0:
            return x.new_empty((B, C, 0))

        # Allocate output
        y = torch.empty((B, C, L_out), device=x.device, dtype=x.dtype)

        # Flatten batch and channels into rows for kernel
        x2 = x.view(B * C, L_in)
        y2 = y.view(B * C, L_out)

        N_ROWS = B * C
        # Choose a reasonable block size; favor fewer CTAs to reduce overhead while keeping occupancy
        if L_out >= 2048:
            BLOCK = 256
            num_warps = 8
        else:
            BLOCK = 128
            num_warps = 4

        grid = (N_ROWS, triton.cdiv(L_out, BLOCK))
        avgpool1d_forward_kernel[grid](
            x2,
            y2,
            N_ROWS,
            L_in,
            L_out,
            x2.stride(0),  # stride over rows in elements
            y2.stride(0),
            self.stride,
            self.padding,
            KERNEL_SIZE=self.kernel_size,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=4,
        )

        return y


batch_size = 16
in_channels = 32
input_length = 128
kernel_size = 4
stride = 2
padding = 1

def get_inputs():
    x = torch.randn(batch_size, in_channels, input_length)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]