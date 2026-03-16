import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _spatial_mean_subtract_kernel(
    x_ptr,       # *: [N, C, D, H, W] contiguous in spatial dims
    y_ptr,       # *: [N, C, D, H, W] output
    stride_n,    # stride along N dimension (elements)
    stride_c,    # stride along C dimension (elements)
    S,           # total spatial elements per (n, c) = D*H*W
    N,           # batch size
    C,           # channels
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # Map program id to (n, c)
    n = pid // C
    c = pid % C

    # Base pointers for this (n, c) plane
    base_x = x_ptr + n * stride_n + c * stride_c
    base_y = y_ptr + n * stride_n + c * stride_c

    offsets = tl.arange(0, BLOCK_SIZE)
    tl.static_assert(BLOCK_SIZE % 128 == 0)

    # First pass: compute spatial mean using scalar accumulator (low register pressure)
    sum_acc = tl.full((), 0.0, dtype=tl.float32)
    i = 0
    while i < S:
        idx = i + offsets
        mask = idx < S
        vals = tl.load(base_x + idx, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
        sum_acc += tl.sum(vals, axis=0)
        i += BLOCK_SIZE

    denom = tl.full((), S, dtype=tl.float32)
    mean = sum_acc / denom  # scalar in fp32

    # Second pass: subtract mean and write out
    i = 0
    while i < S:
        idx = i + offsets
        mask = idx < S
        vals = tl.load(base_x + idx, mask=mask, other=0.0, eviction_policy="evict_last")
        out = vals.to(tl.float32) - mean
        tl.store(base_y + idx, out.to(vals.dtype), mask=mask)
        i += BLOCK_SIZE


class ModelNew(nn.Module):
    """
    A 3D convolutional transpose layer followed by Batch Normalization and subtraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)

        # Use Triton kernel on CUDA for fast spatial mean subtraction; fallback to PyTorch otherwise
        if x.is_cuda and not x.requires_grad:
            x_contig = x.contiguous()  # ensure spatial dims are contiguous
            N, C, D, H, W = x_contig.shape
            S = D * H * W
            y = torch.empty_like(x_contig)
            stride_n, stride_c = x_contig.stride(0), x_contig.stride(1)

            # One program per (n, c) plane over spatially-contiguous memory
            grid = lambda meta: (N * C,)
            _spatial_mean_subtract_kernel[grid](
                x_contig, y,
                stride_n, stride_c,
                S, N, C,
            )
            return y
        else:
            return x - torch.mean(x, dim=(2, 3, 4), keepdim=True)  # Subtract mean along spatial dimensions


batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]