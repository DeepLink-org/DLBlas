import torch
import torch.nn as nn

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _scale_hardtanh_gelu_kernel(
    x_ptr,  # [rows, cols]
    y_ptr,  # [rows, cols]
    rows: tl.constexpr,
    cols: tl.constexpr,
    stride_x,  # stride between rows in elements
    stride_y,  # stride between rows in elements
    scale,
    minv,
    maxv,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # row id
    pid_n = tl.program_id(1)  # block along N

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m = pid_m

    # Boundary mask (keep even if grid matches to satisfy safety constraint)
    mask = (m < rows) & (offs_n < cols)

    # Pointers
    x_ptrs = x_ptr + m * stride_x + offs_n
    y_ptrs = y_ptr + m * stride_y + offs_n

    # Hint for the compiler on contiguity/align to enable vectorization
    tl.max_contiguous(offs_n, BLOCK_N)
    tl.multiple_of(offs_n, 16)

    # Fast path for full tiles to avoid masked memory ops on interior blocks
    n_start = pid_n * BLOCK_N
    full_tile = (n_start + BLOCK_N) <= cols

    # Stream from global to avoid polluting L1 for this pure epilogue
    x = tl.load(x_ptrs, cache_modifier=".cg") if full_tile else tl.load(x_ptrs, mask=mask, other=0.0, cache_modifier=".cg")

    # Compute in fp32 for numerical stability/accuracy parity with PyTorch GELU
    xf = x.to(tl.float32)
    # scale
    xf = xf * scale
    # hardtanh clamp
    xf = tl.minimum(tl.maximum(xf, minv), maxv)
    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    y32 = 0.5 * xf * (1.0 + libdevice.erf(xf * inv_sqrt2))

    # Cast back to original dtype for storage
    y = y32.to(x.dtype)

    # Store
    if full_tile:
        tl.store(y_ptrs, y)
    else:
        tl.store(y_ptrs, y, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, scaling, hardtanh, and GELU activation.
    Fuses scale + hardtanh + GELU into a single Triton kernel for speed.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)
        self.gelu = nn.GELU()  # kept for parity with original interface

    def _post_ops_triton(self, y: torch.Tensor) -> torch.Tensor:
        rows, cols = y.shape
        # Ensure contiguous last-dim for coalesced access (Linear output is typically contiguous)
        if not y.is_contiguous():
            y = y.contiguous()

        # In-place is safe since computation is elementwise
        x_ptr = y
        y_ptr = y
        stride = y.stride(0)  # elements between consecutive rows

        # Choose tile size to minimize grid overhead and maximize throughput
        BLOCK_N = 256
        if cols >= 1024 and (cols % 1024 == 0):
            BLOCK_N = 1024
        elif cols >= 512 and (cols % 512 == 0):
            BLOCK_N = 512

        grid = (rows, triton.cdiv(cols, BLOCK_N))
        _scale_hardtanh_gelu_kernel[grid](
            x_ptr, y_ptr,
            rows, cols,
            stride, stride,
            self.scaling_factor,
            float(self.hardtanh.min_val),
            float(self.hardtanh.max_val),
            BLOCK_N=BLOCK_N,
            num_warps=8 if BLOCK_N >= 512 else 4,
            num_stages=1,
        )
        return y

    def forward(self, x):
        # GEMM (uses cuBLAS/cuDNN fastpath)
        y = self.gemm(x)
        # Fused post-ops on GPU via Triton; CPU falls back to PyTorch ops
        if y.is_cuda:
            return self._post_ops_triton(y)
        else:
            y = y * self.scaling_factor
            y = self.hardtanh(y)
            y = self.gelu(y)
            return y


batch_size = 128
in_features = 1024
out_features = 512
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]