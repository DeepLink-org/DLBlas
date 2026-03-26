import torch
import torch.nn as nn

import triton
import triton.language as tl

@triton.jit
def _row_softmax_kernel(X_ptr, Y_ptr, stride_xm, stride_xn, stride_ym, stride_yn, N, BLOCK: tl.constexpr):
    # Each program handles one row (batch element)
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    x_row_ptr = X_ptr + row * stride_xm
    y_row_ptr = Y_ptr + row * stride_ym

    # Pass 1: compute the maximum value in the row for numerical stability
    n_tiles = tl.cdiv(N, BLOCK)
    max_val = tl.full([1], -float("inf"), dtype=tl.float32)
    i = 0
    while i < n_tiles:
        cols = i * BLOCK + offs
        mask = cols < N
        x = tl.load(x_row_ptr + cols * stride_xn, mask=mask, other=-float("inf"))
        x = x.to(tl.float32)
        tile_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, tile_max)
        i += 1

    # Pass 2: compute the denominator (sum of exp(x - max))
    denom = tl.zeros([1], dtype=tl.float32)
    i = 0
    while i < n_tiles:
        cols = i * BLOCK + offs
        mask = cols < N
        x = tl.load(x_row_ptr + cols * stride_xn, mask=mask, other=-float("inf"))
        x = x.to(tl.float32)
        e = tl.exp(x - max_val)
        denom += tl.sum(e, axis=0)
        i += 1

    inv_denom = 1.0 / denom

    # Pass 3: compute normalized softmax and store
    i = 0
    while i < n_tiles:
        cols = i * BLOCK + offs
        mask = cols < N
        x = tl.load(x_row_ptr + cols * stride_xn, mask=mask, other=-float("inf"))
        x = x.to(tl.float32)
        y = tl.exp(x - max_val) * inv_denom
        tl.store(y_row_ptr + cols * stride_yn, y, mask=mask)
        i += 1


class ModelNew(nn.Module):
    """
    Simple model that performs a Softmax activation using a Triton-optimized kernel on CUDA.
    Falls back to torch.softmax on CPU or unsupported devices.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor along dim=1.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        # Fallback to PyTorch implementation for non-CUDA tensors or non-2D inputs
        if (not x.is_cuda) or (x.dim() != 2):
            return torch.softmax(x, dim=1)

        B, N = x.shape
        y = torch.empty_like(x)

        # Choose a block size as a power of two up to 4096 for good performance
        BLOCK = min(4096, triton.next_power_of_2(N))

        # Launch one program per row
        grid = (B,)
        _row_softmax_kernel[grid](
            x, y,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            N,
            BLOCK=BLOCK,
            num_warps=8 if BLOCK > 1024 else 4,
            num_stages=2,
        )
        return y

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed