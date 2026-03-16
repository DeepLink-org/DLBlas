import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_nchw_kernel(
    x_ptr, y_ptr,
    B, C, H, W,
    stride_b, stride_c, stride_h, stride_w,
    eps,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_w = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    w_start = pid_w * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W

    # Base offset for this (b, h) slice
    base = b * stride_b + h * stride_h

    # Fast path: if all channels fit in one BLOCK_C tile, load once and write once
    if C <= BLOCK_C:
        c_idx = tl.arange(0, BLOCK_C)
        c_mask = c_idx < C
        mask = c_mask[:, None] & mask_w[None, :]

        ptrs = x_ptr + base + c_idx[:, None] * stride_c + w_offsets[None, :] * stride_w
        x_tile = tl.load(ptrs, mask=mask, other=0.0)

        x_f32 = x_tile.to(tl.float32)
        sum_sq = tl.sum(x_f32 * x_f32, axis=0)
        # inv_rms = 1 / sqrt(mean + eps)
        inv_rms = tl.rsqrt(sum_sq / C + eps)
        inv_rms = tl.where(mask_w, inv_rms, 0.0)

        y_tile = x_tile * inv_rms[None, :].to(x_tile.dtype)
        out_ptrs = y_ptr + base + c_idx[:, None] * stride_c + w_offsets[None, :] * stride_w
        tl.store(out_ptrs, y_tile, mask=mask)
        return

    # General path: two-pass approach across channels
    # Pass 1: compute sum of squares across channels for each w in the tile
    sum_sq = tl.zeros((BLOCK_W,), dtype=tl.float32)
    c = 0
    while c < C:
        c_idx = c + tl.arange(0, BLOCK_C)
        c_mask = c_idx < C
        mask = c_mask[:, None] & mask_w[None, :]

        ptrs = x_ptr + base + c_idx[:, None] * stride_c + w_offsets[None, :] * stride_w
        x_tile = tl.load(ptrs, mask=mask, other=0.0)
        x_f32 = x_tile.to(tl.float32)
        sum_sq += tl.sum(x_f32 * x_f32, axis=0)
        c += BLOCK_C

    # Compute inv_rms = 1 / sqrt(mean + eps)
    inv_rms = tl.rsqrt(sum_sq / C + eps)
    inv_rms = tl.where(mask_w, inv_rms, 0.0)

    # Pass 2: normalize and store
    c = 0
    while c < C:
        c_idx = c + tl.arange(0, BLOCK_C)
        c_mask = c_idx < C
        mask = c_mask[:, None] & mask_w[None, :]

        in_ptrs = x_ptr + base + c_idx[:, None] * stride_c + w_offsets[None, :] * stride_w
        out_ptrs = y_ptr + base + c_idx[:, None] * stride_c + w_offsets[None, :] * stride_w

        x_tile = tl.load(in_ptrs, mask=mask, other=0.0)
        y_tile = x_tile * inv_rms[None, :].to(x_tile.dtype)
        tl.store(out_ptrs, y_tile, mask=mask)

        c += BLOCK_C


class ModelNew(nn.Module):
    """
    Simple model that performs RMS Normalization.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        # Use Triton kernel for CUDA 4D tensors in NCHW layout; otherwise fallback to PyTorch
        if (
            x.is_cuda
            and x.dim() == 4
            and x.size(1) == self.num_features
            and x.is_contiguous()
        ):
            B, C, H, W = x.shape
            y = torch.empty_like(x)

            # Extract element-wise strides
            stride_b, stride_c, stride_h, stride_w = x.stride()

            # Tuned tile sizes for H200
            BLOCK_W = 256
            BLOCK_C = 64

            grid = (B * H, triton.cdiv(W, BLOCK_W))

            # Cast eps to float32 for numerical stability; Triton will upcast as needed
            eps = float(self.eps)

            _rmsnorm_nchw_kernel[grid](
                x, y,
                B, C, H, W,
                stride_b, stride_c, stride_h, stride_w,
                eps,
                BLOCK_W=BLOCK_W,
                BLOCK_C=BLOCK_C,
                num_warps=8,
                num_stages=5,
            )
            return y
        else:
            # Reference PyTorch implementation
            rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
            return x / rms


batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]