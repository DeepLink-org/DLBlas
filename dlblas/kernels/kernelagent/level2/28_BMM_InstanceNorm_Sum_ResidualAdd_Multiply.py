import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=4),
    ],
    key=["F", "BLOCK"],
)
@triton.jit
def _rownorm_addmul_kernel(
    x_ptr,      # pointer to [B, F] input (after linear)
    y_ptr,      # pointer to [B, F] input y
    out_ptr,    # pointer to [B, F] output
    B,          # number of rows (batch size)
    F,          # number of features (out_features)
    stride_x,   # stride between consecutive rows of x in elements
    stride_y,   # stride between consecutive rows of y in elements
    stride_out, # stride between consecutive rows of out in elements
    eps,        # epsilon for numerical stability
    inv_F,      # 1.0 / F
    BLOCK: tl.constexpr,  # block size (next power of 2 >= F)
):
    pid = tl.program_id(0)  # row id
    offs = tl.arange(0, BLOCK)
    # Hints for better codegen on contiguous rows
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, BLOCK)

    # Masks to guard OOB
    row_mask = pid < B
    col_mask = offs < F
    mask = row_mask & col_mask

    # Base pointers for this row
    x_row_ptr = x_ptr + pid * stride_x + offs
    y_row_ptr = y_ptr + pid * stride_y + offs
    out_row_ptr = out_ptr + pid * stride_out + offs

    # Load row slices
    x_row = tl.load(x_row_ptr, mask=mask, other=0.0)
    y_row = tl.load(y_row_ptr, mask=mask, other=0.0)

    # Compute mean and variance across the row directly from x_row
    sum_x = tl.sum(x_row, axis=0)
    sum_x2 = tl.sum(x_row * x_row, axis=0)
    mean = sum_x * inv_F
    var = sum_x2 * inv_F - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = tl.rsqrt(var + eps)

    # Fuse normalize + add + mul: (x_hat + y) * y
    # out = y*y + y*(x - mean)*rstd
    y_sq = y_row * y_row
    out_row = y_sq + y_row * (x_row - mean) * rstd

    tl.store(out_row_ptr, out_row, mask=mask)


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


class ModelNew(nn.Module):
    """
    Model that performs a batch matrix multiplication (linear), instance normalization (row-wise over features),
    summation with y, and elementwise multiplication by y. The InstanceNorm2d in the reference normalizes over the
    last dimension (treated as spatial width with C=1), which is equivalent to per-row normalization here.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        # Keep for structural parity; not used in the optimized forward
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)
        self.eps = float(eps)

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Input tensor of shape (batch_size, out_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Linear transformation
        x = self.bmm(x)

        # Fast Triton path on CUDA: replicate InstanceNorm2d (C=1, H=1, W=F) semantics:
        # per-row normalization across feature dimension (mean/var over width).
        if x.is_cuda and y.is_cuda:
            B, F = x.shape
            x_c = x.contiguous()
            y_c = y.contiguous()
            out = torch.empty_like(y_c)

            # Choose BLOCK as next power of 2 >= F
            BLOCK = _next_power_of_2(F)

            grid = (B,)
            _rownorm_addmul_kernel[grid](
                x_c, y_c, out,
                B, F,
                x_c.stride(0), y_c.stride(0), out.stride(0),
                self.eps,
                1.0 / float(F),
                BLOCK=BLOCK,
            )
            return out
        else:
            # CPU or non-CUDA fallback: replicate semantics exactly.
            mean = x.mean(dim=1, keepdim=True)
            var = (x.pow(2).mean(dim=1, keepdim=True) - mean.pow(2)).clamp_min(0)
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            return (x_norm + y) * y


batch_size = 128
in_features = 64
out_features = 128

def get_inputs():
    return [torch.randn(batch_size, in_features), torch.randn(batch_size, out_features)]

def get_init_inputs():
    return [in_features, out_features]