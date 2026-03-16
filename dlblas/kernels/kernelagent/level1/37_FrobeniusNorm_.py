import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _sumsq_kernel(x_ptr, n_elements, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    tl.max_contiguous(offsets, BLOCK)
    mask = offsets < n_elements
    # Load as source dtype then accumulate in fp32
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(x * x, axis=0)
    # Accumulate per-CTA partial sum into a single global scalar
    tl.atomic_add(out_ptr, s)


@triton.jit
def _reduce_partials_kernel(partials_ptr, n_partials, out_ptr, BLOCK: tl.constexpr):
    # Kept for compatibility (unused in this optimized path)
    offs = tl.arange(0, BLOCK)
    acc = 0.0
    idx = 0
    while idx < n_partials:
        offsets = idx + offs
        mask = offsets < n_partials
        vals = tl.load(partials_ptr + offsets, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
        idx += BLOCK
    tl.store(out_ptr, acc)


@triton.jit
def _scale_kernel(x_ptr, y_ptr, n_elements, sumsq_ptr, BLOCK: tl.constexpr):
    # Load global sum of squares and compute inverse Frobenius norm
    ss = tl.load(sumsq_ptr)  # float32
    inv_norm = tl.rsqrt(ss)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    tl.max_contiguous(offsets, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x.to(tl.float32) * inv_norm
    tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    """
    Simple model that performs Frobenius norm normalization.
    """
    def __init__(self):
        """
        Initializes the Frobenius norm normalization layer.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Frobenius norm normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        # Fallback for unsupported dtypes/devices or degenerate cases
        if (not x.is_cuda) or (x.dtype == torch.float64) or (x.numel() == 0):
            norm = torch.norm(x, p='fro')
            return x / norm

        # Contiguous memory for efficient Triton access
        x_contig = x if x.is_contiguous() else x.contiguous()
        n_elements = x_contig.numel()

        # Larger block to reduce grid size and atomics; tuned for Hopper (H200)
        BLOCK = 16384
        grid = (triton.cdiv(n_elements, BLOCK),)

        # Accumulate sum of squares in fp32 using atomics (no partials buffer, fewer kernel launches)
        sumsq = torch.zeros(1, device=x_contig.device, dtype=torch.float32)
        _sumsq_kernel[grid](x_contig, n_elements, sumsq, BLOCK=BLOCK, num_warps=8, num_stages=4)

        # Result dtype follows PyTorch's promotion for division by float32 scalar
        out_dtype = torch.promote_types(x_contig.dtype, torch.float32)
        y = torch.empty_like(x_contig, dtype=out_dtype)

        # Pass 2: scale by inverse Frobenius norm
        _scale_kernel[grid](x_contig, y, n_elements, sumsq, BLOCK=BLOCK, num_warps=8, num_stages=4)

        return y

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return []