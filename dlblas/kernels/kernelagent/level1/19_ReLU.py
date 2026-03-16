import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=1),
    ],
    key=['n_elements'],
)
@triton.jit
def _relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr, IS_FP: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Vectorization / coalescing hints
    tl.multiple_of(offsets, 16)
    tl.max_contiguous(offsets, 16)

    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0)

    zero = tl.zeros([BLOCK_SIZE], dtype=x.dtype)
    y = tl.maximum(x, zero)

    # Explicitly propagate NaNs for floating types to match torch.relu semantics
    if IS_FP:
        nan_mask = x != x
        y = tl.where(nan_mask, x, y)

    tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        # Fallback to PyTorch if not CUDA or unsupported dtype/empty
        if (
            (not x.is_cuda)
            or x.numel() == 0
            or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            return torch.relu(x)

        n_elements = x.numel()
        # For small tensors, PyTorch's native kernel is typically faster
        if n_elements < (1 << 20):
            return torch.relu(x)

        # Ensure contiguous memory for efficient kernel execution
        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)

        is_fp = x_contig.dtype in (torch.float16, torch.bfloat16, torch.float32)

        grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
        _relu_kernel[grid](
            x_contig.view(-1),
            y.view(-1),
            n_elements=n_elements,
            IS_FP=is_fp,
        )
        return y.view_as(x_contig)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed