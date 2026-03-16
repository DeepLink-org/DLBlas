import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _hardsigmoid_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Hints for better codegen/coalescing
    tl.max_contiguous(offsets, BLOCK_SIZE)
    tl.multiple_of(offsets, 8)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Piecewise-linear hardsigmoid with NaN propagation:
    # y = 0            if x <= -3
    # y = 1            if x >= 3
    # y = x/6 + 0.5    otherwise
    below = x <= -3.0
    above = x >= 3.0
    y_mid = x * (1.0 / 6.0) + 0.5
    y = tl.where(below, 0.0, y_mid)
    y = tl.where(above, 1.0, y)

    tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    """
    Simple model that performs a HardSigmoid activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardSigmoid activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardSigmoid applied, same shape as input.
        """
        # Fallback to PyTorch if not on CUDA or unsupported dtype
        if (not x.is_cuda) or (x.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
            return torch.nn.functional.hardsigmoid(x)

        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)

        n_elements = x_contig.numel()
        # Use a moderately large tile and more warps to improve throughput on Hopper.
        BLOCK_SIZE = 2048
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _hardsigmoid_kernel[grid](x_contig, y, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=2)

        return y

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed