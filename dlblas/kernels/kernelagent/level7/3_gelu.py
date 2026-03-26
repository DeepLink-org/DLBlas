import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def gelu_exact_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input and promote to fp32 for accurate math on fp16/bf16
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    # Keep constants in fp32 and use FMA for better ILP
    inv_sqrt2 = 0.70710678  # 1 / sqrt(2) in fp32
    t = x_f32 * inv_sqrt2
    erf_t = tl.math.erf(t)
    half_x = x_f32 * 0.5
    y_f32 = tl.fma(half_x, erf_t, half_x)

    # Store back to output (implicit cast to destination dtype)
    tl.store(y_ptr + offsets, y_f32, mask=mask, eviction_policy="evict_first")


class ModelNew(nn.Module):
    """
    Simple model that performs a GELU activation using a Triton kernel on CUDA tensors.
    Falls back to torch.nn.functional.gelu for non-CUDA or unsupported dtypes.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        # Fallback to PyTorch for CPU or unsupported dtypes
        if (not x.is_cuda) or (x.dtype not in (torch.float16, torch.float32, torch.bfloat16)):
            return torch.nn.functional.gelu(x)

        # Ensure contiguous for coalesced memory access
        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)

        n_elements = x_contig.numel()
        # Larger block size to reduce launch overhead and improve bandwidth utilization
        BLOCK_SIZE = 4096

        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        gelu_exact_kernel[grid](
            x_contig, y, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=2
        )
        return y

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed