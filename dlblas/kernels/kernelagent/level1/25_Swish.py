import torch
import torch.nn as nn

# Try importing Triton; fall back gracefully if unavailable
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def _swish_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        # Hints for better codegen and memory coalescing
        tl.multiple_of(offs, 16)
        tl.max_contiguous(offs, BLOCK_SIZE)

        # Streaming loads/stores to reduce cache pollution on H200
        x = tl.load(x_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg")

        # Compute Swish in fp32 for numerical robustness
        xf = x.to(tl.float32)
        LOG2E = 1.4426950408889634  # 1 / ln(2)
        e = tl.exp2(-xf * LOG2E)
        s = 1.0 / (1.0 + e)
        y = (xf * s).to(x.dtype)

        tl.store(y_ptr + offs, y, mask=mask, cache_modifier=".cg")


class ModelNew(nn.Module):
    """
    Simple model that performs a Swish activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Swish activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Swish applied, same shape as input.
        """
        # CPU or Triton-unavailable fallback, or when gradients are required
        if (not x.is_cuda) or (not _TRITON_AVAILABLE) or x.requires_grad:
            return x * torch.sigmoid(x)

        # Ensure contiguous for coalesced memory access
        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)
        n_elements = x_contig.numel()
        if n_elements == 0:
            return y

        # Heuristic block size/warps for this problem size
        if n_elements >= (1 << 18):      # >= 262,144
            block_size = 32768
            num_warps = 8
        elif n_elements >= (1 << 16):    # >= 65,536
            block_size = 8192
            num_warps = 4
        else:
            block_size = 2048
            num_warps = 2

        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _swish_kernel[grid](x_contig, y, n_elements, BLOCK_SIZE=block_size, num_warps=num_warps, num_stages=2)

        return y

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed