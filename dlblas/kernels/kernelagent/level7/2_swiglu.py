import torch
import torch.nn as nn

# Try to import Triton; if unavailable, we will fall back to PyTorch ops.
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def _swish_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

        # Compute in fp32 for numerical stability
        x_f32 = x.to(tl.float32)
        # Swish: x * sigmoid(x) == x / (1 + exp(-x))
        den = 1.0 + tl.exp(-x_f32)
        out = (x_f32 / den).to(x.dtype)

        tl.store(y_ptr + offsets, out, mask=mask)


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
        # Use Triton kernel on CUDA tensors with supported dtypes; otherwise fall back to PyTorch
        use_triton = (
            _TRITON_AVAILABLE
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        )
        if not use_triton:
            return x * torch.sigmoid(x)

        # Handle empty tensors safely
        if x.numel() == 0:
            return x.clone()

        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)
        n_elements = x_contig.numel()

        # Tune BLOCK_SIZE and launch config for better throughput
        BLOCK_SIZE = 4096
        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
        _swish_kernel[grid](
            x_contig, y, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=2
        )
        return y.view_as(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed