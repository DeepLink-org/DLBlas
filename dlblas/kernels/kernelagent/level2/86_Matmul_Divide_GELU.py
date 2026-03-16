import torch
import torch.nn as nn

# Use Triton to fuse division and GELU for post-linear activation on CUDA
try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=8, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def fused_div_gelu_kernel(x_ptr, out_ptr, n_elements, inv_divisor, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Replace division by multiplication for speed and precision stability
    y = x * inv_divisor
    # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.70710678118654752440084436210485  # 1 / sqrt(2)
    t = y * inv_sqrt2
    erf_t = libdevice.erf(t)
    out = 0.5 * y * (1.0 + erf_t)
    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, divides by a scalar, and applies GELU activation.
    Uses a fused Triton kernel to combine division and GELU for improved performance on large tensors,
    and falls back to PyTorch ops for smaller tensors to minimize launch overhead.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = float(divisor)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # cuBLAS-backed linear for peak performance
        x = self.linear(x)

        # If not on CUDA or dtype unsupported, use PyTorch path
        if (not TRITON_AVAILABLE) or (not x.is_cuda) or (x.dtype != torch.float32):
            x = x / self.divisor
            x = torch.nn.functional.gelu(x)
            return x

        # Heuristic: for small tensors, PyTorch's highly optimized kernels are faster due to lower launch overhead.
        n_elements = x.numel()
        if n_elements < (1 << 20):  # 1,048,576 elements threshold; tuneable
            x = x / self.divisor
            x = torch.nn.functional.gelu(x)
            return x

        # Ensure contiguous for coalesced memory access
        x = x.contiguous()
        out = torch.empty_like(x)

        # Use multiplication by reciprocal to avoid per-element division
        inv_divisor = 1.0 / self.divisor

        # Autotuned kernel launch; meta['BLOCK_SIZE'] chosen by Triton
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        fused_div_gelu_kernel[grid](x, out, n_elements, inv_divisor)
        return out


batch_size = 128
input_size = 512
output_size = 1024
divisor = 10.0

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, output_size, divisor]