import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_epilogue_swish_div_clamp_tanh(
    x_ptr,  # input pointer
    y_ptr,  # output pointer
    n_elements,  # total number of elements
    BLOCK: tl.constexpr,  # block size
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tl.max_contiguous(offsets, BLOCK)
    mask = offsets < n_elements

    # Load and upcast to fp32 for stable math
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # Swish: x * sigmoid(x) == x / (1 + exp(-x))
    y = xf / (1.0 + tl.exp(-xf))
    # Divide by 2
    y = y * 0.5

    # Clamp to [-1, 1]
    y = tl.maximum(y, -1.0)
    y = tl.minimum(y, 1.0)

    # tanh(y) using exp formulation: tanh(y) = (e^(2y) - 1) / (e^(2y) + 1)
    e2y = tl.exp(2.0 * y)
    y = (e2y - 1.0) / (e2y + 1.0)

    # Final clamp to [-1, 1]
    y = tl.maximum(y, -1.0)
    y = tl.minimum(y, 1.0)

    # Cast back and store
    y = y.to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    """
    Simple model that performs a gemm, swish, divide, clamp, tanh, and clamp operations.
    Uses a Triton kernel to fuse the elementwise epilogue for improved performance.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # GEMM using highly-optimized cuBLAS via PyTorch
        x = self.gemm(x)

        # Fused epilogue on CUDA using Triton; CPU falls back to PyTorch ops
        if x.is_cuda and x.is_contiguous():
            out = torch.empty_like(x)
            n_elements = x.numel()
            if n_elements == 0:
                return out
            BLOCK = 4096
            grid = lambda META: (triton.cdiv(n_elements, META["BLOCK"]),)
            _fused_epilogue_swish_div_clamp_tanh[grid](
                x, out, n_elements, BLOCK=BLOCK, num_warps=4, num_stages=2
            )
            return out
        else:
            # Fallback path: preserve exact semantics of the original implementation
            x = x * torch.sigmoid(x)  # Swish activation
            x = x / 2.0
            x = torch.clamp(x, min=-1.0, max=1.0)  # Clamp between -1 and 1
            x = torch.tanh(x)  # Tanh activation
            x = torch.clamp(x, min=-1.0, max=1.0)  # Clamp between -1 and 1
            return x


batch_size = 128
in_features = 1024
out_features = 512

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]