import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _swish_scale_kernel(x_ptr, y_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Numerically stable sigmoid:
    # sigmoid(x) = 1 / (1 + exp(-x)) for x>=0; = exp(x) / (1 + exp(x)) for x<0
    z = tl.exp(-tl.abs(x))
    s = tl.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))

    out = (x * s) * scale
    tl.store(y_ptr + offs, out, mask=mask)


def swish_scale_triton(x: torch.Tensor, scale: float) -> torch.Tensor:
    # Fallback to PyTorch if input is not on CUDA or dtype unsupported
    if (not x.is_cuda) or (x.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        return x * torch.sigmoid(x) * scale

    # In-place to minimize memory traffic and allocation overhead
    x_c = x.contiguous()
    n_elements = x_c.numel()
    scale_scalar = float(scale)

    # Heuristic tuning for H200
    if n_elements <= 262_144:
        BLOCK_SIZE = 1024
        num_warps = 4
        num_stages = 2
    elif n_elements <= (1 << 20):
        BLOCK_SIZE = 4096
        num_warps = 8
        num_stages = 2
    else:
        BLOCK_SIZE = 8192
        num_warps = 8
        num_stages = 2

    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    _swish_scale_kernel[grid](
        x_c, x_c, n_elements, scale_scalar,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=num_stages
    )
    return x_c


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Swish activation, and scales the result.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.matmul(x)
        # Fused Swish + scaling via Triton (in-place)
        x = swish_scale_triton(x, self.scaling_factor)
        return x


batch_size = 128
in_features = 1024
out_features = 512
scaling_factor = 2.0


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, scaling_factor]