import torch
import torch.nn as nn
import math
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def _gelu_tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Constants for GELU approximation
    c = 0.7978845608028654  # sqrt(2/pi)
    ca = 0.035677408136300125  # c * 0.044715

    # u = sqrt(2/pi) * (x + 0.044715 * x^3) = x * (c + ca * x^2)
    x2 = x_f32 * x_f32
    u = x_f32 * (c + ca * x2)

    # Use identity: 0.5*x*(1 + tanh(u)) == x * sigmoid(2u)
    s = 1.0 / (1.0 + tl.exp(-2.0 * u))
    y_f32 = x_f32 * s
    y = y_f32.to(x.dtype)

    tl.store(y_ptr + offs, y, mask=mask)


def _gelu_tanh_triton(x: torch.Tensor) -> torch.Tensor:
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)
    n_elements = x_contig.numel()
    if n_elements == 0:
        return y.view_as(x)
    BLOCK_SIZE = 4096
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    _gelu_tanh_kernel[grid](x_contig, y, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
    return y.view_as(x)


class ModelNew(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x):
        if x.is_cuda and x.is_contiguous():
            return _gelu_tanh_triton(x)
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

batch_size = 2000
dim = 2000

def get_inputs():
    return [torch.randn(batch_size, dim)]

def get_init_inputs():
    return []