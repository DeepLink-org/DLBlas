import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _softplus_kernel(
    x_ptr,  # *const input
    y_ptr,  # *mut output
    n_elements,
    THRESHOLD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # PyTorch F.softplus default behavior (beta=1, threshold=20):
    # if x > threshold: y = x
    # else: y = log1p(exp(x))
    # Use a numerically-stable equivalent for the "else" branch:
    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    cond = x32 > THRESHOLD
    abs_x = tl.abs(x32)
    stable_term = tl.maximum(x32, 0.0) + tl.log(1.0 + tl.exp(-abs_x))
    y32 = tl.where(cond, x32, stable_term)

    y = y32.to(x.dtype)
    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    """
    Simple model that performs a Softplus activation using a Triton kernel on CUDA tensors.
    Falls back to torch.nn.functional.softplus for unsupported dtypes/devices.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback for non-CUDA tensors or unsupported dtypes
        if (not x.is_cuda) or (x.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
            return torch.nn.functional.softplus(x)

        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)
        n_elements = x_contig.numel()
        if n_elements == 0:
            return y

        BLOCK_SIZE = 4096

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        _softplus_kernel[grid](
            x_contig, y, n_elements, THRESHOLD=20.0, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=2
        )
        return y

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed