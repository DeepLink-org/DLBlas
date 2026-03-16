import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _log_softmax_row_fused_kernel(x_ptr, y_ptr, D, BLOCK_SIZE: tl.constexpr):
    """
    One program processes one row (length D) and computes log_softmax in a single pass
    using registers:
      y = x - (m + log(sum(exp(x - m))))
    """
    pid = tl.program_id(axis=0)
    row_start = pid * D
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    # Load row and upcast to fp32 for stable reductions
    x = tl.load(x_ptr + row_start + offs, mask=mask, other=-float("inf"))
    x32 = x.to(tl.float32)

    # Stable log-softmax: subtract row max, sum exp, subtract log-sum
    m = tl.max(x32, axis=0)
    x_shift = x32 - m
    exp_x = tl.exp(x_shift)
    denom = tl.sum(exp_x, axis=0)
    log_denom = tl.log(denom)
    y = x_shift - log_denom

    # Store back (implicit cast to output dtype)
    tl.store(y_ptr + row_start + offs, y, mask=mask)


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _log_softmax_triton(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Use Triton path for common 2D case along the last dimension (dim==1 or -1) on CUDA
    if (
        x.is_cuda
        and x.ndim == 2
        and (dim in (1, -1))
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        B, D = x.shape
        if B == 0 or D == 0:
            return torch.log_softmax(x, dim=dim)

        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)

        BLOCK_SIZE = _next_power_of_2(D)

        # Heuristic tuning for H200
        if D >= 16384:
            num_warps = 16
            num_stages = 2
        elif D >= 8192:
            num_warps = 8
            num_stages = 2
        elif D >= 4096:
            num_warps = 8
            num_stages = 2
        else:
            num_warps = 4
            num_stages = 2

        grid = (B,)
        _log_softmax_row_fused_kernel[grid](
            x_contig, y, D,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return y

    # Fallback for all other cases
    return torch.log_softmax(x, dim=dim)


class ModelNew(nn.Module):
    """
    Simple model that performs a LogSoftmax activation, accelerated with Triton on CUDA when possible.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _log_softmax_triton(x, self.dim)


batch_size = 16
dim = 16384

def get_inputs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(batch_size, dim, device=device, dtype=torch.float32)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed