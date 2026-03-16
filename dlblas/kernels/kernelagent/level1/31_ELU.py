import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _elu_kernel(x_ptr, y_ptr, N, alpha, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Help compiler with vectorized/coalesced memory ops
    tl.max_contiguous(offs, 128)

    x = tl.load(x_ptr + offs, mask=mask, other=0)

    # ELU: y = x if x > 0 else alpha * (exp(x) - 1)
    # Use x_neg to avoid exponentiating positive values
    x_neg = tl.minimum(x, 0)
    # Faster exponent: exp(x) = 2^(x / ln(2))
    inv_ln2 = 1.4426950408889634
    exp_term = tl.exp2(x_neg * inv_ln2)
    neg_part = (exp_term - 1) * alpha
    y = tl.where(x > 0, x, neg_part)

    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    """
    Simple model that performs an ELU activation.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the ELU model.

        Args:
            alpha (float, optional): The alpha parameter for the ELU function. Defaults to 1.0.
        """
        super(ModelNew, self).__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ELU applied, same shape as input.
        """
        # Fallback to PyTorch for CPU, unsupported dtypes, or if gradients are required.
        if (not x.is_cuda) or x.requires_grad or (x.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
            return F.elu(x, alpha=self.alpha)

        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)
        N = x_contig.numel()

        # Choose a good tile and launch config without autotune overhead
        if N >= 131072:
            BLOCK_SIZE = 8192
            num_warps = 8
        elif N >= 32768:
            BLOCK_SIZE = 4096
            num_warps = 8
        elif N >= 8192:
            BLOCK_SIZE = 2048
            num_warps = 4
        else:
            BLOCK_SIZE = 1024
            num_warps = 4

        def grid(meta):
            return (triton.cdiv(N, meta['BLOCK_SIZE']),)

        _elu_kernel[grid](x_contig, y, N, self.alpha, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=2)
        return y

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return [1.0]  # Provide alpha value for initialization