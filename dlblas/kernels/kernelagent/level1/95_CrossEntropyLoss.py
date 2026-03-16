import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _cross_entropy_rowwise_kernel(
    x_ptr,             # *f32 [N, C]
    t_ptr,             # *i64 [N]
    out_ptr,           # *f32 [N]
    stride_x_batch,    # int
    stride_x_class,    # int
    N,                 # int
    C,                 # int
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # row id
    row_in_bounds = pid < N

    # Pointers to the start of this row
    row_ptr = x_ptr + pid * stride_x_batch

    # Load logits of this row
    offs = tl.arange(0, BLOCK_SIZE)
    mask_cls = offs < C
    mask = mask_cls & row_in_bounds

    x = tl.load(row_ptr + offs * stride_x_class, mask=mask, other=-float("inf"))

    # numerically stable log-sum-exp
    m = tl.max(x, axis=0)
    x_shift = x - m
    expx = tl.exp(x_shift)
    sumexp = tl.sum(expx, axis=0)
    logsumexp = tl.log(sumexp) + m

    # Load target index and gather target logit
    tgt = tl.load(t_ptr + pid, mask=row_in_bounds, other=0)
    # ensure index dtype for address arithmetic
    tgt = tgt.to(tl.int64)
    x_t = tl.load(row_ptr + tgt * stride_x_class, mask=row_in_bounds, other=0.0)

    # per-sample negative log-likelihood
    nll = logsumexp - x_t

    # Write output
    tl.store(out_ptr + pid, nll, mask=row_in_bounds)


class ModelNew(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Fallback to PyTorch if not CUDA
        if not predictions.is_cuda or not targets.is_cuda:
            return torch.nn.functional.cross_entropy(predictions, targets)

        # Shapes
        N, C = predictions.shape
        x = predictions
        t = targets

        # Ensure proper dtypes and memory layout
        x = x.contiguous()
        t = t.contiguous()

        # Allocate output buffer for per-sample losses
        out = torch.empty(N, device=x.device, dtype=x.dtype)

        # Next power-of-two block size for classes dimension
        BLOCK_SIZE = 1 << (C - 1).bit_length()

        grid = lambda meta: (N,)

        _cross_entropy_rowwise_kernel[grid](
            x,
            t,
            out,
            x.stride(0),
            x.stride(1),
            N,
            C,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Mean reduction to match torch.nn.functional.cross_entropy default
        return out.mean()


batch_size = 4096
num_classes = 10
input_shape = (num_classes, )  # Output for each class
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randint(0, num_classes, (batch_size,))]

def get_init_inputs():
    return []
