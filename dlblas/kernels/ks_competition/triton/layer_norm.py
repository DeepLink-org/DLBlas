import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layer_norm_lastdim_fwd_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Row pointers
    x_row_ptr = x_ptr + pid * stride_xm + offs * stride_xn
    y_row_ptr = y_ptr + pid * stride_ym + offs * stride_yn

    # Load row with masking
    x = tl.load(x_row_ptr, mask=mask, other=0.0)

    # Compute mean over the last dimension
    invN = 1.0 / N
    mean = tl.sum(x, axis=0) * invN

    # Center and mask tail lanes to avoid affecting variance
    diff = x - mean
    diff = tl.where(mask, diff, 0.0)

    # Compute variance and inverse stddev
    var = tl.sum(diff * diff, axis=0) * invN
    inv_std = tl.rsqrt(var + eps)

    # Normalize (gamma=1, beta=0 for default LayerNorm(10))
    y = diff * inv_std

    # Store results
    tl.store(y_row_ptr, y, mask=mask)


def _layer_norm_lastdim_triton(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # Fallback for non-CUDA tensors or unsupported dims (keeps functional parity)
    if not x.is_cuda or x.dim() != 2:
        return torch.nn.LayerNorm(10, eps=eps).to(x.device)(x)

    M, N = x.shape
    # Ensure we match the exact behavior of LayerNorm(10): normalize over last dim of size 10
    assert (
        N == 10
    ), "This optimized kernel assumes normalized_shape=10 as in the original program."

    # Use input strides directly to avoid extra copies
    y = torch.empty_like(x)

    BLOCK_SIZE = 32  # covers N=10, provides good vectorization
    grid = (M,)

    layer_norm_lastdim_fwd_kernel[grid](
        x,
        y,
        M,
        N,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
        num_stages=1,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        # Original: torch.nn.LayerNorm(10).cuda()(x)
        # Equivalent: LayerNorm over last dim=10 with gamma=1, beta=0, eps=1e-5
        return _layer_norm_lastdim_triton(x, eps=1e-5)


def get_inputs():
    x = torch.rand(10, 10, device="npu")
    return [x]


def get_init_inputs():
    return []


torch.manual_seed(42)
out = ModelNew(*get_init_inputs()).forward(*get_inputs())
print(out)
