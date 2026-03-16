import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_mish2_rowwise(
    x_ptr,        # [M, K]
    w_ptr,        # [N, K] (row-major: weight)
    b_ptr,        # [N]
    y_ptr,        # [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    # Guard against empty grids
    if pid_m >= M:
        return

    offs_n = tl.arange(0, BLOCK_N)
    # Accumulator for this row over a block of N outputs
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Base pointers
    x_row_ptr = x_ptr + pid_m * stride_xm

    # Iterate over K in chunks
    k0 = 0
    while k0 < K:
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # Load a chunk of the input row x[m, k]
        x_vals = tl.load(x_row_ptr + offs_k * stride_xk, mask=k_mask, other=0.0).to(tl.float32)  # [BLOCK_K]

        # Load corresponding weights W[n, k] for a block of N outputs
        w_ptrs = w_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)  # [BLOCK_N, BLOCK_K]
        w_mask = (offs_n[:, None] < N) & (k_mask[None, :])
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)  # [BLOCK_N, BLOCK_K]

        # Accumulate dot product for this N-block: sum_k W[n, k] * x_vals[k]
        prods = w_vals * x_vals[None, :]
        acc += tl.sum(prods, axis=1)

        k0 += BLOCK_K

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias

    # Apply Mish twice with PyTorch-like softplus threshold behavior (threshold=20)
    th = 20.0

    # First Mish: x * tanh(softplus(x))
    abs_acc = tl.abs(acc)
    sp1_stable = tl.log(1.0 + tl.exp(-abs_acc)) + tl.maximum(acc, 0.0)
    sp1 = tl.where(acc > th, acc, sp1_stable)
    t1 = tl.exp(-2.0 * sp1)
    tanh_sp1 = 1.0 - 2.0 * t1 / (1.0 + t1)
    m1 = acc * tanh_sp1

    # Second Mish on m1
    abs_m1 = tl.abs(m1)
    sp2_stable = tl.log(1.0 + tl.exp(-abs_m1)) + tl.maximum(m1, 0.0)
    sp2 = tl.where(m1 > th, m1, sp2_stable)
    t2 = tl.exp(-2.0 * sp2)
    tanh_sp2 = 1.0 - 2.0 * t2 / (1.0 + t2)
    out = m1 * tanh_sp2

    # Store results
    y_ptrs = y_ptr + pid_m * stride_ym + offs_n * stride_yn
    tl.store(y_ptrs, out, mask=offs_n < N)


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Mish, and applies Mish again.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Fallback to PyTorch path when Triton path is not supported
        use_triton = x.is_cuda and (x.dtype == torch.float32)
        if not use_triton:
            x = self.linear(x)
            x = torch.nn.functional.mish(x)
            x = torch.nn.functional.mish(x)
            return x

        # Inputs
        A = x.contiguous()  # [M, K]
        W = self.linear.weight.contiguous()  # [N, K]
        B = self.linear.bias.contiguous() if (self.linear.bias is not None) else torch.zeros(W.shape[0], device=A.device, dtype=A.dtype)

        M, K = A.shape
        N = W.shape[0]
        Y = torch.empty((M, N), device=A.device, dtype=A.dtype)

        if M == 0 or N == 0:
            return Y

        # Launch one row-wise CTA per input row
        grid = (M,)
        BLOCK_N = 32
        BLOCK_K = 8
        linear_mish2_rowwise[grid](
            A, W, B, Y,
            M, N, K,
            A.stride(0), A.stride(1),
            W.stride(0), W.stride(1),
            Y.stride(0), Y.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=1,
        )
        return Y


batch_size = 128
in_features = 10
out_features = 20

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]