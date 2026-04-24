import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_dot_fused_kernel(
    X_ptr,  # float32 [M, K]
    W_ptr,  # float32 [N, K]
    O_ptr,  # float32 [M, N]
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    sq_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    k = 0
    while k < K:
        k_offsets = k + offs_k
        k_mask = k_offsets < K

        # Load X tile: [BM, BK]
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + k_offsets[None, :] * stride_xk)
        x = tl.load(x_ptrs, mask=(mask_m[:, None] & k_mask[None, :]), other=0.0)

        # Load W tile as [BN, BK] for better access, then transpose on dot
        w_ptrs = W_ptr + (offs_n[:, None] * stride_wn + k_offsets[None, :] * stride_wk)
        w = tl.load(w_ptrs, mask=(mask_n[:, None] & k_mask[None, :]), other=0.0)

        # Accumulate partial GEMM and squared sum for RMS
        acc += tl.dot(x, tl.trans(w))
        sq_acc += tl.sum(x * x, axis=1)

        k += BLOCK_K

    # Compute per-row RMS scale: rsqrt(mean(x^2) + eps)
    scale = tl.rsqrt(sq_acc / K + eps)
    acc = acc * scale[:, None]

    # Store output
    o_ptrs = O_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(o_ptrs, acc, mask=(mask_m[:, None] & mask_n[None, :]))


class ModelNew(nn.Module):

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        residual: torch.Tensor,
        mhc_fn: torch.Tensor,
        mhc_norm_weight: torch.Tensor | None,
        mhc_norm_eps: float,
    ) -> torch.Tensor:
        torch.backends.cuda.matmul.allow_tf32 = True
        # Apply optional mhc_norm_weight exactly as in reference
        if mhc_norm_weight is not None:
            mhc_fn = mhc_fn * mhc_norm_weight
        # Flatten and cast to float per reference semantics
        residual = residual.flatten(2, 3).float()
        assert mhc_fn.dtype == residual.dtype == torch.float

        # Shapes:
        # residual: (n0, n1, K)
        # mhc_fn: (N, K)
        n0, n1, K = residual.shape
        N = mhc_fn.shape[0]
        M = n0 * n1

        # Flatten residual over (n0*n1, K)
        X = residual.reshape(M, K).contiguous()
        W = mhc_fn.contiguous()
        O = torch.empty((M, N), dtype=torch.float32, device=residual.device)

        # Strides in elements
        stride_xm, stride_xk = X.stride()
        stride_wn, stride_wk = W.stride()
        stride_om, stride_on = O.stride()

        # Choose block sizes with small-shape awareness and large-K tiling
        def pick_blocks(M, N, K):
            if K >= 4096:
                bk = 512
                num_stages = 4
            else:
                bk = 256
                num_stages = 3
            # favor small tiles for small M/N to keep registers/latency balanced
            bm = 16 if M <= 16 else (32 if M <= 64 else 64)
            bn = 32 if N <= 32 else (64 if N <= 128 else 128)
            return bm, bn, bk, num_stages

        bm, bn, bk, num_stages = pick_blocks(M, N, K)
        grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))

        # Choose warps based on tile size to balance occupancy vs ILP
        num_warps = 4 if (bm * bn) >= 512 else 2

        rms_dot_fused_kernel[grid](
            X, W, O,
            M, N, K,
            stride_xm, stride_xk,
            stride_wn, stride_wk,
            stride_om, stride_on,
            mhc_norm_eps,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        mixes = O.reshape(n0, n1, N)
        torch.backends.cuda.matmul.allow_tf32 = False
        return mixes

n1=13
mhc_mult=4
hidden_size=1280
generate_normw=False

def generate_norm_fn_test_data(
    n1: int,
    mhc_mult: int,
    hidden_size: int,
    generate_normw: bool,
) -> dict[str, torch.Tensor]:
    n0 = 1
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    mhc_hidden_size = mhc_mult * hidden_size

    residual = (
        torch.randn((n0, n1, mhc_mult, hidden_size), dtype=torch.float)
        .mul(1 + torch.arange(mhc_mult).mul(0.01).view(1, 1, -1, 1))
        .bfloat16()
    )

    fn = (
        torch.randn((mhc_mult3, mhc_mult, hidden_size), dtype=torch.float)
        * 1e-4
        * (1 + torch.arange(mhc_mult).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)

    if generate_normw:
        normw = torch.randn((mhc_hidden_size,), dtype=torch.float) * 0.1 + 1.0
    else:
        normw = None

    out_grad = torch.randn((n0, n1, mhc_mult3), dtype=torch.float)

    return [residual,fn,normw,out_grad,1e-6]


def get_inputs():
    residual, fn, normw, out_grad, mhc_norm_eps = generate_norm_fn_test_data(n1, mhc_mult, hidden_size, generate_normw)
    return [residual, fn, None, mhc_norm_eps]

def get_init_inputs():
    return []
