import math
import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _linear_gelu_softmax_rowwise(
    x_ptr,         # [B, K]
    w_ptr,         # [N, K]
    b_ptr,         # [N]
    y_ptr,         # [B, N]
    stride_x,      # stride between rows of x (in elements)
    stride_w_n,    # stride for weight along N (in elements)
    stride_w_k,    # stride for weight along K (in elements)
    stride_y,      # stride between rows of y (in elements)
    B, K, N,       # dimensions
    NUM_N_TILES: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    x_row_ptr = x_ptr + pid * stride_x
    y_row_ptr = y_ptr + pid * stride_y

    cols_n = tl.arange(0, BLOCK_N)
    cols_k = tl.arange(0, BLOCK_K)

    inv_sqrt2 = 0.7071067811865476
    neg_inf = -float("inf")

    # Fast path when the entire N fits in one tile: keep everything in registers
    if NUM_N_TILES == 1:
        j = cols_n
        j_mask = j < N

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        k_start = 0
        for _ in tl.static_range(NUM_K_TILES):
            k = k_start + cols_k
            k_mask = k < K

            x_vals = tl.load(x_row_ptr + k, mask=k_mask, other=0.0).to(tl.float32)

            w_ptrs = w_ptr + j[:, None] * stride_w_n + k[None, :] * stride_w_k
            wk_mask = j_mask[:, None] & k_mask[None, :]
            w_vals = tl.load(w_ptrs, mask=wk_mask, other=0.0, cache_modifier=".cg").to(tl.float32)

            acc += tl.sum(w_vals * x_vals[None, :], axis=1)
            k_start += BLOCK_K

        bias_vals = tl.load(b_ptr + j, mask=j_mask, other=0.0).to(tl.float32)
        logits = acc + bias_vals
        gelu_vals = 0.5 * logits * (1.0 + libdevice.erf(logits * inv_sqrt2))

        gelu_masked = tl.where(j_mask, gelu_vals, neg_inf)
        row_max = tl.max(gelu_masked, axis=0)
        z = gelu_vals - row_max
        num = tl.exp(z)
        num = tl.where(j_mask, num, 0.0)
        denom = tl.sum(num, axis=0)
        out = num / denom
        tl.store(y_row_ptr + j, out, mask=j_mask)
        return

    # Generic path for multi-tile N: 3-pass algorithm with minimal recomputation
    row_max = neg_inf
    n_start = 0
    for _ in tl.static_range(NUM_N_TILES):
        j = n_start + cols_n
        j_mask = j < N

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        k_start = 0
        for __ in tl.static_range(NUM_K_TILES):
            k = k_start + cols_k
            k_mask = k < K

            x_vals = tl.load(x_row_ptr + k, mask=k_mask, other=0.0).to(tl.float32)

            w_ptrs = w_ptr + j[:, None] * stride_w_n + k[None, :] * stride_w_k
            wk_mask = j_mask[:, None] & k_mask[None, :]
            w_vals = tl.load(w_ptrs, mask=wk_mask, other=0.0, cache_modifier=".cg").to(tl.float32)

            acc += tl.sum(w_vals * x_vals[None, :], axis=1)
            k_start += BLOCK_K

        bias_vals = tl.load(b_ptr + j, mask=j_mask, other=0.0).to(tl.float32)
        logits = acc + bias_vals
        gelu_vals = 0.5 * logits * (1.0 + libdevice.erf(logits * inv_sqrt2))

        tl.store(y_row_ptr + j, gelu_vals, mask=j_mask)
        row_max = tl.maximum(row_max, tl.max(tl.where(j_mask, gelu_vals, neg_inf), axis=0))
        n_start += BLOCK_N

    denom = 0.0
    n_start = 0
    for _ in tl.static_range(NUM_N_TILES):
        j = n_start + cols_n
        j_mask = j < N
        gelu_vals = tl.load(y_row_ptr + j, mask=j_mask, other=neg_inf)
        num = tl.exp(gelu_vals - row_max)
        tl.store(y_row_ptr + j, tl.where(j_mask, num, 0.0), mask=j_mask)
        denom += tl.sum(tl.where(j_mask, num, 0.0), axis=0)
        n_start += BLOCK_N

    inv_denom = 1.0 / denom
    n_start = 0
    for _ in tl.static_range(NUM_N_TILES):
        j = n_start + cols_n
        j_mask = j < N
        numer = tl.load(y_row_ptr + j, mask=j_mask, other=0.0)
        out = numer * inv_denom
        tl.store(y_row_ptr + j, out, mask=j_mask)
        n_start += BLOCK_N


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies GELU, and then applies Softmax.
    Uses a fused Triton kernel on CUDA; exact PyTorch ops on CPU.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # CUDA path: fuse Linear + GELU + Softmax using a single kernel
        if x.is_cuda:
            X = x.contiguous()
            W = self.linear.weight.contiguous()
            b = self.linear.bias
            if b is None:
                b = torch.zeros(W.shape[0], device=W.device, dtype=W.dtype)
            b = b.contiguous()

            B, K = X.shape
            N = W.shape[0]
            Y = torch.empty((B, N), device=X.device, dtype=X.dtype)

            BLOCK_N = min(128, max(16, _next_power_of_two(N)))
            BLOCK_K = min(128, max(32, _next_power_of_two(K)))
            NUM_N_TILES = (N + BLOCK_N - 1) // BLOCK_N
            NUM_K_TILES = (K + BLOCK_K - 1) // BLOCK_K

            # Heuristic for Hopper-class GPUs: small N -> fewer warps to reduce overhead
            num_warps = 1 if BLOCK_N <= 32 else 2

            grid = (B,)
            _linear_gelu_softmax_rowwise[grid](
                X, W, b, Y,
                X.stride(0), W.stride(0), W.stride(1), Y.stride(0),
                B, K, N,
                NUM_N_TILES=NUM_N_TILES,
                NUM_K_TILES=NUM_K_TILES,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
                num_warps=num_warps,
                num_stages=2,
            )
            return Y
        else:
            # CPU fallback: original sequence
            x = self.linear(x)
            x = torch.nn.functional.gelu(x)
            x = torch.nn.functional.softmax(x, dim=1)
            return x


batch_size = 128
in_features = 100
out_features = 10

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]