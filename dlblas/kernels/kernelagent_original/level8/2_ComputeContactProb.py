"""
Compute Contact Probability (distogram logits -> contact probability)

From: protenix/model/sample_confidence.py:compute_contact_prob
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


def get_bin_centers(min_bin: float, max_bin: float, no_bins: int) -> torch.Tensor:
    """
    distogram bins centers（常见做法：线性等间隔）
    """
    edges = torch.linspace(min_bin, max_bin, no_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers


@triton.jit
def _contact_prob_kernel_2d(
    logits_ptr,          # *f32 / *f16, [M, N, C]
    out_ptr,             # *f32 / *f16, [M, N]
    M, N, C,             # sizes
    K,                   # thres_idx: number of first bins to sum
    stride_m, stride_n, stride_c,   # input strides (in elements)
    out_stride_m, out_stride_n,     # output strides (in elements)
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n_blk = tl.program_id(1)

    # Offsets along N (second) dimension for this program
    offs_n = pid_n_blk * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = tl.arange(0, BLOCK_C)

    mask_n = offs_n < N
    mask_c = offs_c < C

    # Pointers to a tile [BLOCK_N, BLOCK_C] at fixed m = pid_m
    ptrs = logits_ptr + pid_m * stride_m + offs_n[:, None] * stride_n + offs_c[None, :] * stride_c

    # Load logits; use -inf for masked elements so they don't affect max
    x = tl.load(ptrs, mask=mask_n[:, None] & mask_c[None, :], other=-float("inf")).to(tl.float32)

    # Numerically stable softmax components along C per (m, n) row
    row_max = tl.max(x, axis=1)  # [BLOCK_N]; will be -inf for invalid rows
    # For invalid rows, set row_max to 0 to avoid NaNs in (x - row_max)
    row_max = tl.where(mask_n, row_max, 0.0)
    x = x - row_max[:, None]

    # Exponentials; masked columns already mapped to -inf so exp -> 0
    e = tl.exp(x)

    # Denominator: sum over all bins
    denom = tl.sum(e, axis=1)  # [BLOCK_N]

    # Numerator: sum over first K bins; no need to additionally mask by C since e=0 beyond C
    take_mask = offs_c < K  # [BLOCK_C]
    num = tl.sum(e * take_mask[None, :], axis=1)  # [BLOCK_N]

    # Avoid invalid divisions for out-of-bound rows
    inv_denom = tl.where(mask_n, 1.0 / denom, 0.0)
    prob = num * inv_denom  # [BLOCK_N], valid only where mask_n is True

    # Store output
    out_ptrs = out_ptr + pid_m * out_stride_m + offs_n * out_stride_n
    tl.store(out_ptrs, prob, mask=mask_n)


def compute_contact_prob(
    distogram_logits: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    thres: float = 8.0,
) -> torch.Tensor:
    """
    Args:
        distogram_logits: [N_token, N_token, no_bins]
    Returns:
        contact_prob: [N_token, N_token]
    """
    # Compute threshold index exactly as in reference implementation
    bins = get_bin_centers(min_bin, max_bin, no_bins).to(distogram_logits.device)
    thres_idx = int((bins < thres).sum().item())

    # Fallback if tensor is not on CUDA
    if not distogram_logits.is_cuda:
        distogram_prob = torch.softmax(distogram_logits, dim=-1)
        return distogram_prob[..., :thres_idx].sum(dim=-1)

    M, N, C = distogram_logits.shape
    assert C == no_bins, "Last dimension must equal no_bins"

    # Output tensor [M, N] with same dtype/device as input
    out = torch.empty((M, N), device=distogram_logits.device, dtype=distogram_logits.dtype)

    # Strides (in elements)
    stride_m, stride_n, stride_c = distogram_logits.stride()
    out_stride_m, out_stride_n = out.stride()

    # Choose tiling parameters
    def next_pow2(x: int) -> int:
        return 1 << (int(x) - 1).bit_length()

    BLOCK_C = min(128, next_pow2(int(C)))
    BLOCK_N = 16  # tile multiple rows in N to amortize kernel overhead

    grid = (M, triton.cdiv(N, BLOCK_N))
    _contact_prob_kernel_2d[grid](
        distogram_logits,
        out,
        M, N, C,
        thres_idx,
        stride_m, stride_n, stride_c,
        out_stride_m, out_stride_n,
        BLOCK_N=BLOCK_N,
        BLOCK_C=BLOCK_C,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, min_bin: float = 2.3125, max_bin: float = 21.6875, no_bins: int = 64, thres: float = 8.0):
        super().__init__()
        self.min_bin = float(min_bin)
        self.max_bin = float(max_bin)
        self.no_bins = int(no_bins)
        self.thres = float(thres)

    def forward(self, distogram_logits: torch.Tensor) -> torch.Tensor:
        return compute_contact_prob(
            distogram_logits=distogram_logits,
            min_bin=self.min_bin,
            max_bin=self.max_bin,
            no_bins=self.no_bins,
            thres=self.thres,
        )


# ==========================================
# Hyperparameters & Data Generation
# ==========================================

N_TOKEN = 256
NO_BINS = 64
MIN_BIN = 2.3125
MAX_BIN = 21.6875
THRES = 8.0


def get_inputs():
    device = 'cuda'
    torch.manual_seed(42)

    distogram_logits = torch.randn(N_TOKEN, N_TOKEN, NO_BINS, device=device)

    return [distogram_logits]


def get_init_inputs():
    return [MIN_BIN, MAX_BIN, NO_BINS, THRES]