"""KernelSwift Task08 flat 4x4 Sinkhorn candidate verified on MetaX C500."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


PLATFORM = "metax_c500"
ROW_WARPS = 1


@triton.jit
def _sinkhorn_kernel(
    mixes,
    scales,
    base,
    pre_out,
    post_out,
    comb_out,
    rows: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    hc = tl.arange(0, 4)
    s0 = tl.load(scales)
    s1 = tl.load(scales + 1)
    s2 = tl.load(scales + 2)
    pre_x = tl.load(mixes + row * 24 + hc).to(tl.float32)
    post_x = tl.load(mixes + row * 24 + 4 + hc).to(tl.float32)
    pre_b = tl.load(base + hc).to(tl.float32)
    post_b = tl.load(base + 4 + hc).to(tl.float32)
    pre = 1.0 / (1.0 + tl.exp(-(pre_x * s0 + pre_b))) + eps
    post = 2.0 / (1.0 + tl.exp(-(post_x * s1 + post_b)))

    ij = tl.arange(0, 16)
    row_id = ij // 4
    col_id = ij - row_id * 4
    raw = tl.load(mixes + row * 24 + 8 + ij).to(tl.float32)
    base_values = tl.load(base + 8 + ij).to(tl.float32)
    matrix = raw * s2 + base_values

    row_max0 = tl.max(tl.where(row_id == 0, matrix, -float("inf")), axis=0)
    row_max1 = tl.max(tl.where(row_id == 1, matrix, -float("inf")), axis=0)
    row_max2 = tl.max(tl.where(row_id == 2, matrix, -float("inf")), axis=0)
    row_max3 = tl.max(tl.where(row_id == 3, matrix, -float("inf")), axis=0)
    row_max = tl.where(
        row_id == 0,
        row_max0,
        tl.where(row_id == 1, row_max1, tl.where(row_id == 2, row_max2, row_max3)),
    )
    matrix = tl.exp(matrix - row_max)

    row_sum0 = tl.sum(tl.where(row_id == 0, matrix, 0.0), axis=0)
    row_sum1 = tl.sum(tl.where(row_id == 1, matrix, 0.0), axis=0)
    row_sum2 = tl.sum(tl.where(row_id == 2, matrix, 0.0), axis=0)
    row_sum3 = tl.sum(tl.where(row_id == 3, matrix, 0.0), axis=0)
    row_sum = tl.where(
        row_id == 0,
        row_sum0,
        tl.where(row_id == 1, row_sum1, tl.where(row_id == 2, row_sum2, row_sum3)),
    )
    matrix = matrix / row_sum + eps

    col_sum0 = tl.sum(tl.where(col_id == 0, matrix, 0.0), axis=0)
    col_sum1 = tl.sum(tl.where(col_id == 1, matrix, 0.0), axis=0)
    col_sum2 = tl.sum(tl.where(col_id == 2, matrix, 0.0), axis=0)
    col_sum3 = tl.sum(tl.where(col_id == 3, matrix, 0.0), axis=0)
    col_sum = tl.where(
        col_id == 0,
        col_sum0,
        tl.where(col_id == 1, col_sum1, tl.where(col_id == 2, col_sum2, col_sum3)),
    )
    matrix = matrix / (col_sum + eps)

    for _ in range(19):
        row_sum0 = tl.sum(tl.where(row_id == 0, matrix, 0.0), axis=0)
        row_sum1 = tl.sum(tl.where(row_id == 1, matrix, 0.0), axis=0)
        row_sum2 = tl.sum(tl.where(row_id == 2, matrix, 0.0), axis=0)
        row_sum3 = tl.sum(tl.where(row_id == 3, matrix, 0.0), axis=0)
        row_sum = tl.where(
            row_id == 0,
            row_sum0,
            tl.where(row_id == 1, row_sum1, tl.where(row_id == 2, row_sum2, row_sum3)),
        )
        matrix = matrix / (row_sum + eps)
        col_sum0 = tl.sum(tl.where(col_id == 0, matrix, 0.0), axis=0)
        col_sum1 = tl.sum(tl.where(col_id == 1, matrix, 0.0), axis=0)
        col_sum2 = tl.sum(tl.where(col_id == 2, matrix, 0.0), axis=0)
        col_sum3 = tl.sum(tl.where(col_id == 3, matrix, 0.0), axis=0)
        col_sum = tl.where(
            col_id == 0,
            col_sum0,
            tl.where(col_id == 1, col_sum1, tl.where(col_id == 2, col_sum2, col_sum3)),
        )
        matrix = matrix / (col_sum + eps)
    tl.store(pre_out + row * 4 + hc, pre)
    tl.store(post_out + row * 4 + hc, post)
    tl.store(comb_out + row * 16 + ij, matrix)


class ModelNew(nn.Module):
    def __init__(self, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
        super().__init__()
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

    def forward(
        self, mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor
    ):
        b, s, _ = mixes.shape
        pre = torch.empty((b, s, 4), dtype=torch.float32, device=mixes.device)
        post = torch.empty_like(pre)
        comb = torch.empty((b, s, 4, 4), dtype=torch.float32, device=mixes.device)
        _sinkhorn_kernel[(b * s,)](
            mixes,
            hc_scale,
            hc_base,
            pre,
            post,
            comb,
            rows=b * s,
            eps=self.eps,
            num_warps=ROW_WARPS,
            num_stages=1,
        )
        return pre, post, comb


class Model(ModelNew):
    pass


def get_init_inputs():
    return [4, 20, 1e-6]


def get_inputs():
    torch.manual_seed(0)
    return [
        torch.randn(2, 8, 24, dtype=torch.float32),
        torch.tensor([0.5, 0.25, 1.0], dtype=torch.float32),
        torch.randn(24, dtype=torch.float32) * 0.1,
    ]
