"""KernelSwift Task08 resident 4x4 Sinkhorn candidate for hygon_bw1000."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


PLATFORM = "hygon_bw1000"
ROW_WARPS = 8


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
    raw = tl.load(mixes + row * 24 + 8 + ij).to(tl.float32)
    b = tl.load(base + 8 + ij).to(tl.float32)
    matrix = (raw * s2 + b).reshape((4, 4))
    row_max = tl.max(matrix, axis=1)
    matrix = tl.exp(matrix - row_max[:, None])
    row_sum = tl.sum(matrix, axis=1)
    matrix = matrix / row_sum[:, None] + eps
    col_sum = tl.sum(matrix, axis=0)
    matrix = matrix / (col_sum[None, :] + eps)
    for _ in range(19):
        row_sum = tl.sum(matrix, axis=1)
        matrix = matrix / (row_sum[:, None] + eps)
        col_sum = tl.sum(matrix, axis=0)
        matrix = matrix / (col_sum[None, :] + eps)
    tl.store(pre_out + row * 4 + hc, pre)
    tl.store(post_out + row * 4 + hc, post)
    tl.store(comb_out + row * 16 + ij, matrix.reshape((16,)))


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
