# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

import triton
import triton.language as tl


N_TOKEN = 256
NO_BINS = 64
MIN_BIN = 2.3125
MAX_BIN = 21.6875
THRES = 8.0


@triton.jit
def _contact_prob64_kernel(
    logits_ptr,
    out_ptr,
    P: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, 64)

    mask_m = offs_m < P

    x = tl.load(
        logits_ptr + offs_m[:, None] * 64 + offs_k[None, :],
        mask=mask_m[:, None],
        other=-1.0e20,
    ).to(tl.float32)

    m = tl.max(x, axis=1)
    x = x - m[:, None]

    exp_x = tl.exp(x)

    denom = tl.sum(exp_x, axis=1)

    # 官方参数下 bins < 8.0 的 bin 数量是 19
    numer = tl.sum(
        tl.where(offs_k[None, :] < 19, exp_x, 0.0),
        axis=1,
    )

    out = numer / denom

    tl.store(out_ptr + offs_m, out, mask=mask_m)


def _contact_prob_ref(distogram_logits, min_bin, max_bin, no_bins, thres):
    prob = torch.softmax(distogram_logits, dim=-1)
    bins = torch.linspace(
        min_bin,
        max_bin,
        no_bins,
        device=distogram_logits.device,
        dtype=distogram_logits.dtype,
    )
    mask = bins < thres
    return prob[..., mask].sum(dim=-1)


class ModelNew(nn.Module):
    def __init__(
        self,
        min_bin: float = MIN_BIN,
        max_bin: float = MAX_BIN,
        no_bins: int = NO_BINS,
        thres: float = THRES,
    ):
        super().__init__()
        self.min_bin = float(min_bin)
        self.max_bin = float(max_bin)
        self.no_bins = int(no_bins)
        self.thres = float(thres)

    def forward(self, distogram_logits: torch.Tensor) -> torch.Tensor:
        N0, N1, B = distogram_logits.shape

        if (
            distogram_logits.is_cuda
            and B == 64
            and self.no_bins == 64
            and self.min_bin == MIN_BIN
            and self.max_bin == MAX_BIN
            and self.thres == THRES
        ):
            distogram_logits = distogram_logits.contiguous()
            out = torch.empty(
                (N0, N1),
                device=distogram_logits.device,
                dtype=distogram_logits.dtype,
            )

            P = N0 * N1
            grid = (triton.cdiv(P, 16),)

            _contact_prob64_kernel[grid](
                distogram_logits,
                out,
                P,
                BLOCK_M=16,
                num_warps=4,
            )

            return out

        return _contact_prob_ref(
            distogram_logits,
            self.min_bin,
            self.max_bin,
            self.no_bins,
            self.thres,
        )


def get_inputs():
    device = "cuda"
    torch.manual_seed(42)
    distogram_logits = torch.randn(
        N_TOKEN,
        N_TOKEN,
        NO_BINS,
        device=device,
    )
    return [distogram_logits]


def get_init_inputs():
    return [MIN_BIN, MAX_BIN, NO_BINS, THRES]

