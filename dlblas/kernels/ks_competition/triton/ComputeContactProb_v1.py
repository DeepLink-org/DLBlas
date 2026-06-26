# -*- coding: utf-8 -*-
import torch
import torch_npu
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def contact_prob_kernel(
    logits_ptr,
    out_ptr,
    n_token: tl.constexpr,
    n_bin: tl.constexpr,
    thres_idx,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_m_blocks = tl.cdiv(n_token, BLOCK_SIZE_M)
    pid_m = pid // num_m_blocks
    pid_n = pid % num_m_blocks

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_bin = tl.arange(0, n_bin)

    mask_m = offs_m < n_token
    mask_n = offs_n < n_token

    logits = tl.load(
        logits_ptr + offs_m[:, None, None] * n_token * n_bin
        + offs_n[None, :, None] * n_bin
        + offs_bin[None, None, :],
        mask=mask_m[:, None, None] & mask_n[None, :, None],
        other=-float("inf")
    )

    # Stable softmax
    logits_max = tl.max(logits, axis=-1, keep_dims=True)
    exp_logits = tl.exp(logits - logits_max)
    sum_exp = tl.sum(exp_logits, axis=-1, keep_dims=True)
    prob = exp_logits / sum_exp

    # ========== 娣囶喖顦查敍姘瑝閻€劌鍨忛悧鍥风礉閻€劍甯洪惍浣虹柈閸旂姴澧� thres_idx 娑擄拷 bin ==========
    bin_mask = offs_bin[None, None, :] < thres_idx
    contact = tl.sum(prob * bin_mask, axis=-1)

    tl.store(
        out_ptr + offs_m[:, None] * n_token + offs_n[None, :],
        contact,
        mask=mask_m[:, None] & mask_n[None, :]
    )


def triton_contact_prob_forward(logits: torch.Tensor, thres_idx: int):
    assert logits.dim() == 3
    n_token, n_token2, n_bin = logits.shape
    assert n_token == n_token2

    out = torch.empty((n_token, n_token), dtype=logits.dtype, device=logits.device)
    BLOCK_M = 16
    BLOCK_N = 16
    grid = (triton.cdiv(n_token, BLOCK_M) * triton.cdiv(n_token, BLOCK_N),)

    contact_prob_kernel[grid](
        logits,
        out,
        n_token=n_token,
        n_bin=n_bin,
        thres_idx=thres_idx,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, min_bin: float = 2.3125, max_bin: float = 21.6875, no_bins: int = 64, thres: float = 8.0):
        super().__init__()
        self.no_bins = int(no_bins)

        edges = torch.linspace(min_bin, max_bin, self.no_bins + 1)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        self.thres_idx = int((bin_centers < thres).sum().item())

    def forward(self, distogram_logits: torch.Tensor) -> torch.Tensor:
        return triton_contact_prob_forward(distogram_logits, self.thres_idx)


N_TOKEN = 256
NO_BINS = 64
MIN_BIN = 2.3125
MAX_BIN = 21.6875
THRES = 8.0

def get_inputs():
    device = 'npu'
    torch.manual_seed(42)
    distogram_logits = torch.randn(N_TOKEN, N_TOKEN, NO_BINS, device=device)
    return [distogram_logits]

def get_init_inputs():
    return [MIN_BIN, MAX_BIN, NO_BINS, THRES]

if __name__ == "__main__":
    torch_npu.npu.set_device(0)
    device = torch.device("npu:0")

    model = ModelNew(*get_init_inputs()).to(device)
    inputs = [x.to(device) for x in get_inputs()]
    with torch.npu.amp.autocast(dtype=torch.float16):
        res = model(*inputs)
    print(res.shape)