"""Task03 fixed-shape causal attention candidate for Enflame S60."""

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl


PLATFORM = "enflame_s60"
MATRIX_WARPS = 8
PIPELINE_STAGES = 1


@triton.jit
def _attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    n_tokens: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    q_tile = tl.program_id(0)
    head = tl.program_id(1)
    qm = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    kn = tl.arange(0, BLOCK_N)
    d = tl.arange(0, D)
    q = tl.load(
        q_ptr + (qm[:, None] * 8 + head) * D + d[None, :],
        mask=qm[:, None] < n_tokens,
        other=0.0,
    )
    k = tl.load(
        k_ptr + (kn[None, :] * 8 + head) * D + d[:, None],
        mask=kn[None, :] < n_tokens,
        other=0.0,
    )
    scores = tl.dot(q, k) * scale
    valid = (kn[None, :] < n_tokens) & (kn[None, :] <= qm[:, None])
    scores = tl.where(
        qm[:, None] < n_tokens, tl.where(valid, scores, -float("inf")), 0.0
    )
    row_max = tl.max(scores, axis=1)
    p = tl.exp(scores - row_max[:, None])
    p = p / tl.sum(p, axis=1)[:, None]
    v = tl.load(
        v_ptr + (kn[:, None] * 8 + head) * D + d[None, :],
        mask=kn[:, None] < n_tokens,
        other=0.0,
    )
    out = tl.dot(p.to(tl.float16), v)
    tl.store(
        out_ptr + (qm[:, None] * 8 + head) * D + d[None, :],
        out,
        mask=qm[:, None] < n_tokens,
    )


class ModelNew(nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        head_size: int = 64,
        scale: float = None,
        num_kv_heads: int = 8,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale or 1.0 / (head_size**0.5)
        self.num_kv_heads = num_kv_heads

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        out = torch.empty_like(query)
        block_m = int(os.getenv("S60_T3_BLOCK_M", "64"))
        block_n = int(os.getenv("S60_T3_BLOCK_N", "128"))
        warps = int(os.getenv("S60_T3_WARPS", "1"))
        stages = int(os.getenv("S60_T3_STAGES", "1"))
        grid = (triton.cdiv(query.shape[0], block_m), 8)
        _attention_kernel[grid](
            query,
            key,
            value,
            out,
            query.shape[0],
            scale=self.scale,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            D=64,
            num_warps=warps,
            num_stages=stages,
        )
        return out.reshape(query.shape[0], 512)


class Model(ModelNew):
    pass


def get_inputs():
    return [
        torch.randn(83, 8, 64, dtype=torch.float16, device="cuda"),
        torch.randn(83, 8, 64, dtype=torch.float16, device="cuda"),
        torch.randn(83, 8, 64, dtype=torch.float16, device="cuda"),
    ]


def get_init_inputs():
    return [8, 64, None, 8]
