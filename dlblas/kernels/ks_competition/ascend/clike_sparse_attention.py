from __future__ import annotations

import torch
from torch import nn

from clike_910b import load_library


class ModelNew(nn.Module):
    def __init__(self, n_heads: int, head_dim: int):
        super().__init__()
        if head_dim != 128:
            raise ValueError("The optimized AscendC kernel requires head_dim=128")
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.softmax_scale = head_dim**-0.5
        self.attn_sink = nn.Parameter(torch.zeros(n_heads, dtype=torch.float32))

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_idxs: torch.Tensor,
    ) -> torch.Tensor:
        if topk_idxs.shape[-1] != 16:
            raise ValueError("The optimized AscendC kernel requires topk=16")
        load_library()
        indices = topk_idxs.contiguous()
        valid = indices >= 0
        gathered = torch.ops.dlblas_ks_ascendc.sparse_gather(kv.contiguous(), indices)
        q_cube = q.to(torch.float16)
        gathered_cube = gathered.to(torch.float16)
        scores = (
            torch.einsum("bmhd,bmtd->bmht", q_cube, gathered_cube).float()
            * self.softmax_scale
        )
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
        sink = self.attn_sink.float().view(1, 1, self.n_heads, 1)
        maximum = torch.maximum(scores.amax(-1, keepdim=True), sink)
        exponent = torch.exp(scores - maximum).masked_fill(~valid.unsqueeze(2), 0)
        weights = exponent / (
            exponent.sum(-1, keepdim=True) + torch.exp(sink - maximum)
        )
        return torch.einsum(
            "bmht,bmtd->bmhd", weights.to(torch.float16), gathered_cube
        ).to(q.dtype)


batch_size = 8
seq_len = 2600
n_kv = 32
n_heads = 64
head_dim = 128
topk = 16


def get_inputs():
    q = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=torch.bfloat16)
    kv = torch.randn(batch_size, n_kv, head_dim, dtype=torch.bfloat16)
    topk_idxs = torch.randint(0, n_kv, (batch_size, seq_len, topk), dtype=torch.int32)
    return [q, kv, topk_idxs]


def get_init_inputs():
    return [n_heads, head_dim]
