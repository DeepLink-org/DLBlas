from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from clike_910b import load_library


world_size = 1


@dataclass
class ModelArgs:
    max_batch_size: int = 4
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: Literal[None, "ue8m0"] = "ue8m0"
    expert_dtype: Literal[None, "fp4"] = None
    scale_dtype: Literal["fp32", "fp8"] = "fp8"
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 4096
    n_layers: int = 7
    n_hash_layers: int = 0
    n_mtp_layers: int = 1
    n_heads: int = 64
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.0
    swiglu_limit: float = 0.0
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: Tuple[int, ...] = (0, 0, 4, 128, 4, 128, 4, 0)
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6


class ModelNew(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        freqs_cis: torch.Tensor,
        kv_cache: torch.Tensor,
        compress_ratio: int = 4,
    ):
        super().__init__()
        if (
            args.index_n_heads != 16
            or args.index_head_dim != 64
            or args.rope_head_dim != 32
        ):
            raise ValueError(
                "The optimized Indexer expects 16 heads, head_dim=64, and rope_head_dim=32"
            )
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = nn.Linear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.weights_proj = nn.Linear(
            self.dim, self.n_heads, bias=False, dtype=torch.bfloat16
        )
        self.softmax_scale = self.head_dim**-0.5
        self.compress_ratio = compress_ratio
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.register_buffer("kv_cache", kv_cache, persistent=False)

        aligned_cache_len = (kv_cache.size(1) + 15) // 16 * 16
        self.register_buffer(
            "_kv_cache_aligned",
            F.pad(kv_cache, (0, 0, 0, aligned_cache_len - kv_cache.size(1))),
            persistent=False,
        )
        self.register_buffer(
            "_valid_key_counts",
            torch.arange(
                1,
                args.max_seq_len + 1,
                dtype=torch.int64,
                device=kv_cache.device,
            ).unsqueeze(1)
            // compress_ratio,
            persistent=False,
        )

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        load_library()
        batch_size, seq_len, _ = x.shape
        end_pos = start_pos + seq_len
        key_len = end_pos // self.compress_ratio
        if key_len > 650:
            raise ValueError(
                "The optimized Indexer supports at most 650 compressed keys"
            )

        q = F.linear(qr, self.wq_b.weight)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim)).contiguous()
        rope = torch.view_as_complex(
            q[..., -self.rope_head_dim :].float().unflatten(-1, (-1, 2))
        )
        freqs = self.freqs_cis[start_pos:end_pos].view(1, seq_len, 1, -1)
        q[..., -self.rope_head_dim :].copy_(
            torch.view_as_real(rope * freqs).flatten(-2)
        )

        weight_scale = self.softmax_scale * self.n_heads**-0.5
        weights = (F.linear(x, self.weights_proj.weight) * weight_scale).contiguous()
        physical_key_len = (key_len + 15) // 16 * 16
        key = self._kv_cache_aligned[:batch_size, :physical_key_len]

        scores = torch.bmm(
            q.view(batch_size, seq_len * self.n_local_heads, self.head_dim),
            key.transpose(1, 2),
        ).view(batch_size, seq_len, self.n_local_heads, physical_key_len)
        reduced = torch.ops.dlblas_ks_ascendc.indexer_reduce(
            scores,
            weights,
            self.compress_ratio,
            start_pos == 0,
            key_len,
        )

        count = min(self.index_topk, key_len)
        topk_idxs = reduced.topk(count, dim=-1).indices
        if start_pos == 0:
            valid = self._valid_key_counts[:seq_len]
            topk_idxs = torch.where(topk_idxs >= valid, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs


def _make_args():
    return ModelArgs(
        max_batch_size=8,
        max_seq_len=2600,
        dim=1024,
        index_n_heads=16,
        index_head_dim=64,
        index_topk=128,
        q_lora_rank=256,
        rope_head_dim=32,
    )


def get_inputs():
    config = _make_args()
    batch_size = 8
    seq_len = 2600
    x = torch.randn(
        batch_size,
        seq_len,
        config.dim,
        dtype=torch.bfloat16,
        device="npu",
    )
    qr = torch.randn(
        batch_size,
        seq_len,
        config.q_lora_rank,
        dtype=torch.bfloat16,
        device="npu",
    )
    return [x, qr, 0, 0]


def get_init_inputs():
    config = _make_args()
    compress_ratio = 4
    freqs = 1.0 / (
        10000.0
        ** (
            torch.arange(0, config.rope_head_dim, 2)[
                : config.rope_head_dim // 2
            ].float()
            / config.rope_head_dim
        )
    )
    positions = torch.arange(config.max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs).float().npu()
    freqs_cis = torch.polar(torch.ones_like(angles).npu(), angles).view(
        config.max_seq_len, -1
    )
    kv_cache = torch.randn(
        config.max_batch_size,
        config.max_seq_len // compress_ratio,
        config.index_head_dim,
        dtype=torch.bfloat16,
    ).npu()
    return [config, freqs_cis, kv_cache, compress_ratio]
