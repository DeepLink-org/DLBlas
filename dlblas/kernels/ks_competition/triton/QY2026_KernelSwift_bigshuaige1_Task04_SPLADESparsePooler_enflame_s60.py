"""Task04 fused activation and ragged max-pool candidate for Enflame S60."""

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl


PLATFORM = "enflame_s60"
ROW_WARPS = 1


@triton.jit
def _activation_pool_kernel(logits_ptr, lens_ptr, out_ptr,
                            vocab: tl.constexpr, BLOCK_M: tl.constexpr,
                            BLOCK_N: tl.constexpr):
    vocab_tile = tl.program_id(0)
    seq = tl.program_id(1)
    start = 0
    for i in range(4):
        length_i = tl.load(lens_ptr + i)
        start += tl.where(seq > i, length_i, 0)
    length = tl.load(lens_ptr + seq)
    n = vocab_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    pooled = tl.full((BLOCK_N,), -float("inf"), dtype=tl.float32)
    for i in range(BLOCK_M):
        logits = tl.load(
            logits_ptr + (start + i) * vocab + n,
            mask=(i < length) & (n < vocab),
            other=-float("inf"),
        ).to(tl.float32)
        activated = tl.log(1.0 + tl.maximum(logits, 0.0))
        pooled = tl.maximum(pooled, tl.where(i < length, activated,
                                             -float("inf")))
    tl.store(out_ptr + seq * vocab + n, pooled, mask=n < vocab)


class ModelNew(nn.Module):
    def __init__(self, hidden_size: int = 768, vocab_size: int = 30522,
                 pooling: str = "max"):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)
        self.pooling = pooling

    def forward(self, hidden_states: torch.Tensor, seq_lens: torch.Tensor):
        logits = self.decoder(self.layer_norm(self.act(self.dense(hidden_states))))
        result = torch.empty(
            (seq_lens.shape[0], self.decoder.out_features),
            dtype=logits.dtype, device=logits.device,
        )
        block_m = int(os.getenv("S60_T4_BLOCK_M", "32"))
        block_n = int(os.getenv("S60_T4_BLOCK_N", "2048"))
        warps = int(os.getenv("S60_T4_WARPS", "1"))
        grid = (triton.cdiv(self.decoder.out_features, block_n), seq_lens.shape[0])
        _activation_pool_kernel[grid](
            logits, seq_lens, result, vocab=self.decoder.out_features,
            BLOCK_M=block_m, BLOCK_N=block_n, num_warps=warps, num_stages=1,
        )
        return [result[i] for i in range(seq_lens.shape[0])]


class Model(ModelNew):
    pass


def get_inputs():
    return [
        torch.randn(83, 768, device="cuda"),
        torch.tensor([20, 25, 18, 20], dtype=torch.int32, device="cuda"),
    ]


def get_init_inputs():
    return [768, 30522, "max"]
