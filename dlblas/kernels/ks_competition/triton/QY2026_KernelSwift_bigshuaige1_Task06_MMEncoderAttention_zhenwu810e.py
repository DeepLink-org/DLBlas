"""KernelSwift Task06 single-block non-causal attention for zhenwu810e."""

import torch
import torch.nn as nn
import triton
import triton.language as tl


PLATFORM = "zhenwu810e"
MATRIX_WARPS = 4
PIPELINE_STAGES = 1
BLOCK_M = 16
BLOCK_N = 128


@triton.jit
def _batched_attention_kernel(q_ptr, k_ptr, v_ptr, out_ptr, n_tokens: tl.constexpr, scale: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr):
    q_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // 8
    head = batch_head - batch * 8
    qm = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    kn = tl.arange(0, BLOCK_N)
    d = tl.arange(0, D)
    q_base = ((batch * n_tokens + qm[:, None]) * 8 + head) * D
    k_base = ((batch * n_tokens + kn[None, :]) * 8 + head) * D
    q = tl.load(q_ptr + q_base + d[None, :], mask=qm[:, None] < n_tokens, other=0.0)
    k = tl.load(k_ptr + k_base + d[:, None], mask=kn[None, :] < n_tokens, other=0.0)
    scores = tl.dot(q, k) * scale
    scores = tl.where(qm[:, None] < n_tokens, tl.where(kn[None, :] < n_tokens, scores, -float("inf")), 0.0)
    row_max = tl.max(scores, axis=1)
    p = tl.exp(scores - row_max[:, None])
    p = p / tl.sum(p, axis=1)[:, None]
    v_base = ((batch * n_tokens + kn[:, None]) * 8 + head) * D
    v = tl.load(v_ptr + v_base + d[None, :], mask=kn[:, None] < n_tokens, other=0.0)
    out = tl.dot(p.to(tl.float16), v)
    out_base = ((batch * n_tokens + qm[:, None]) * 8 + head) * D
    tl.store(out_ptr + out_base + d[None, :], out, mask=qm[:, None] < n_tokens)


class ModelNew(nn.Module):
    def __init__(self, num_heads: int = 8, head_size: int = 64, num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = 1.0 / (head_size ** 0.5)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        out = torch.empty_like(query)
        bsz, seq_len = query.shape[:2]
        grid = (triton.cdiv(seq_len, BLOCK_M), bsz * 8)
        _batched_attention_kernel[grid](query, key, value, out, seq_len, scale=self.scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=64, num_warps=MATRIX_WARPS, num_stages=PIPELINE_STAGES)
        return out


class Model(ModelNew):
    pass


def get_inputs():
    return [torch.randn(2, 83, 512, dtype=torch.float16, device="cuda"), torch.randn(2, 83, 512, dtype=torch.float16, device="cuda"), torch.randn(2, 83, 512, dtype=torch.float16, device="cuda")]


def get_init_inputs():
    return [8, 64, 8]
