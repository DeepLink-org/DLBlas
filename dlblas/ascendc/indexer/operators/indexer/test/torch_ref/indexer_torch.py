# PyTorch reference implementation of Indexer operator
# Used as ground truth for AscendC kernel verification
# Adapted from origin/indexer.py with world_size=1 (no TP)

import torch
import torch.nn.functional as F
from typing import Optional


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies rotary positional embeddings to the last rope_head_dim elements of x.
    x: (B, S, H, D) or (B, S, D)
    freqs_cis: (S, rd//2) complex
    """
    rd2 = freqs_cis.shape[-1]
    x_rope = x[..., -rd2 * 2:]  # last rope_head_dim elements
    x_rope = x_rope.float().unflatten(-1, (-1, 2))
    x_complex = torch.view_as_complex(x_rope)
    if x_complex.ndim == 3:
        freqs = freqs_cis.view(1, x_complex.size(1), -1)
    else:
        freqs = freqs_cis.view(1, x_complex.size(1), 1, -1)
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    y = x.clone()
    y[..., -rd2 * 2:] = x_rotated.to(y.dtype)
    return y


class IndexerTorchRef(torch.nn.Module):
    """PyTorch reference implementation of Indexer operator.

    Selects top-k compressed KV positions for sparse attention via learned scoring.
    """

    def __init__(self, dim: int, n_heads: int, head_dim: int, rope_head_dim: int,
                 index_topk: int, q_lora_rank: int, compress_ratio: int,
                 kv_cache: torch.Tensor, freqs_cis: torch.Tensor,
                 wq_b_weight: torch.Tensor, weights_proj_weight: torch.Tensor,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.compress_ratio = compress_ratio
        self.softmax_scale = head_dim ** -0.5

        self.wq_b_weight = torch.nn.Parameter(wq_b_weight)
        self.weights_proj_weight = torch.nn.Parameter(weights_proj_weight)
        self.register_buffer('kv_cache', kv_cache)
        self.register_buffer('freqs_cis', freqs_cis)

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        """Forward pass of Indexer.

        Args:
            x: (B, S, dim) bf16 hidden states
            qr: (B, S, q_lora_rank) bf16 low-rank query
            start_pos: int, chunk start position
            offset: int, index offset

        Returns:
            topk_idxs: (B, S, index_topk) int64 top-k indices
        """
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        kv_len = end_pos // ratio

        # Stage A: Linear projections
        q_flat = F.linear(qr, self.wq_b_weight)  # (B, S, H*D)
        q = q_flat.unflatten(-1, (self.n_heads, self.head_dim))  # (B, S, H, D)

        # RoPE on last rope_head_dim elements
        freqs = self.freqs_cis[start_pos:start_pos + seqlen]
        q = apply_rotary_emb(q, freqs)

        weights = F.linear(x, self.weights_proj_weight)  # (B, S, H)
        weights = weights * (self.softmax_scale * self.n_heads ** -0.5)

        # Stage B: Score computation
        # einsum("bshd,btd->bsht", q, kv_cache)
        # q: (B, S, H, D), kv_cache: (B, kv_len, D)
        q_for_einsum = q.permute(0, 2, 1, 3)  # (B, H, S, D)
        kv = self.kv_cache[:bsz, :kv_len, :]  # (B, kv_len, D)
        # (B, H, S, D) @ (B, 1, D, kv_len) -> (B, H, S, kv_len)
        scores = torch.matmul(q_for_einsum, kv.unsqueeze(1).transpose(-1, -2))
        # scores: (B, H, S, kv_len)

        # Stage C: Postprocessing
        # ReLU + weighted sum
        scores_act = torch.relu(scores)  # (B, H, S, kv_len)
        # weights: (B, S, H) -> (B, H, S, 1)
        w = weights.permute(0, 2, 1).unsqueeze(-1)  # (B, H, S, 1)
        index_score = (scores_act * w).sum(dim=1)  # (B, S, kv_len)

        # Causal mask (only for prefill)
        if start_pos == 0:
            row_idx = torch.arange(seqlen, device=x.device).unsqueeze(1)  # (S, 1)
            col_idx = torch.arange(kv_len, device=x.device).unsqueeze(0)  # (1, kv_len)
            threshold = (row_idx + 1) // ratio  # floor((s+1)/ratio)
            causal_mask = col_idx >= threshold  # (S, kv_len)
            index_score = index_score + torch.where(causal_mask, float("-inf"), 0.0)

        # TopK selection
        k = min(self.index_topk, kv_len)
        topk_values, topk_idxs = torch.topk(index_score, k=k, dim=-1)  # (B, S, k)

        # Post-processing
        if start_pos == 0:
            row_idx = torch.arange(seqlen, device=x.device).unsqueeze(1)  # (S, 1)
            threshold = (row_idx + 1) // ratio
            mask = topk_idxs >= threshold
            topk_idxs = torch.where(mask, torch.tensor(-1, dtype=torch.int64, device=x.device),
                                    topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset

        return topk_idxs


def get_indexer_args():
    """Returns default Indexer configuration."""
    return {
        "dim": 1024,
        "n_heads": 16,
        "head_dim": 64,
        "rope_head_dim": 32,
        "index_topk": 128,
        "q_lora_rank": 256,
        "compress_ratio": 4,
        "max_seq_len": 1024,
        "max_batch_size": 4,
    }


def get_init_inputs(device: str = 'npu', dtype: torch.dtype = torch.bfloat16):
    """Create initial weight tensors for Indexer.

    Returns:
        Tuple of (args_dict, wq_b_weight, weights_proj_weight, kv_cache, freqs_cis)
    """
    args = get_indexer_args()
    n_heads = args["n_heads"]
    head_dim = args["head_dim"]
    q_lora_rank = args["q_lora_rank"]
    dim = args["dim"]
    compress_ratio = args["compress_ratio"]
    max_seq_len = args["max_seq_len"]
    max_batch_size = args["max_batch_size"]
    rope_head_dim = args["rope_head_dim"]
    rope_theta = 10000.0

    # Initialize weights (random, like the original)
    wq_b_weight = torch.randn(n_heads * head_dim, q_lora_rank, dtype=dtype, device=device)
    weights_proj_weight = torch.randn(n_heads, dim, dtype=dtype, device=device)

    # Initialize KV cache
    kv_cache = torch.zeros(max_batch_size, max_seq_len // compress_ratio,
                           head_dim, dtype=dtype, device=device)

    # Initialize RoPE frequencies
    freqs = 1.0 / (rope_theta ** (torch.arange(0, rope_head_dim, 2,
                                                device=device)[:rope_head_dim // 2].float() / rope_head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return args, wq_b_weight, weights_proj_weight, kv_cache, freqs_cis


def get_inputs(batch_size: int = 2, seq_len: int = 64, device: str = 'npu',
               dtype: torch.dtype = torch.bfloat16):
    """Create test input tensors.

    Returns:
        List of [x, qr, start_pos, offset]
    """
    args = get_indexer_args()
    dim = args["dim"]
    q_lora_rank = args["q_lora_rank"]

    x = torch.randn(batch_size, seq_len, dim, dtype=dtype, device=device)
    qr = torch.randn(batch_size, seq_len, q_lora_rank, dtype=dtype, device=device)
    start_pos = 0
    offset = 0

    return x, qr, start_pos, offset


if __name__ == "__main__":
    args, wq_b_w, w_proj_w, kv_cache, freqs_cis = get_init_inputs('npu')
    model = IndexerTorchRef(
        dim=args["dim"],
        n_heads=args["n_heads"],
        head_dim=args["head_dim"],
        rope_head_dim=args["rope_head_dim"],
        index_topk=args["index_topk"],
        q_lora_rank=args["q_lora_rank"],
        compress_ratio=args["compress_ratio"],
        kv_cache=kv_cache,
        freqs_cis=freqs_cis,
        wq_b_weight=wq_b_w,
        weights_proj_weight=w_proj_w,
    ).npu()

    x, qr, start_pos, offset = get_inputs(2, 64, 'npu')
    with torch.no_grad():
        result = model(x, qr, start_pos, offset)
    print(f"Forward pass successful! Output shape: {result.shape}, dtype: {result.dtype}")
    print(f"Sample output: {result[0, :5]}")
