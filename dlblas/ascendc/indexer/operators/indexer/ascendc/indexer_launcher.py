# Indexer Launcher — Host-side scheduler for the 4-kernel Indexer operator
# Orchestrates the AscendC-equivalent kernel pipeline on Ascend910B2
#
# Execution flow:
#   Kernel 1: q_projection     (Cube/MatMul) — qr @ wq_weight^T
#   Kernel 2: weights_projection (Cube/MatMul) — x @ w_weight^T
#   Kernel 3: rope_score        (Vector+Cube)  — RoPE + Batched MatMul
#   Kernel 4: postprocess_topk  (Vector)       — ReLU+Sum+Mask+TopK

import time
import torch
from typing import Tuple, Optional

from .kernel_q_proj import q_projection
from .kernel_w_proj import weights_projection
from .kernel_rope_score import rope_score_compute
from .kernel_post_topk import postprocess_topk


class IndexerLauncher:
    """Host-side launcher for the Indexer 4-kernel pipeline.

    Manages weight buffers, intermediate tensors, and kernel execution order.
    Follows the architecture defined in DESIGN.md.
    """

    def __init__(
        self,
        dim: int = 1024,
        n_heads: int = 16,
        head_dim: int = 64,
        rope_head_dim: int = 32,
        index_topk: int = 128,
        q_lora_rank: int = 256,
        compress_ratio: int = 4,
        max_seq_len: int = 1024,
        max_batch_size: int = 4,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "npu",
    ):
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.compress_ratio = compress_ratio
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.dtype = dtype
        self.device = device

        self.HD = n_heads * head_dim  # total Q projection output dim

        # Weight tensors (loaded via load_weights)
        self.wq_weight: Optional[torch.Tensor] = None       # (HD, q_lora_rank) bf16
        self.w_proj_weight: Optional[torch.Tensor] = None   # (n_heads, dim) bf16
        self.kv_cache: Optional[torch.Tensor] = None        # (max_batch, max_kv_len, D) bf16
        self.freqs_cis: Optional[torch.Tensor] = None       # (max_seq_len, rd//2) complex

        # Intermediate tensors (reused across calls)
        self._q_flat: Optional[torch.Tensor] = None    # (B*S, HD) bf16
        self._weights: Optional[torch.Tensor] = None   # (B*S, n_heads) bf16
        self._scores: Optional[torch.Tensor] = None    # (B, H, S, kv_len) bf16

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(
        self,
        wq_weight: torch.Tensor,
        w_proj_weight: torch.Tensor,
        kv_cache: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """Load pre-initialized weight tensors onto device.

        Args:
            wq_weight: (n_heads * head_dim, q_lora_rank) bf16
            w_proj_weight: (n_heads, dim) bf16
            kv_cache: (max_batch_size, max_seq_len // compress_ratio, head_dim) bf16
            freqs_cis: (max_seq_len, rope_head_dim//2) complex64
        """
        self.wq_weight = wq_weight.to(device=self.device, dtype=self.dtype)
        self.w_proj_weight = w_proj_weight.to(device=self.device, dtype=self.dtype)
        self.kv_cache = kv_cache.to(device=self.device)
        self.freqs_cis = freqs_cis.to(device=self.device)
        return self

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,       # (B, S, dim) bf16
        qr: torch.Tensor,      # (B, S, q_lora_rank) bf16
        start_pos: int = 0,
        offset: int = 0,
    ) -> torch.Tensor:
        """Execute the full 4-kernel Indexer pipeline.

        Args:
            x: (B, S, dim) hidden states
            qr: (B, S, q_lora_rank) low-rank query
            start_pos: chunk start position
            offset: index offset to add

        Returns:
            topk_idxs: (B, S, K) int64 top-k KV position indices
        """
        assert self.wq_weight is not None, "Weights not loaded. Call load_weights() first."
        assert self.w_proj_weight is not None, "Weights not loaded."
        assert self.kv_cache is not None, "Weights not loaded."
        assert self.freqs_cis is not None, "Weights not loaded."

        B, S, _ = x.shape
        end_pos = start_pos + S
        kv_len = end_pos // self.compress_ratio

        # --- Kernel 1: Q Projection (MatMul) ---
        # qr: (B, S, q_lora_rank) → flatten → (B*S, q_lora_rank)
        # wq_weight: (HD, q_lora_rank)
        # output: (B*S, HD)
        qr_flat = qr.reshape(-1, self.q_lora_rank)
        self._q_flat = q_projection(qr_flat, self.wq_weight)

        # --- Kernel 2: Weights Projection (MatMul) ---
        # x: (B, S, dim) → flatten → (B*S, dim)
        # w_proj_weight: (n_heads, dim)
        # output: (B*S, n_heads)
        x_flat = x.reshape(-1, self.dim)
        self._weights = weights_projection(x_flat, self.w_proj_weight)

        # --- Kernel 3: RoPE + Score Computation ---
        # q_flat: (B, S, HD) reshaped from (B*S, HD)
        # kv_cache: (B, kv_len, D)
        # freqs_cis: (max_seq_len, rd//2) complex
        q_flat_3d = self._q_flat.reshape(B, S, self.HD)
        self._scores = rope_score_compute(
            q_flat_3d, self.kv_cache, self.freqs_cis,
            B=B, S=S, H=self.n_heads, D=self.head_dim,
            kv_len=kv_len, rd=self.rope_head_dim, start_pos=start_pos,
        )

        # --- Kernel 4: Postprocess + TopK ---
        topk_idxs = postprocess_topk(
            self._scores, self._weights,
            start_pos=start_pos, offset=offset,
            index_topk=self.index_topk,
            compress_ratio=self.compress_ratio,
        )

        return topk_idxs

    # ------------------------------------------------------------------
    # Performance measurement
    # ------------------------------------------------------------------

    def benchmark(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int = 0,
        offset: int = 0,
        warmup: int = 10,
        repeat: int = 100,
    ) -> Tuple[float, float]:
        """Measure end-to-end latency.

        Returns:
            (avg_latency_ms, min_latency_ms)
        """
        # Warmup
        for _ in range(warmup):
            self.forward(x, qr, start_pos, offset)
            torch.npu.synchronize()

        # Benchmark
        latencies = []
        for _ in range(repeat):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            self.forward(x, qr, start_pos, offset)
            torch.npu.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

        import statistics
        return statistics.mean(latencies), min(latencies)
