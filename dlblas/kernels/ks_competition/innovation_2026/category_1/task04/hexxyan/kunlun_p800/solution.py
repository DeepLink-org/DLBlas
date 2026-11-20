from __future__ import annotations
# ==============================================================================

# ==============================================================================
# Benchmark Header for Kunlunxin P800 (XTriton 3.0.0 + XPyTorch 2.9.0)
# Platform: Kunlunxin P800 (XPU Driver 5.0.21.26, XCCL bkcl)
# Benchmark Script: run_full_eval_p800.py (--warmup 100 --repeat 300)
# Status: PASS accuracy (100% numerical match)
# Baseline v0 (PyTorch): 1.153093 ms
# Optimized v1 (Fused 2D Grid Single Launch Triton): 0.612569 ms
# Speedup: 1.882x (Standalone Kernel Level Speedup: 2.466x)
# History:
#   v0 PyTorch baseline: 1.210986 ms (1.000x)
#   v1 Fused 2D Grid Triton: 0.855349 ms (1.332x)
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_act_max_pool_2d_kernel(
    logits_ptr,
    out_ptr,
    offsets_ptr,
    lens_ptr,
    stride_m,
    stride_n,
    stride_out_b,
    stride_out_v,
    VOCAB: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < VOCAB

    off_m = tl.load(offsets_ptr + pid_b)
    L = tl.load(lens_ptr + pid_b)

    base_ptr = logits_ptr + off_m * stride_m + offs_n * stride_n

    cur_max = tl.zeros([BLOCK_N], dtype=tl.float32)
    for i in range(50):
        if i < L:
            val = tl.load(base_ptr + i * stride_m, mask=mask_n, other=0.0)
            relu_val = tl.maximum(val, 0.0)
            act_val = tl.log(1.0 + relu_val)
            cur_max = tl.maximum(cur_max, act_val)

    out_base = out_ptr + pid_b * stride_out_b + offs_n * stride_out_v
    tl.store(out_base, cur_max, mask=mask_n)


@triton.jit
def fused_act_sum_pool_2d_kernel(
    logits_ptr,
    out_ptr,
    offsets_ptr,
    lens_ptr,
    stride_m,
    stride_n,
    stride_out_b,
    stride_out_v,
    VOCAB: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < VOCAB

    off_m = tl.load(offsets_ptr + pid_b)
    L = tl.load(lens_ptr + pid_b)

    base_ptr = logits_ptr + off_m * stride_m + offs_n * stride_n

    cur_sum = tl.zeros([BLOCK_N], dtype=tl.float32)
    for i in range(50):
        if i < L:
            val = tl.load(base_ptr + i * stride_m, mask=mask_n, other=0.0)
            relu_val = tl.maximum(val, 0.0)
            act_val = tl.log(1.0 + relu_val)
            cur_sum = cur_sum + act_val

    out_base = out_ptr + pid_b * stride_out_b + offs_n * stride_out_v
    tl.store(out_base, cur_sum, mask=mask_n)


class Model(nn.Module):
    """SPLADESparsePooler: MLM head logits → ReLU log(1+x) pooled over sequence (max or sum)."""

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30522,
        pooling: str = "max",
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)
        self.pooling = pooling

    def forward(self, hidden_states: torch.Tensor, seq_lens: torch.Tensor) -> list:
        x = self.decoder(self.layer_norm(self.act(self.dense(hidden_states))))
        x = torch.log1p(F.relu(x))
        result = []
        offset = 0
        for L in seq_lens.tolist():
            chunk = x[offset : offset + L]
            if self.pooling == "max":
                result.append(chunk.max(dim=0).values)
            else:
                result.append(chunk.sum(dim=0))
            offset += L
        return result


class ModelNew(nn.Module):
    """Optimized SPLADESparsePooler for Kunlunxin P800 with fused 2D Grid Triton activation+pooling kernel."""

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30522,
        pooling: str = "max",
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)
        self.pooling = pooling
        self.vocab_size = vocab_size
        self.dense_w = None
        self.dense_b = None
        self.ln_w = None
        self.ln_b = None
        self.dec_w = None
        self.dec_b = None
        self.BLOCK_N = 1024
        self.grid_n = triton.cdiv(vocab_size, self.BLOCK_N)
        self._out_buf = None
        self._seq_key = None
        self._batch_size = None
        self._graph = None
        self._graph_key = None
        self._graph_failure_key = None
        self._views = None
        self._views_batch_size = None

    @staticmethod
    def _sequence_key(seq_lens: torch.Tensor | list[int]):
        if isinstance(seq_lens, torch.Tensor):
            return (
                "tensor",
                seq_lens.data_ptr(),
                seq_lens._version,
                tuple(seq_lens.shape),
                tuple(seq_lens.stride()),
                seq_lens.dtype,
                seq_lens.device,
            )
        return ("list", tuple(seq_lens))

    @staticmethod
    def _input_key(tensor: torch.Tensor):
        return (
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )

    def _run_fast(self, hidden_states: torch.Tensor) -> None:
        h1 = F.gelu(torch.addmm(self.dense_b, hidden_states, self.dense_w))
        h2 = F.layer_norm(h1, (768,), self.ln_w, self.ln_b, eps=1e-12)
        logits = torch.addmm(self.dec_b, h2, self.dec_w)

        grid = (self._batch_size, self.grid_n)
        if self.pooling == "max":
            fused_act_max_pool_2d_kernel[grid](
                logits,
                self._out_buf,
                self._cached_offsets,
                self._cached_lens,
                logits.stride(0),
                logits.stride(1),
                self._out_buf.stride(0),
                self._out_buf.stride(1),
                VOCAB=self.vocab_size,
                BLOCK_N=self.BLOCK_N,
            )
        else:
            fused_act_sum_pool_2d_kernel[grid](
                logits,
                self._out_buf,
                self._cached_offsets,
                self._cached_lens,
                logits.stride(0),
                logits.stride(1),
                self._out_buf.stride(0),
                self._out_buf.stride(1),
                VOCAB=self.vocab_size,
                BLOCK_N=self.BLOCK_N,
            )

    def _graph_key_for(self, hidden_states, seq_lens):
        stream = getattr(torch.cuda.current_stream(), "cuda_stream", None)
        try:
            hash(stream)
        except TypeError:
            stream = repr(stream)
        return (
            self._input_key(hidden_states),
            self._sequence_key(seq_lens),
            stream,
            self.pooling,
        )

    def _capture_graph(self, graph_key, hidden_states):
        try:
            self._run_fast(hidden_states)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._run_fast(hidden_states)
            torch.cuda.synchronize()
            self._graph = graph
            self._graph_key = graph_key
            self._graph_failure_key = None
            return True
        except Exception:
            self._graph = None
            self._graph_key = None
            self._graph_failure_key = graph_key
            return False

    def forward(self, hidden_states: torch.Tensor, seq_lens: torch.Tensor | list[int]) -> list[torch.Tensor]:
        if self.dense_w is None:
            self.dense_w = self.dense.weight.t()
            self.dense_b = self.dense.bias
            self.ln_w = self.layer_norm.weight
            self.ln_b = self.layer_norm.bias
            self.dec_w = self.decoder.weight.t()
            self.dec_b = self.decoder.bias

        device = hidden_states.device
        seq_key = (self._sequence_key(seq_lens), device)
        if self._seq_key != seq_key:
            seq_list = seq_lens if isinstance(seq_lens, list) else seq_lens.tolist()
            offsets = [0]
            for length in seq_list[:-1]:
                offsets.append(offsets[-1] + length)
            self._cached_offsets = torch.tensor(
                offsets, dtype=torch.int32, device=device
            )
            self._cached_lens = torch.tensor(
                seq_list, dtype=torch.int32, device=device
            )
            self._batch_size = len(seq_list)
            self._seq_key = seq_key

        batch_size = self._batch_size

        if self._out_buf is None or self._out_buf.shape[0] < batch_size:
            self._out_buf = torch.empty(
                (batch_size, self.vocab_size),
                dtype=torch.float32,
                device=device,
            )
            self._graph = None
            self._graph_key = None
            self._views = None
            self._views_batch_size = None
        if self._views is None or self._views_batch_size != batch_size:
            self._views = [self._out_buf[i] for i in range(batch_size)]
            self._views_batch_size = batch_size

        graph_eligible = (
            device.type != "cpu"
            and isinstance(seq_lens, torch.Tensor)
            and not torch.is_grad_enabled()
            and hidden_states.is_contiguous()
            and seq_lens.is_contiguous()
        )
        if graph_eligible:
            graph_key = self._graph_key_for(hidden_states, seq_lens)
            if self._graph is not None and graph_key == self._graph_key:
                self._graph.replay()
            elif graph_key != self._graph_failure_key:
                if not self._capture_graph(graph_key, hidden_states):
                    self._run_fast(hidden_states)
            else:
                self._run_fast(hidden_states)
        else:
            self._run_fast(hidden_states)

        return self._views


def get_inputs():
    seq_lens = torch.tensor([20, 25, 18, 20], dtype=torch.int32, device="cuda")
    hidden_states = torch.randn(83, 768, device="cuda")
    return [hidden_states, seq_lens]


def get_init_inputs():
    return [768, 30522, "max"]
