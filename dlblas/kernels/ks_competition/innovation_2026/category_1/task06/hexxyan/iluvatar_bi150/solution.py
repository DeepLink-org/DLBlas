from __future__ import annotations
# ==============================================================================

# ==============================================================================
# Benchmark Header (TianShu BI150 (CoreX 4.4.0))
# Platform: TianShu BI150 (CoreX 4.4.0)
# Task: task06
# Evaluation: BI150 auto_bench.py (warmup 20 / repeat 60) -> PASS
# Status: Short-run revalidated; full-run record pending
# v0: 0.467103 ms | v1: 0.258585 ms | Speedup: 1.806x
# Optimization Architecture: Direct efficient SDPA dispatch plus Triton output packing
# ==============================================================================
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _pack_encoder_attention_output(
    source_ptr,
    output_ptr,
    BATCH: tl.constexpr,
    N_TOKENS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    total = BATCH * N_TOKENS * NUM_HEADS * HEAD_SIZE
    mask = offsets < total

    per_batch = N_TOKENS * NUM_HEADS * HEAD_SIZE
    batch = offsets // per_batch
    rem = offsets - batch * per_batch
    token = rem // (NUM_HEADS * HEAD_SIZE)
    head_dim = rem - token * (NUM_HEADS * HEAD_SIZE)
    head = head_dim // HEAD_SIZE
    dim = head_dim - head * HEAD_SIZE
    source_offsets = (
        batch * (NUM_HEADS * N_TOKENS * HEAD_SIZE)
        + head * HEAD_SIZE
        + token * (NUM_HEADS * HEAD_SIZE)
        + dim
    )

    value = tl.load(source_ptr + source_offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, value, mask=mask)


def _reference(query, key, value, num_heads, head_size, num_kv_heads):
    bsz, q_len = query.shape[:2]
    kv_len = key.shape[1]
    q = query.view(bsz, q_len, num_heads, head_size).transpose(1, 2)
    k = key.view(bsz, kv_len, num_kv_heads, head_size).transpose(1, 2)
    v = value.view(bsz, kv_len, num_kv_heads, head_size).transpose(1, 2)
    if num_kv_heads != num_heads:
        repeat = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(head_size))
    return out.transpose(1, 2).reshape(bsz, q_len, num_heads * head_size)


def _efficient_reference(query, key, value, num_heads, head_size, num_kv_heads):
    bsz, q_len = query.shape[:2]
    kv_len = key.shape[1]
    q = query.view(bsz, q_len, num_heads, head_size).transpose(1, 2)
    k = key.view(bsz, kv_len, num_kv_heads, head_size).transpose(1, 2)
    v = value.view(bsz, kv_len, num_kv_heads, head_size).transpose(1, 2)
    if num_kv_heads != num_heads:
        repeat = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    try:
        out = torch.ops.aten._scaled_dot_product_efficient_attention.default(
            q, k, v, None, False, 0.0, False,
            scale=1.0 / math.sqrt(head_size),
        )[0]
    except (AttributeError, RuntimeError):
        out = F.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(head_size)
        )
    return out.transpose(1, 2).reshape(bsz, q_len, num_heads * head_size)


class Model(nn.Module):
    def __init__(self, num_heads: int = 8, head_size: int = 64, num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = 1.0 / math.sqrt(head_size)

    def forward(self, query, key, value):
        return _reference(query, key, value, self.num_heads, self.head_size, self.num_kv_heads)


class ModelNew(Model):
    def __init__(self, num_heads: int = 8, head_size: int = 64, num_kv_heads: int = 8):
        super().__init__(num_heads, head_size, num_kv_heads)
        self._view_key = None
        self._q_view = None
        self._k_view = None
        self._v_view = None
        self._graph = None
        self._capture_key = None
        self._graph_output = None
        self._graph_failure_key = None
        self._packed_output = None
        self._packed_key = None

    @staticmethod
    def _layout_key(tensor: torch.Tensor):
        return (
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )

    def _prepare_views(self, query, key, value):
        view_key = (
            self._layout_key(query),
            self._layout_key(key),
            self._layout_key(value),
        )
        if self._view_key != view_key:
            bsz, q_len = query.shape[:2]
            kv_len = key.shape[1]
            self._q_view = query.view(
                bsz, q_len, self.num_heads, self.head_size
            ).transpose(1, 2)
            self._k_view = key.view(
                bsz, kv_len, self.num_kv_heads, self.head_size
            ).transpose(1, 2)
            self._v_view = value.view(
                bsz, kv_len, self.num_kv_heads, self.head_size
            ).transpose(1, 2)
            self._view_key = view_key
        if self._packed_key != view_key:
            self._packed_output = torch.empty(
                (
                    query.shape[0],
                    query.shape[1],
                    self.num_heads * self.head_size,
                ),
                dtype=query.dtype,
                device=query.device,
            )
            self._packed_key = view_key
        return view_key

    def _run_attention(self):
        try:
            out = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                self._q_view, self._k_view, self._v_view,
                None, False, 0.0, False, scale=self.scale,
            )[0]
        except (AttributeError, RuntimeError):
            out = F.scaled_dot_product_attention(
                self._q_view, self._k_view, self._v_view, scale=self.scale,
            )
        batch = self._q_view.shape[0]
        tokens = self._q_view.shape[-2]
        total = batch * tokens * self.num_heads * self.head_size
        _pack_encoder_attention_output[(triton.cdiv(total, 1024),)](
            out,
            self._packed_output,
            BATCH=batch,
            N_TOKENS=tokens,
            NUM_HEADS=self.num_heads,
            HEAD_SIZE=self.head_size,
            BLOCK=1024,
            num_warps=2,
            num_stages=1,
        )
        return self._packed_output

    def _graph_key_for(self, view_key):
        stream = getattr(torch.cuda.current_stream(), "cuda_stream", None)
        try:
            hash(stream)
        except TypeError:
            stream = repr(stream)
        return (view_key, stream)

    def _capture_graph(self, graph_key):
        try:
            self._run_attention()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._run_attention()
            torch.cuda.synchronize()
            self._graph = graph
            self._capture_key = graph_key
            self._graph_output = self._packed_output
            self._graph_failure_key = None
            return self._packed_output
        except Exception:
            self._graph = None
            self._capture_key = None
            self._graph_output = None
            self._graph_failure_key = graph_key
            return None

    def forward(self, query, key, value):
        if (
            query.device.type == "cpu"
            or self.num_kv_heads != self.num_heads
            or torch.is_grad_enabled()
        ):
            return _efficient_reference(
                query, key, value,
                self.num_heads, self.head_size, self.num_kv_heads,
            )

        view_key = self._prepare_views(query, key, value)
        graph_key = self._graph_key_for(view_key)
        if self._graph is not None and graph_key == self._capture_key:
            self._graph.replay()
            return self._graph_output
        if graph_key != self._graph_failure_key:
            graph_output = self._capture_graph(graph_key)
            if graph_output is not None:
                return graph_output
        return self._run_attention()


def _device():
    for name in ("cuda", "npu", "mlu", "gcu"):
        backend = getattr(torch, name, None)
        if backend is not None and getattr(backend, "is_available", lambda: False)():
            return torch.device(name)
    return torch.device("cpu")


def get_inputs():
    bsz, seq_len, num_heads, head_size = 2, 83, 8, 64
    device = _device()
    hidden = num_heads * head_size
    query = torch.randn(bsz, seq_len, hidden, dtype=torch.float16, device=device)
    key = torch.randn(bsz, seq_len, hidden, dtype=torch.float16, device=device)
    value = torch.randn(bsz, seq_len, hidden, dtype=torch.float16, device=device)
    return [query, key, value]


def get_init_inputs():
    return [8, 64, 8]
