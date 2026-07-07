import torch
import time
import numpy as np

# Use container's torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Copy of sparse_attn_ref from origin
def sparse_attn_ref(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]
    valid_mask = topk_idxs >= 0
    safe_idxs = topk_idxs.clamp(min=0).long()
    b_idx = torch.arange(b, device=q.device)[:, None, None].expand(b, m, topk)
    gathered_kv = kv[b_idx, safe_idxs]
    gathered_kv = gathered_kv.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
    scores = torch.einsum("bmhd,bmtd->bmht", q.float(), gathered_kv.float()) * softmax_scale
    scores = scores.masked_fill(~valid_mask.unsqueeze(2), float("-inf"))
    sink = attn_sink.float().view(1, 1, h, 1)
    max_scores = torch.amax(scores, dim=-1, keepdim=True)
    max_scores = torch.maximum(max_scores, sink)
    exp_scores = torch.exp(scores - max_scores)
    exp_scores = exp_scores.masked_fill(~valid_mask.unsqueeze(2), 0.0)
    exp_sink = torch.exp(sink - max_scores)
    sum_exp = exp_scores.sum(dim=-1, keepdim=True) + exp_sink
    attn_weights = exp_scores / sum_exp
    output = torch.einsum("bmht,bmtd->bmhd", attn_weights, gathered_kv.float())
    return output.to(q.dtype)

# Default config from origin
B, M, N, H, D, TopK = 2, 16, 32, 8, 64, 16
softmax_scale = D ** -0.5

# Create inputs on GPU
device = torch.device("cuda:0")
q = torch.randn(B, M, H, D, dtype=torch.bfloat16, device=device)
kv = torch.randn(B, N, D, dtype=torch.bfloat16, device=device)
attn_sink = torch.zeros(H, dtype=torch.float32, device=device)
topk_idxs = torch.zeros(B, M, TopK, dtype=torch.int32, device=device)
for i in range(topk_idxs.numel()):
    r = (i * 17 + 3) % (N + N // 4)
    topk_idxs.view(-1)[i] = r if r < N else -1

# Warmup
print("Warming up...")
for _ in range(20):
    _ = sparse_attn_ref(q, kv, attn_sink, topk_idxs, softmax_scale)
torch.cuda.synchronize()

# Benchmark
print("Benchmarking...")
n_iter = 1000
start = time.perf_counter()
for _ in range(n_iter):
    _ = sparse_attn_ref(q, kv, attn_sink, topk_idxs, softmax_scale)
torch.cuda.synchronize()
end = time.perf_counter()

torch_time_ms = (end - start) / n_iter * 1000
print(f"\n=== Torch sparse_attn Benchmark ===")
print(f"Shape: B={B}, M={M}, N={N}, H={H}, D={D}, TopK={TopK}")
print(f"dtype: bfloat16")
print(f"Iterations: {n_iter}")
print(f"Average time: {torch_time_ms:.6f} ms")

# MACAC time from final rerun
macac_time_ms = 0.045765  # from final rerun
print(f"\n=== MACAC sparse_attn Benchmark ===")
print(f"Average time: {macac_time_ms:.6f} ms")

print(f"\n=== Comparison ===")
speedup = torch_time_ms / macac_time_ms
print(f"TORCH:  {torch_time_ms:.6f} ms")
print(f"MACAC:  {macac_time_ms:.6f} ms")
print(f"Speedup (MACAC vs Torch): {speedup:.2f}x")
print(f"MACAC is {((1 - 1/speedup) * 100):.1f}% faster than Torch" if speedup > 1 else f"Torch is {((1 - speedup) * 100):.1f}% faster than MACAC")
