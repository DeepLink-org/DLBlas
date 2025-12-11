import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.deeplink as dl

from dlblas.kernels.ascend.fused_rms_norm_rope import _compute_cos_sin_cache
from dlblas.kernels.ascend.fused_rms_norm_rope import rms_norm_rope


DEVICE = "npu"


def _rms_norm_kernel(input: torch.Tensor, weight: torch.Tensor, epsilon: float) -> None:
    # TODO: Remove this contiguous call when the kernel is updated to support non-contiguous input
    # If removed, also need to remove contiguous in MatcherRMSNorm
    input_contiguous = input.contiguous()
    return torch.nn.functional.rms_norm(input_contiguous, weight.shape, weight, epsilon)


def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def _rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_shape = query.shape
    query = query.view(num_tokens, -1, head_size)
    query_rot = query[..., :head_size]
    query_pass = query[..., head_size:]
    query_rot = apply_rotary_emb_torch(query_rot, cos, sin, is_neox)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.view(num_tokens, -1, head_size)
    key_rot = key[..., :head_size]
    key_pass = key[..., head_size:]
    key_rot = apply_rotary_emb_torch(key_rot, cos, sin, is_neox)
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key


def _apply_qk_norm_rope(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
) -> torch.Tensor:
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim

    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q_weight = torch.ones(head_dim, dtype=torch.float32, device="npu")

    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_dim, head_dim)
    q_by_head = _rms_norm_kernel(q_by_head, q_weight, 1e-5)
    q = q_by_head.view(q.shape)

    k_weight = torch.ones(head_dim, dtype=torch.float32, device="npu")
    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // head_dim, head_dim)
    k_by_head = _rms_norm_kernel(k_by_head, k_weight, 1e-5)
    k = k_by_head.view(k.shape)

    cache = _compute_cos_sin_cache(head_dim)
    q, k = _rotary_embedding(positions, q, k, head_dim, cache, True)
    return torch.cat([q, k, v], dim=-1)


def test_rms_norm_rope():
    """test rms norm rope."""
    num_heads, num_kv_heads, head_dim = 4, 4, 16
    num_tokens = 4

    total_dim = (num_heads + 2 * num_kv_heads) * head_dim
    qkv_base = torch.randn(num_tokens, total_dim, dtype=torch.float32, device="npu")
    qkv_base1 = qkv_base.clone()
    positions = torch.arange(num_tokens, dtype=torch.long, device="npu")
    positions1 = positions.clone()

    torch_output = _apply_qk_norm_rope(
        qkv=qkv_base,
        positions=positions,
        num_heads_q=num_heads,
        num_heads_kv=num_kv_heads,
        head_dim=head_dim,
    )

    triton_output = rms_norm_rope(
        qkv=qkv_base1,
        positions=positions1,
        num_heads_q=num_heads,
        num_heads_kv=num_kv_heads,
        head_dim=head_dim,
        num_tokens=num_tokens,
    )
    assert torch.allclose(torch_output, triton_output, atol=1e-2, rtol=0)
    print("test rms_norm_rope passed!")

    # 性能测试部分
    def benchmark_fn(fn, *args):
        return triton.testing.do_bench(lambda: fn(*args), warmup=10, rep=20)

    # Triton 版本性能
    tri_time = benchmark_fn(rms_norm_rope, qkv_base1, positions1, num_heads, num_kv_heads, head_dim, num_tokens)

    # PyTorch 版本性能
    torch_time = benchmark_fn(_apply_qk_norm_rope, qkv_base, positions, num_heads, num_kv_heads, head_dim)

    # 打印性能对比结果
    print(f"\n=== 性能对比 ===")
    print(f"Triton: {tri_time:.4f} ms | PyTorch: {torch_time:.4f} ms | 加速比: {torch_time/tri_time:.2f}x")

if __name__ == "__main__":
    test_rms_norm_rope()
