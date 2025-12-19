import torch
from torch import Tensor
from dlblas.kernels.ascend.qwen3_next_atten_prolog import (
    attention_prolog_triton,
    fused_sigmoid_single_norm_rope_matmul_triton,
    fused_single_norm_rope_matmul_triton,
    partial_matmul_triton,
)
from tests.kernels.ascend.common import benchmark_test
from tests.kernels.ascend.test_rms_norm import rms_norm_ref
from tests.kernels.ascend.test_apply_rotary_pos_emb import apply_rotary_pos_emb_ref

device_ = "npu"
dtype_ = torch.float16
seq_len = 4096 * 8
hidden_size = 2048
head_dim = 256
num_q_heads = 16
num_kv_heads = 8
partial_rotary_factor = 0.25
rope_head_dim = int(partial_rotary_factor * head_dim)
hidden_states = torch.randn((seq_len, hidden_size), dtype=dtype_, device=device_)
weight = torch.randn(
    hidden_size,
    ((2 * num_q_heads + 2 * num_kv_heads) * head_dim),
    dtype=dtype_,
    device=device_,
)
rmsnorm_gamma_q = torch.randn((head_dim), dtype=dtype_, device=device_)
rmsnorm_gamma_k = torch.randn((head_dim), dtype=dtype_, device=device_)
cos = torch.randn((seq_len, rope_head_dim), dtype=dtype_, device=device_)
sin = torch.randn((seq_len, rope_head_dim), dtype=dtype_, device=device_)


def attention_prolog_ref(
    hidden_states: Tensor,
    weight: Tensor,
    rmsnorm_gamma_q: Tensor,
    rmsnorm_gamma_k: Tensor,
    cos: Tensor,
    sin: Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    partial_rotary_factor: float,
):
    qkv_states = torch.matmul(hidden_states, weight)
    qkv_states = qkv_states.flatten(0, -2)
    sections = (
        2 * num_q_heads * head_dim,
        num_kv_heads * head_dim,
        num_kv_heads * head_dim,
    )
    q, k, v = qkv_states.split(sections, dim=-1)
    query_states = q.unflatten(-1, (2 * num_q_heads, head_dim)).contiguous()
    key_states = k.unflatten(-1, (num_kv_heads, head_dim)).contiguous()
    value_state = v.unflatten(-1, (num_kv_heads, head_dim)).contiguous()

    query_states, gate = query_states.view(
        *query_states.shape[:-2], -1, 2 * head_dim
    ).chunk(2, dim=-1)
    query_states = query_states.contiguous()
    query_states = rms_norm_ref(query_states, rmsnorm_gamma_q, eps=1e-6)
    key_states = rms_norm_ref(key_states, rmsnorm_gamma_k, eps=1e-6)
    q_rope = query_states[..., :rope_head_dim]
    q_rope_emb = apply_rotary_pos_emb_ref(q_rope, cos, sin, unsqueeze_dim=1)
    query_states[..., :rope_head_dim] = q_rope_emb

    k_rope = key_states[..., :rope_head_dim]
    k_rope_emb = apply_rotary_pos_emb_ref(k_rope, cos, sin, unsqueeze_dim=1)
    key_states[..., :rope_head_dim] = k_rope_emb
    gate = gate.sigmoid()
    return query_states, key_states, value_state, gate


def test_fused_single_norm_and_partial_rope(do_bench=False):
    qkv_states = torch.matmul(hidden_states, weight)
    qkv_states = qkv_states.flatten(0, -2)
    sections = (
        2 * num_q_heads * head_dim,
        num_kv_heads * head_dim,
        num_kv_heads * head_dim,
    )
    qq_ref, key_ref, v_ref = qkv_states.split(sections, dim=-1)
    # v_ref = v_ref.unflatten(-1, (num_kv_heads, head_dim)).contiguous()
    qq_triton = partial_matmul_triton(
        hidden_states, weight, b_n_start=0, n=2 * num_q_heads * head_dim
    )
    torch.testing.assert_close(qq_ref, qq_triton, rtol=1e-02, atol=1e-02)
    if do_bench:
        benchmark_test(
            partial_matmul_triton,
            partial_matmul_triton,
            (hidden_states, weight, 0, 2 * num_q_heads * head_dim),
            "partial_matmul_triton_qq",
        )
    query_states, gate = qq_triton.view(seq_len, num_q_heads, 2 * head_dim).chunk(
        2, dim=-1
    )
    q_rope_ref = rms_norm_ref(query_states, rmsnorm_gamma_q, eps=1e-6)
    q_rope = q_rope_ref[..., :rope_head_dim]
    q_rope_emb = apply_rotary_pos_emb_ref(q_rope, cos, sin, unsqueeze_dim=1)
    q_rope_ref[..., :rope_head_dim] = q_rope_emb
    key_triton, q_rope_triton, gate_triton = (
        fused_sigmoid_single_norm_rope_matmul_triton(
            # partial_matmul
            hidden_states=hidden_states,
            weight=weight,
            b_n_start=2 * num_q_heads * head_dim,
            n=num_kv_heads * head_dim,
            # fused sigmoid, norm and partial rope
            qq=qq_triton.view(seq_len, num_q_heads, head_dim * 2),
            norm_weight=rmsnorm_gamma_q,
            cos=cos,
            sin=sin,
            partial_rotary_factor=partial_rotary_factor,
            eps=1e-6,
        )
    )
    torch.testing.assert_close(q_rope_ref, q_rope_triton, rtol=1e-02, atol=1e-02)
    torch.testing.assert_close(key_ref, key_triton, rtol=1e-02, atol=1e-02)
    torch.testing.assert_close(gate.sigmoid(), gate_triton, rtol=1e-02, atol=1e-02)
    if do_bench:
        benchmark_test(
            fused_sigmoid_single_norm_rope_matmul_triton,
            fused_sigmoid_single_norm_rope_matmul_triton,
            (
                hidden_states,
                weight,
                2 * num_q_heads * head_dim,
                num_kv_heads * head_dim,
                qq_triton.view(seq_len, num_q_heads, head_dim * 2),
                rmsnorm_gamma_q,
                cos,
                sin,
                partial_rotary_factor,
                1e-6,
            ),
            "fused_qa_sigmoid_qb_norm_rope_k_matmul_triton",
        )

    key_states_ref = key_ref.unflatten(-1, (num_kv_heads, head_dim)).contiguous()
    key_states_ref = rms_norm_ref(key_states_ref, rmsnorm_gamma_k, eps=1e-6)
    k_rope = key_states_ref[..., :rope_head_dim]
    k_rope_emb = apply_rotary_pos_emb_ref(k_rope, cos, sin, unsqueeze_dim=1)
    key_states_ref[..., :rope_head_dim] = k_rope_emb
    v_triton, k_rope_triton = fused_single_norm_rope_matmul_triton(
        # partial_matmul
        hidden_states=hidden_states,
        weight=weight,
        b_n_start=2 * num_q_heads * head_dim + num_kv_heads * head_dim,
        n=num_kv_heads * head_dim,
        # fused norm and partial rope
        x=key_triton.view(seq_len, num_kv_heads, head_dim),
        norm_weight=rmsnorm_gamma_k,
        cos=cos,
        sin=sin,
        partial_rotary_factor=partial_rotary_factor,
        eps=1e-6,
        inplace=True,
    )
    torch.testing.assert_close(v_ref, v_triton, rtol=1e-02, atol=1e-02)
    torch.testing.assert_close(key_states_ref, k_rope_triton, rtol=1e-02, atol=1e-02)
    if do_bench:
        benchmark_test(
            fused_single_norm_rope_matmul_triton,
            fused_single_norm_rope_matmul_triton,
            (
                hidden_states,
                weight,
                2 * num_q_heads * head_dim + num_kv_heads * head_dim,
                num_kv_heads * head_dim,
                key_triton.view(seq_len, num_kv_heads, head_dim),
                rmsnorm_gamma_k,
                cos,
                sin,
                partial_rotary_factor,
                1e-6,
                True,
            ),
            "fused_k_norm_rope_v_matmul_triton",
        )


def test_attention_prolog(do_bench=False):
    q_rope_ref, k_rope_ref, v_ref, gates_ref = attention_prolog_ref(
        hidden_states=hidden_states,
        weight=weight,
        rmsnorm_gamma_q=rmsnorm_gamma_q,
        rmsnorm_gamma_k=rmsnorm_gamma_k,
        cos=cos,
        sin=sin,
        eps=1e-6,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        partial_rotary_factor=partial_rotary_factor,
    )
    q_rope_calc, k_rope_calc, v_calc, gates_calc = attention_prolog_triton(
        hidden_states=hidden_states,
        weight=weight,
        rmsnorm_gamma_q=rmsnorm_gamma_q,
        rmsnorm_gamma_k=rmsnorm_gamma_k,
        cos=cos,
        sin=sin,
        eps=1e-6,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        partial_rotary_factor=partial_rotary_factor,
    )
    torch.testing.assert_close(v_ref, v_calc, rtol=1e-02, atol=1e-02)
    torch.testing.assert_close(k_rope_ref, k_rope_calc, rtol=0.05, atol=0.05)
    torch.testing.assert_close(q_rope_ref, q_rope_calc, rtol=0.05, atol=0.05)
    torch.testing.assert_close(gates_ref, gates_calc, rtol=1e-02, atol=1e-02)
    if do_bench:
        benchmark_test(
            attention_prolog_ref,
            attention_prolog_triton,
            (
                hidden_states,
                weight,
                rmsnorm_gamma_q,
                rmsnorm_gamma_k,
                cos,
                sin,
                1e-6,
                num_q_heads,
                num_kv_heads,
                head_dim,
                partial_rotary_factor,
            ),
            "attention_prolog_v2",
        )


weight_q = torch.randn(
    hidden_size, 2 * num_q_heads * head_dim, dtype=dtype_, device=device_
)
weight_k = torch.randn(
    hidden_size, num_kv_heads * head_dim, dtype=dtype_, device=device_
)


def qkv_partial_matmul_ref():
    qkv = torch.matmul(hidden_states, weight)
    return qkv


def partial_matmul_q_triton():
    q = partial_matmul_triton(
        hidden_states, weight, b_n_start=0, n=2 * num_q_heads * head_dim
    )
    return q


def partial_matmul_k_triton():
    k = partial_matmul_triton(
        hidden_states,
        weight,
        b_n_start=2 * num_q_heads * head_dim,
        n=num_kv_heads * head_dim,
    )
    return k


def partial_matmul_v_triton():
    v = partial_matmul_triton(
        hidden_states,
        weight,
        b_n_start=2 * num_q_heads * head_dim + num_kv_heads * head_dim,
        n=num_kv_heads * head_dim,
    )
    return v


def bench_partial_matmul():
    benchmark_test(
        qkv_partial_matmul_ref,
        qkv_partial_matmul_ref,
        (),
        "qkv_partial_matmul_ref",
    )
    benchmark_test(
        partial_matmul_q_triton,
        partial_matmul_q_triton,
        (),
        "partial_matmul_q_triton",
    )
    benchmark_test(
        partial_matmul_k_triton,
        partial_matmul_k_triton,
        (),
        "partial_matmul_k_triton",
    )
    benchmark_test(
        partial_matmul_v_triton,
        partial_matmul_v_triton,
        (),
        "partial_matmul_v_triton",
    )


if __name__ == "__main__":
    test_attention_prolog(do_bench=True)
    test_fused_single_norm_and_partial_rope(do_bench=True)
