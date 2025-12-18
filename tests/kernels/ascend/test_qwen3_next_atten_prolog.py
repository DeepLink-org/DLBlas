import torch
from torch import Tensor
from dlblas.kernels.ascend.qwen3_next_atten_prolog_v1 import (
    attention_prolog_triton_v1,
    fused_matmul_norm_rotary_emb_triton,
    fused_matmul_norm_sigmoid_triton,
    partial_matmul_triton,
)
from dlblas.kernels.ascend.qwen3_next_atten_prolog_v2 import (
    attention_prolog_triton_v2,
    fused_single_norm_rope_matmul_triton,
)
from dlblas.kernels.ascend.rms_norm import rms_norm_block_triton
from tests.kernels.ascend.common import benchmark_test
from tests.kernels.ascend.test_rms_norm import rms_norm_ref
from tests.kernels.ascend.test_apply_rotary_pos_emb import apply_rotary_pos_emb_ref

device_ = "npu"
dtype_ = torch.float16

seq_len = 4096 * 4
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


def test_attention_prolog_split(do_bench=False):
    qkv_states = torch.matmul(hidden_states, weight)
    qkv_states = qkv_states.flatten(0, -2)
    sections = (
        2 * num_q_heads * head_dim,
        num_kv_heads * head_dim,
        num_kv_heads * head_dim,
    )
    q_ref, k_ref, v_ref = qkv_states.split(sections, dim=-1)
    q_calc = partial_matmul_triton(
        hidden_states, weight, b_n_start=0, n=2 * num_q_heads * head_dim
    )
    torch.testing.assert_close(q_ref, q_calc, rtol=1e-02, atol=1e-02)
    qq_ref = q_ref.unflatten(-1, (num_q_heads, 2 * head_dim)).contiguous()
    q_ref, gate_ref = qq_ref.chunk(2, dim=-1)
    q_ref = q_ref.view(-1, num_q_heads, head_dim).contiguous()
    key_ref = k_ref.unflatten(-1, (num_kv_heads, head_dim)).contiguous()
    value_ref = v_ref.unflatten(-1, (num_kv_heads, head_dim)).contiguous()
    q_norm_ref = rms_norm_ref(q_ref, rmsnorm_gamma_q, eps=1e-6)
    k_norm_ref = rms_norm_ref(key_ref, rmsnorm_gamma_k, eps=1e-6)

    q_rope = q_norm_ref[..., :rope_head_dim]
    q_rope_emb = apply_rotary_pos_emb_ref(q_rope, cos, sin, unsqueeze_dim=1)
    q_norm_ref[..., :rope_head_dim] = q_rope_emb
    q_rope_ref = q_norm_ref

    k_rope = k_norm_ref[..., :rope_head_dim]
    k_rope_emb = apply_rotary_pos_emb_ref(k_rope, cos, sin, unsqueeze_dim=1)
    k_norm_ref[..., :rope_head_dim] = k_rope_emb
    k_rope_ref = k_norm_ref

    q_sigmoid_ref = gate_ref.sigmoid()
    k_calc, q_norm_calc, q_sigmoid_calc = fused_matmul_norm_sigmoid_triton(
        hidden_states,
        weight,
        b_n_start=2 * num_q_heads * head_dim,
        n=num_kv_heads * head_dim,
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        qq=qq_ref,
        rmsnorm_gamma_q=rmsnorm_gamma_q,
        eps=1e-6,
    )
    torch.testing.assert_close(k_ref, k_calc, rtol=1e-02, atol=1e-02)
    torch.testing.assert_close(q_norm_ref, q_norm_calc, rtol=1e-02, atol=1e-02)
    torch.testing.assert_close(q_sigmoid_ref, q_sigmoid_calc, rtol=1e-02, atol=1e-02)

    v_calc, q_rope_calc, k_rope_calc = fused_matmul_norm_rotary_emb_triton(
        # v_partial_matmul
        hidden_states=hidden_states,
        weight=weight,
        b_n_start=2 * num_q_heads * head_dim + num_kv_heads * head_dim,
        n=num_kv_heads * head_dim,
        # k_rmsnorm
        k_input=k_ref.contiguous().view(seq_len, num_kv_heads, head_dim),
        rmsnorm_gamma_k=rmsnorm_gamma_k,
        eps=1e-6,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        # q_k_rotary_pos
        q_input=q_norm_ref.contiguous().view(seq_len, num_q_heads, head_dim),
        cos=cos,
        sin=sin,
        partial_rotary_factor=partial_rotary_factor,
        inplace=False,
    )
    torch.testing.assert_close(value_ref, v_calc, rtol=1e-02, atol=1e-02)
    torch.testing.assert_close(q_rope_ref, q_rope_calc, rtol=1e-02, atol=1e-02)
    torch.testing.assert_close(k_rope_ref, k_rope_calc, rtol=0.05, atol=0.05)
    if do_bench:
        benchmark_test(
            fused_matmul_norm_sigmoid_triton,
            fused_matmul_norm_sigmoid_triton,
            (
                hidden_states,
                weight,
                2 * num_q_heads * head_dim,
                num_kv_heads * head_dim,
                num_q_heads,
                head_dim,
                qq_ref,
                rmsnorm_gamma_q,
                1e-6,
            ),
            "fused_matmul_norm_sigmoid_triton",
        )


def test_fused_single_norm_and_partial_rope(do_bench=False):
    qkv_states = torch.matmul(hidden_states, weight)
    qkv_states = qkv_states.flatten(0, -2)
    sections = (
        2 * num_q_heads * head_dim,
        num_kv_heads * head_dim,
        num_kv_heads * head_dim,
    )
    q, key_states, v = qkv_states.split(sections, dim=-1)
    q = torch.randn((seq_len, num_q_heads, head_dim), dtype=dtype_, device=device_)
    q_test = q.clone()

    key_triton, q_out_triton = fused_single_norm_rope_matmul_triton(
        hidden_states,
        weight,
        2 * num_q_heads * head_dim,
        num_kv_heads * head_dim,
        q,
        rmsnorm_gamma_q,
        cos,
        sin,
        partial_rotary_factor,
        inplace=True,
    )
    q_norm = rms_norm_block_triton(q_test, rmsnorm_gamma_q, eps=1e-06)
    q_norm_rope = q_norm[..., :rope_head_dim]
    q_ref_rope_emb = apply_rotary_pos_emb_ref(q_norm_rope, cos, sin, unsqueeze_dim=1)
    q_norm[..., :rope_head_dim] = q_ref_rope_emb

    torch.testing.assert_close(key_states, key_triton, rtol=1e-02, atol=1e-02)
    torch.testing.assert_close(q_norm, q_out_triton, rtol=0.02, atol=0.02)

    if do_bench:
        benchmark_test(
            fused_single_norm_rope_matmul_triton,
            fused_single_norm_rope_matmul_triton,
            (
                hidden_states,
                weight,
                2 * num_q_heads * head_dim,
                num_kv_heads * head_dim,
                q,
                rmsnorm_gamma_q,
                cos,
                sin,
                partial_rotary_factor,
            ),
            "fused_single_norm_rope_q_matmul_k_triton",
        )


def test_attention_prolog_v1(do_bench=False):
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
    q_rope_calc, k_rope_calc, v_calc, gates_calc = attention_prolog_triton_v1(
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
    torch.testing.assert_close(q_rope_ref, q_rope_calc, rtol=0.05, atol=0.05)
    torch.testing.assert_close(k_rope_ref, k_rope_calc, rtol=0.05, atol=0.05)
    torch.testing.assert_close(gates_ref, gates_calc, rtol=1e-02, atol=1e-02)
    if do_bench:
        benchmark_test(
            attention_prolog_ref,
            attention_prolog_triton_v1,
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
                1.0,
            ),
            "attention_prolog",
        )


def test_attention_prolog_v2(do_bench=False):
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
    q_rope_calc, k_rope_calc, v_calc, gates_calc = attention_prolog_triton_v2(
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
            attention_prolog_triton_v2,
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
    test_attention_prolog_split()
    # test_attention_prolog_v2(do_bench=True)
    # test_fused_single_norm_and_partial_rope(do_bench=True)
