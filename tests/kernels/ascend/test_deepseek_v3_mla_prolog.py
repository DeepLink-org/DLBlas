import pytest
import torch
from dlblas.kernels.ascend.apply_rotary_pos_emb import partial_rope_qk_triton
from dlblas.kernels.ascend.deepseek_v3_mla_prolog import (
    fused_matmul_ckvkr_and_rms_norm_cq_triton,
    fused_matmul_and_rms_norm_and_rotary_pos_emb_triton,
    fused_matmul_qn_and_rotary_pos_emb_qr_triton,
)
from dlblas.kernels.ascend.deepseek_v3_mla_prolog_v2 import (
    fused_matmul_qn_and_rotary_pos_emb_qk_triton,
    mla_prolog_triton_v2,
)
from tests.kernels.ascend.common import benchmark_test
from tests.kernels.ascend.test_apply_rotary_pos_emb import apply_rotary_pos_emb_ref
from tests.kernels.ascend.test_rms_norm import rms_norm_ref

device_ = "npu"
dtype_ = torch.float16
dtype_str = "float16"

seq_len = 4096 * 4
num_heads = 128  # N: head number
hidden_size = 7168  # He: hidden_size
qk_nope_head_dim = 128  # D: qk 不含位置编码维度
qk_rope_head_dim = 64  # Dr: qk 位置编码维度
q_head_dim = qk_nope_head_dim + qk_rope_head_dim
q_lora_rank = 1536  # q 低秩矩阵维度
kv_lora_rank = 512  # Hckv: kv 低秩矩阵维度

hidden_states = torch.randn((seq_len, hidden_size), dtype=dtype_, device=device_)
weight_dq = torch.randn((hidden_size, q_lora_rank), dtype=dtype_, device=device_)
weightDkvKr = torch.randn(
    (hidden_size, kv_lora_rank + qk_rope_head_dim), dtype=dtype_, device=device_
)
weight_uq_qr = torch.randn(
    (q_lora_rank, num_heads * q_head_dim), dtype=dtype_, device=device_
)
weight_uk = torch.randn(
    (num_heads, qk_nope_head_dim, kv_lora_rank), dtype=dtype_, device=device_
)
rmsnormGammaCq = torch.randn((q_lora_rank), dtype=dtype_, device=device_)
rmsnormGammaCkv = torch.randn((kv_lora_rank), dtype=dtype_, device=device_)
cos = torch.randn((seq_len, qk_rope_head_dim), dtype=dtype_, device=device_)
sin = torch.randn((seq_len, qk_rope_head_dim), dtype=dtype_, device=device_)

matmul_qc_qr_out_same = torch.randn(
    [seq_len, num_heads * q_head_dim], dtype=dtype_, device=device_
)

kv_same = torch.randn(
    [seq_len, (kv_lora_rank + qk_rope_head_dim)], dtype=dtype_, device=device_
)


def mla_prolog_triton(out_matmul_cq, do_bench=False):
    kv, rmsnorm_out = fused_matmul_ckvkr_and_rms_norm_cq_triton(
        mat_a=hidden_states,
        mat_b=weightDkvKr,
        input=out_matmul_cq,
        weight=rmsnormGammaCq,
        eps=1e-06,
    )
    if do_bench:
        benchmark_test(
            fused_matmul_ckvkr_and_rms_norm_cq_triton,
            fused_matmul_ckvkr_and_rms_norm_cq_triton,
            (hidden_states, weightDkvKr, out_matmul_cq, rmsnormGammaCq, 1e-06),
            "fused_matmul_ckvkr_and_rms_norm_cq_triton",
        )
    # 减少累计误差
    assert kv.shape == kv_same.shape
    matmul_qc_qr_out, kv_cache, kr_cache = (
        fused_matmul_and_rms_norm_and_rotary_pos_emb_triton(
            rmsnorm_out, weight_uq_qr, kv_same, rmsnormGammaCkv, cos, sin
        )
    )
    # 减少累计误差
    assert matmul_qc_qr_out.shape == matmul_qc_qr_out_same.shape
    q, q_rope = fused_matmul_qn_and_rotary_pos_emb_qr_triton(
        matmul_qc_qr_out_same, weight_uk, cos, sin
    )
    return q, q_rope, kv_cache, kr_cache, kv, matmul_qc_qr_out


def test_mla_prolog_part():

    # out_matmul_cq [b, seq, q_lora_rank] <- [b, seq, hidden_size], [hidden_size, q_lora_rank]
    out_matmul_cq = torch.matmul(hidden_states, weight_dq)

    # rmsnorm_cq_out [b, seq, q_lora_rank]
    rmsnorm_cq_out = rms_norm_ref(out_matmul_cq, rmsnormGammaCq, eps=1e-06)

    # matmul_qc_qr_out [b, seq, num_heads * q_head_dim] <- [b, seq, q_lora_rank], [q_lora_rank, num_heads * q_head_dim]
    matmul_qc_qr_out = torch.matmul(rmsnorm_cq_out, weight_uq_qr)
    q = matmul_qc_qr_out.view(seq_len, num_heads, q_head_dim)

    # q_nope: (b, q_len, num_heads, qk_nope_head_dim)
    # q_pe: (b, q_len, num_heads, qk_rope_head_dim)
    q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)

    q_nope = q_nope.view(seq_len, num_heads, qk_nope_head_dim)
    # weight_uk: [num_heads, qk_nope_head_dim, kv_lora_rank]
    matmul_qn_ref = torch.bmm(q_nope.transpose(0, 1), weight_uk)
    q_nope_out_ref = (
        matmul_qn_ref.transpose(0, 1)
        .contiguous()
        .view(seq_len, num_heads, kv_lora_rank)
    )
    # other branch
    matmul_ckv_kr_out = torch.matmul(hidden_states, weightDkvKr)
    # 减少累计误差
    matmul_ckv_kr_out = matmul_ckv_kr_out.view(
        seq_len, 1, kv_lora_rank + qk_rope_head_dim
    )
    k_pe = matmul_ckv_kr_out[..., kv_lora_rank:]
    value_states = matmul_ckv_kr_out[..., :kv_lora_rank]
    kv_cache_ref = rms_norm_ref(value_states, rmsnormGammaCkv, eps=1e-06)
    q_pe_ref = apply_rotary_pos_emb_ref(
        q_pe,
        cos,
        sin,
        unsqueeze_dim=1,
    )
    kr_cache_ref = apply_rotary_pos_emb_ref(
        k_pe,
        cos,
        sin,
        unsqueeze_dim=1,
    )
    q_rope_triton, k_rope_triton = partial_rope_qk_triton(
        q=q,
        k=matmul_ckv_kr_out,
        cos=cos,
        sin=sin,
    )
    torch.testing.assert_close(q_pe_ref, q_rope_triton, rtol=1e-02, atol=1e-02)
    torch.testing.assert_close(kr_cache_ref, k_rope_triton, rtol=1e-02, atol=1e-02)
    matmul_qn_triton, q_rope_triton, k_rope_triton = (
        fused_matmul_qn_and_rotary_pos_emb_qk_triton(
            q, weight_uk, matmul_ckv_kr_out, cos, sin
        )
    )
    torch.testing.assert_close(q_pe_ref, q_rope_triton, rtol=1e-02, atol=1e-02)
    torch.testing.assert_close(kr_cache_ref, k_rope_triton, rtol=1e-02, atol=1e-02)
    torch.testing.assert_close(q_nope_out_ref, matmul_qn_triton, rtol=1e-02, atol=1e-02)

    q_triton, q_pe_triton, kv_cache_triton, kr_cache_triton = mla_prolog_triton_v2(
        out_matmul_cq,
        hidden_states,
        weightDkvKr,
        weight_uq_qr,
        rmsnormGammaCq,
        rmsnormGammaCkv,
        cos,
        sin,
        weight_uk,
    )
    # torch.testing.assert_close(q_nope_out_ref, q_triton, rtol=1e-02, atol=1e-02)
    # torch.testing.assert_close(q_pe_ref, q_pe_triton, rtol=1e-02, atol=1e-02)
    torch.testing.assert_close(kv_cache_ref, kv_cache_triton, rtol=1e-02, atol=1e-02)
    torch.testing.assert_close(kr_cache_ref, kr_cache_triton, rtol=0.08, atol=0.05)


def mla_prolog_ref(out_matmul_cq):

    # out_matmul_cq [b, seq, q_lora_rank] <- [b, seq, hidden_size], [hidden_size, q_lora_rank]

    # rmsnorm_cq_out [b, seq, q_lora_rank]
    rmsnorm_cq_out = rms_norm_ref(out_matmul_cq, rmsnormGammaCq, eps=1e-06)

    # matmul_qc_qr_out [b, seq, num_heads * q_head_dim] <- [b, seq, q_lora_rank], [q_lora_rank, num_heads * q_head_dim]
    matmul_qc_qr_out = torch.matmul(rmsnorm_cq_out, weight_uq_qr)
    # 减少累计误差
    q = matmul_qc_qr_out_same.view(seq_len, num_heads, q_head_dim)

    # q_nope: (b, q_len, num_heads, qk_nope_head_dim)
    # q_pe: (b, q_len, num_heads, qk_rope_head_dim)
    q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)

    q_nope = q_nope.view(seq_len, num_heads, qk_nope_head_dim)
    # weight_uk: [num_heads, qk_nope_head_dim, kv_lora_rank]
    matmul_qn_out = torch.bmm(q_nope.transpose(0, 1), weight_uk)
    q_nope_out = (
        matmul_qn_out.transpose(0, 1)
        .contiguous()
        .view(seq_len, num_heads, kv_lora_rank)
    )
    # other branch
    matmul_ckv_kr_out = torch.matmul(hidden_states, weightDkvKr)
    # 减少累计误差
    kv_same_ = kv_same.view(seq_len, 1, kv_lora_rank + qk_rope_head_dim)
    k_pe = kv_same_[..., kv_lora_rank:]
    value_states = kv_same_[..., :kv_lora_rank]
    kv_cache_ref = rms_norm_ref(value_states, rmsnormGammaCkv, eps=1e-06)
    q_pe_ref = apply_rotary_pos_emb_ref(
        q_pe,
        cos,
        sin,
        unsqueeze_dim=1,
    )
    kr_cache_ref = apply_rotary_pos_emb_ref(
        k_pe,
        cos,
        sin,
        unsqueeze_dim=1,
    )
    return (
        q_nope_out,
        q_pe_ref,
        kv_cache_ref,
        kr_cache_ref,
        matmul_ckv_kr_out,
        matmul_qc_qr_out,
    )


def test_mla_prolog(do_bench=False):
    out_matmul_cq = torch.matmul(hidden_states, weight_dq)
    (
        q_triton,
        q_pe_triton,
        kv_cache_triton,
        kr_cache_triton,
        kv_triton,
        matmul_qc_qr_triton,
    ) = mla_prolog_triton(out_matmul_cq)
    q_ref, q_pe_ref, kv_cache_ref, kr_cache_ref, kv_ref, matmul_qc_qr_ref = (
        mla_prolog_ref(out_matmul_cq)
    )
    torch.testing.assert_close(q_ref, q_triton, rtol=1e-02, atol=1e-02)
    torch.testing.assert_close(q_pe_ref, q_pe_triton, rtol=1e-02, atol=1e-02)
    torch.testing.assert_close(kv_cache_ref, kv_cache_triton, rtol=1e-02, atol=1e-02)
    torch.testing.assert_close(kr_cache_ref, kr_cache_triton, rtol=1e-02, atol=1e-02)

    torch.testing.assert_close(kv_ref, kv_triton, rtol=1e-02, atol=1e-02)
    torch.testing.assert_close(
        matmul_qc_qr_ref, matmul_qc_qr_triton, rtol=0.1, atol=0.1
    )
    if do_bench:
        benchmark_test(
            mla_prolog_ref,
            mla_prolog_triton,
            (out_matmul_cq,),
            "mla_prolog_triton",
        )


if __name__ == "__main__":
    test_mla_prolog_part(do_bench=True)
    # test_mla_prolog(do_bench=True)
