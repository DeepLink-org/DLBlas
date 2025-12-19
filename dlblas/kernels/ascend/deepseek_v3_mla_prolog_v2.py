import torch
import triton
import triton.language as tl
import triton.language.extra.deeplink as dl
from torch import Tensor
from dlblas.kernels.ascend.apply_rotary_pos_emb import partial_rope_qk_kernel
from dlblas.utils.op_helper import grouped_launch_diagonal
from dlblas.kernels.ascend.rms_norm import rms_norm_block_kernel
from dlblas.utils.device_utils import NUM_CORES


@triton.jit
def matmul_kernel(
    mat_a,
    mat_b,
    mat_c,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
        task_m_idx, task_n_idx = grouped_launch_diagonal(
            block_idx, NUM_BLOCKS_M, NUM_BLOCKS_N, BLOCK_TRESHHOLD
        )
        m_start = task_m_idx * BLOCK_M
        n_start = task_n_idx * BLOCK_N
        mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (
                k_start + tl.arange(0, BLOCK_K)
            )[None, :]
            mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                (k_start + tl.arange(0, BLOCK_K)) < K
            )[None, :]
            mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)
            dl.compile_hint(mat_a_block, "dot_pad_only_k")
            mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + (
                n_start + tl.arange(0, BLOCK_N)
            )[None, :]
            mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & (
                (n_start + tl.arange(0, BLOCK_N)) < N
            )[None, :]
            mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
            dl.compile_hint(mat_b_block, "dot_pad_only_k")
            mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
        mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (
            n_start + tl.arange(0, BLOCK_N)
        )[None, :]
        mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
            (n_start + tl.arange(0, BLOCK_N)) < N
        )[None, :]
        tl.store(
            mat_c + mat_c_offset,
            mat_c_block.to(mat_c.dtype.element_ty),
            mask=mat_c_mask,
        )


@triton.jit
def fused_matmul_ckvkr_and_rms_norm_cq_kernel(
    # matmul params
    mat_a,
    mat_b,
    mat_c,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
    # rms norm params
    input,
    weight,
    output,
    n_rows,
    input_row_stride: tl.constexpr,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_NORM: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    # matmul_kv + rmsnorm_q
    with dl.async_task(scope=dl.async_task.cube):
        matmul_kernel(
            mat_a=mat_a,
            mat_b=mat_b,
            mat_c=mat_c,
            M=M,
            N=N,
            K=K,
            NUM_CORES=NUM_CORES,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            BLOCK_TRESHHOLD=BLOCK_TRESHHOLD,
        )
    with dl.async_task(scope=dl.async_task.vector):
        rms_norm_block_kernel(
            input=input,
            weight=weight,
            output=output,
            n_rows=n_rows,
            input_row_stride=input_row_stride,
            eps=eps,
            N_COLS=N_COLS,
            BLOCK=BLOCK_NORM,
            NUM_CORES=NUM_CORES,
        )


def fused_matmul_ckvkr_and_rms_norm_cq_triton(
    mat_a: Tensor, mat_b: Tensor, input: Tensor, weight: Tensor, eps=1e-6
):
    assert mat_a.is_contiguous()
    if len(mat_a.shape) == 3:
        b, seq_len = mat_a.size(0), mat_a.size(1)
        mat_a = mat_a.view(-1, mat_a.shape[-1])
    elif len(mat_a.shape) == 2:
        b, seq_len = 1, mat_a.size(0)
    assert mat_a.size(-1) == mat_b.size(0)
    assert len(mat_a.shape) == 2 and len(mat_b.shape) == 2
    m = mat_a.shape[0]
    k = mat_a.shape[1]
    n = mat_b.shape[1]
    mat_c = torch.empty(b * seq_len, n, dtype=mat_a.dtype, device=mat_a.device)

    assert input.is_contiguous()
    feat_size = weight.shape[0]
    assert b * seq_len == input.numel() // input.size(-1)
    assert feat_size == input.size(-1)
    input_stride = input.stride(-2)
    rms_norm_out = torch.empty_like(input)

    """
    NPU芯片更加亲和512B对齐场景,如下分块通用性能较好,可以使用autotune选取最优
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256    
    """
    fused_matmul_ckvkr_and_rms_norm_cq_kernel[(NUM_CORES,)](
        mat_a=mat_a,
        mat_b=mat_b,
        mat_c=mat_c,
        M=m,
        N=n,
        K=k,
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=256,
        BLOCK_TRESHHOLD=8,
        input=input,
        weight=weight,
        output=rms_norm_out,
        n_rows=b * seq_len,
        input_row_stride=input_stride,
        eps=eps,
        N_COLS=feat_size,
        BLOCK_NORM=4,
        NUM_CORES=NUM_CORES,
    )
    return mat_c, rms_norm_out


@triton.jit
def matmul_qn_kernel(
    mat_a,
    mat_b,
    mat_c,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
    Q_HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
):
    # q = matmul_qc_qr_out.view(b * seq_len, num_heads, q_head_dim)
    # q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    # q_nope = q_nope.view(b * seq_len, num_heads, qk_nope_head_dim)
    # matmul_qn_out = torch.bmm(q_nope.transpose(0, 1), weight_uk)
    # 左矩阵q的nope部分，右矩阵weight_uk: [num_heads, qk_nope_head_dim, kv_lora_rank]，实现bmm逻辑
    # mat_c: [b * seq_len, num_heads, kv_lora_rank]
    # K == qk_nope_head_dim

    pid = tl.program_id(axis=0)
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    for head_idx in range(NUM_HEADS):  # batch matmul
        for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
            task_m_idx, task_n_idx = grouped_launch_diagonal(
                block_idx, NUM_BLOCKS_M, NUM_BLOCKS_N, BLOCK_TRESHHOLD
            )
            m_start = task_m_idx * BLOCK_M
            n_start = task_n_idx * BLOCK_N
            # begin k
            mat_a_offset = (
                ((m_start + tl.arange(0, BLOCK_M)) * (NUM_HEADS * Q_HEAD_DIM))[:, None]
                + (head_idx * Q_HEAD_DIM)
                + tl.arange(0, BLOCK_K)[None, :]
            )
            mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                (tl.arange(0, BLOCK_K)) < K
            )[None, :]
            mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)
            dl.compile_hint(mat_a_block, "dot_pad_only_k")
            mat_b_offset = (
                (head_idx * K * N)
                + (tl.arange(0, BLOCK_K) * N)[:, None]
                + (n_start + tl.arange(0, BLOCK_N))[None, :]
            )
            mat_b_mask = (tl.arange(0, BLOCK_K) < K)[:, None] & (
                (n_start + tl.arange(0, BLOCK_N)) < N
            )[None, :]
            mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
            dl.compile_hint(mat_b_block, "dot_pad_only_k")
            mat_c_block = tl.dot(mat_a_block, mat_b_block)
            # end k
            mat_c_offset = (
                ((m_start + tl.arange(0, BLOCK_M)) * NUM_HEADS * N)[:, None]
                + (head_idx * N)
                + (n_start + tl.arange(0, BLOCK_N))[None, :]
            )
            mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                (n_start + tl.arange(0, BLOCK_N)) < N
            )[None, :]
            tl.store(
                mat_c + mat_c_offset,
                mat_c_block.to(mat_c.dtype.element_ty),
                mask=mat_c_mask,
            )


def matmul_qn_triton(mat_a: Tensor, weight_uk: Tensor):
    assert mat_a.is_contiguous()
    b, seq_len, num_heads, q_head_dim = mat_a.shape
    num_heads, qk_nope_head_dim, kv_lora_rank = weight_uk.shape
    mat_a = mat_a.view(b, seq_len, num_heads, q_head_dim)
    mat_c = torch.empty(
        (b, seq_len, num_heads, kv_lora_rank), dtype=mat_a.dtype, device=mat_a.device
    )
    matmul_qn_kernel[(NUM_CORES,)](
        mat_a=mat_a,
        mat_b=weight_uk,
        mat_c=mat_c,
        M=b * seq_len,
        N=kv_lora_rank,
        K=qk_nope_head_dim,
        NUM_CORES=NUM_CORES,
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=qk_nope_head_dim,
        BLOCK_TRESHHOLD=6,
        Q_HEAD_DIM=q_head_dim,
        NUM_HEADS=num_heads,
    )
    return mat_c


@triton.jit
def fused_matmul_qn_and_rotary_pos_emb_qk_kernel(
    # matmul
    Q,
    MAT_B,
    MAT_C,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
    NUM_HEADS_Q: tl.constexpr,
    # rotary pos emb
    Q_INPUT,
    K_INPUT,
    COS,
    SIN,
    Q_EMBED,
    K_EMBED,
    total_seq_len,
    NOPE_DIM_Q: tl.constexpr,
    NOPE_DIM_K: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    NUM_HEADS_K: tl.constexpr,
):
    with dl.async_task(scope=dl.async_task.vector):
        # rope_qk
        partial_rope_qk_kernel(
            Q=Q_INPUT,
            K=K_INPUT,
            COS=COS,
            SIN=SIN,
            Q_EMBED=Q_EMBED,
            K_EMBED=K_EMBED,
            total_seq_len=total_seq_len,
            NOPE_DIM_Q=NOPE_DIM_Q,
            NOPE_DIM_K=NOPE_DIM_K,
            ROPE_DIM=ROPE_DIM,
            BLOCK=BLOCK_ROPE,
            NUM_Q_HEADS=NUM_HEADS_Q,
            NUM_K_HEADS=NUM_HEADS_K,
            NUM_CORES=NUM_CORES,
        )
    with dl.async_task(scope=dl.async_task.cube):
        # batch matmul
        matmul_qn_kernel(
            mat_a=Q,
            mat_b=MAT_B,
            mat_c=MAT_C,
            M=total_seq_len,
            N=N,
            K=K,
            NUM_CORES=NUM_CORES,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            BLOCK_TRESHHOLD=BLOCK_TRESHHOLD,
            Q_HEAD_DIM=NOPE_DIM_Q + ROPE_DIM,
            NUM_HEADS=NUM_HEADS_Q,
        )


def fused_matmul_qn_and_rotary_pos_emb_qk_triton(
    q: Tensor, weight_uk: Tensor, k: Tensor, cos: Tensor, sin: Tensor
):
    assert cos.shape == sin.shape
    rope_dim = cos.size(-1)
    seq_len = cos.numel() // cos.size(-1)
    assert q.is_contiguous() and k.is_contiguous()
    num_heads_q, nope_dim, kv_lora_rank = weight_uk.shape
    q = q.view(seq_len, num_heads_q, rope_dim + nope_dim)
    num_heads_k = 1
    k = k.view(seq_len, num_heads_k, rope_dim + kv_lora_rank)
    assert seq_len == q.numel() // q.size(-1) // q.size(-2)
    assert seq_len == k.numel() // k.size(-1) // k.size(-2)
    mat_c = torch.empty(
        (seq_len, num_heads_q, kv_lora_rank), dtype=q.dtype, device=q.device
    )
    q_embed = torch.empty(
        (seq_len, num_heads_q, rope_dim), dtype=q.dtype, device=q.device
    )
    k_embed = torch.empty(
        (seq_len, num_heads_k, rope_dim), dtype=k.dtype, device=k.device
    )
    fused_matmul_qn_and_rotary_pos_emb_qk_kernel[(NUM_CORES,)](
        Q=q,
        MAT_B=weight_uk,
        MAT_C=mat_c,
        N=kv_lora_rank,
        K=nope_dim,
        NUM_CORES=NUM_CORES,
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=nope_dim,
        BLOCK_TRESHHOLD=6,
        NUM_HEADS_Q=num_heads_q,
        # rotary pos emb
        Q_INPUT=q,
        K_INPUT=k,
        COS=cos,
        SIN=sin,
        Q_EMBED=q_embed,
        K_EMBED=k_embed,
        total_seq_len=seq_len,
        NOPE_DIM_Q=q.size(-1) - rope_dim,
        NOPE_DIM_K=k.size(-1) - rope_dim,
        ROPE_DIM=rope_dim,
        BLOCK_ROPE=256,
        NUM_HEADS_K=num_heads_k,
    )
    return mat_c, q_embed, k_embed


@triton.jit
def rms_norm_ckv_kernel(
    X,
    W,
    O,
    n_rows,
    input_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    eps: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    # X是matmul_ckv_kr_out，加载kv_lora_rank部分进行rms norm
    # X shape: [b, seq, 1, kv_lora_rank + qk_rope_head_dim]
    pid = tl.program_id(0)
    dim_range = tl.arange(0, KV_LORA_RANK)
    w = tl.load(W + dim_range)
    w = tl.expand_dims(w, 0)
    w = tl.broadcast_to(w, (BLOCK, KV_LORA_RANK))
    NUM_BLOCKS = tl.cdiv(n_rows, BLOCK)
    for row_block_id in range(pid, NUM_BLOCKS, NUM_CORES):
        pos_offset = row_block_id * BLOCK + tl.arange(0, BLOCK)
        pos_mask = (pos_offset < n_rows)[:, None]
        x = tl.load(
            X + pos_offset[:, None] * input_row_stride + dim_range[None, :],
            mask=pos_mask,
        )
        xf = x.to(tl.float32)
        var = tl.sum(xf * xf, 1) / KV_LORA_RANK
        qrt = tl.expand_dims(tl.math.rsqrt(var + eps), 1)
        out = xf * tl.broadcast_to(qrt, (BLOCK, KV_LORA_RANK))
        out = w * out.to(x.dtype)
        tl.store(
            O + pos_offset[:, None] * output_row_stride + dim_range[None, :],
            out,
            mask=pos_mask,
        )


def rms_norm_ckv_triton(
    input: Tensor,
    weight: Tensor,
    kv_lora_rank: int,
    eps: float = 1e-6,
):
    assert input.is_contiguous()
    assert kv_lora_rank <= input.size(-1)
    seq_len = input.numel() // input.size(-1)
    output = torch.empty(
        (input.size(0), input.size(1), input.size(2), kv_lora_rank),
        dtype=input.dtype,
        device=input.device,
    )
    rms_norm_ckv_kernel[(NUM_CORES,)](
        X=input,
        W=weight,
        O=output,
        n_rows=seq_len,
        input_row_stride=input.stride(-2),
        output_row_stride=output.stride(-2),
        eps=eps,
        KV_LORA_RANK=kv_lora_rank,
        BLOCK=128,
        NUM_CORES=NUM_CORES,
    )
    return output


@triton.jit
def fused_matmul_qb_and_rmsnorm_kv_kernel(
    MAT_A,
    MAT_B,
    MAT_C,
    total_seq_len,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
    # rms_norm ckv
    KV,
    W,
    RMS_NORM_O,
    input_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_NORM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    with dl.async_task(scope=dl.async_task.cube):
        matmul_kernel(
            mat_a=MAT_A,
            mat_b=MAT_B,
            mat_c=MAT_C,
            M=total_seq_len,
            N=N,
            K=K,
            NUM_CORES=NUM_CORES,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            BLOCK_TRESHHOLD=BLOCK_TRESHHOLD,
        )
    with dl.async_task(scope=dl.async_task.vector):
        rms_norm_ckv_kernel(
            X=KV,
            W=W,
            O=RMS_NORM_O,
            n_rows=total_seq_len,
            input_row_stride=input_row_stride,
            output_row_stride=output_row_stride,
            eps=eps,
            KV_LORA_RANK=NOPE_DIM,
            BLOCK=BLOCK_NORM,
            NUM_CORES=NUM_CORES,
        )


def fused_matmul_qb_and_rmsnorm_kv_triton(
    mat_a: Tensor, weight_uq_qr: Tensor, kv: Tensor, rmsnormGammaCkv: Tensor
):
    mat_b = weight_uq_qr
    assert mat_a.is_contiguous()
    assert mat_a.size(0) == kv.size(0)
    seq_len = mat_a.size(0)
    assert mat_a.size(-1) == mat_b.size(0)
    assert len(mat_a.shape) == 2 and len(mat_b.shape) == 2
    m = mat_a.shape[0]
    k = mat_a.shape[1]
    n = mat_b.shape[1]
    mat_c = torch.empty(m, n, dtype=mat_a.dtype, device=mat_a.device)
    # rms_norm_ckv
    assert kv.is_contiguous()
    kv_lora_rank = rmsnormGammaCkv.size(-1)
    assert kv_lora_rank <= kv.size(-1)
    rms_norm_output = torch.empty(
        (seq_len, 1, kv_lora_rank), dtype=mat_a.dtype, device=mat_a.device
    )
    fused_matmul_qb_and_rmsnorm_kv_kernel[(NUM_CORES,)](
        MAT_A=mat_a,
        MAT_B=mat_b,
        MAT_C=mat_c,
        total_seq_len=m,
        N=n,
        K=k,
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=256,
        BLOCK_TRESHHOLD=6,
        # rms_norm ckv
        KV=kv,
        W=rmsnormGammaCkv,
        RMS_NORM_O=rms_norm_output,
        input_row_stride=kv.stride(-2),
        output_row_stride=rms_norm_output.stride(-2),
        eps=1e-6,
        BLOCK_NORM=16,
        NOPE_DIM=kv_lora_rank,
        NUM_CORES=NUM_CORES,
    )
    return mat_c.view(seq_len, n), rms_norm_output


def mla_prolog_triton_v2(
    out_matmul_cq,
    hidden_states,
    weightDkvKr,
    weight_uq_qr,
    rmsnormGammaCq,
    rmsnormGammaCkv,
    cos,
    sin,
    weight_uk,
):
    kv, rmsnorm_out = fused_matmul_ckvkr_and_rms_norm_cq_triton(
        mat_a=hidden_states,
        mat_b=weightDkvKr,
        input=out_matmul_cq,
        weight=rmsnormGammaCq,
        eps=1e-06,
    )
    matmul_qc_qr_out, kv_cache = fused_matmul_qb_and_rmsnorm_kv_triton(
        rmsnorm_out, weight_uq_qr, kv, rmsnormGammaCkv
    )
    q, q_rope, kr_cache = fused_matmul_qn_and_rotary_pos_emb_qk_triton(
        q=matmul_qc_qr_out, weight_uk=weight_uk, k=kv, cos=cos, sin=sin
    )
    return (
        q,
        q_rope,
        kv_cache,
        kr_cache,
    )
