import torch
import triton
import triton.language as tl
import triton.language.extra.deeplink as dl
from torch import Tensor
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
    mat_c = torch.empty(b, seq_len, n, dtype=mat_a.dtype, device=mat_a.device)

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
    for head_idx in range(NUM_HEADS):
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
def rotary_pos_emb_qr_kernel(
    X,
    COS,
    SIN,
    O,
    total_seq_len,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    # q = matmul_qc_qr_out.view(total_seq_len, num_heads, q_head_dim)
    # q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    # X是q，对q_pe实现rotary_pos_emb
    pid = tl.program_id(0)
    NUM_BLOCKS = tl.cdiv(total_seq_len, BLOCK)
    Q_HEAD_DIM = NOPE_DIM + ROPE_DIM
    for seq_block_id in range(pid, NUM_BLOCKS, NUM_CORES):
        pos_offset = seq_block_id * BLOCK + tl.arange(0, BLOCK)
        pos_mask = pos_offset < total_seq_len
        half_dim: tl.constexpr = ROPE_DIM // 2
        cso_dim_offset_l = tl.arange(0, half_dim)
        cso_dim_offset_h = cso_dim_offset_l + half_dim
        seq_mask = pos_mask[:, None]
        cs_offset_l = pos_offset[:, None] * ROPE_DIM + cso_dim_offset_l[None, :]
        cs_offset_h = pos_offset[:, None] * ROPE_DIM + cso_dim_offset_h[None, :]

        cos_l = tl.load(COS + cs_offset_l, mask=seq_mask)
        cos_h = tl.load(COS + cs_offset_h, mask=seq_mask)
        sin_l = tl.load(SIN + cs_offset_l, mask=seq_mask)
        sin_h = tl.load(SIN + cs_offset_h, mask=seq_mask)

        x_dim_offset_l = NOPE_DIM + tl.arange(0, half_dim)
        x_dim_offset_h = x_dim_offset_l + half_dim
        for head_id in range(NUM_HEADS):
            x_base_offset = (pos_offset[:, None] * NUM_HEADS * Q_HEAD_DIM) + (
                head_id * Q_HEAD_DIM
            )
            x_l = tl.load(X + x_base_offset + x_dim_offset_l[None, :], mask=seq_mask)
            x_h = tl.load(X + x_base_offset + x_dim_offset_h[None, :], mask=seq_mask)
            o_l = x_l * cos_l - x_h * sin_l
            o_h = x_h * cos_h + x_l * sin_h
            o_base_offset = (pos_offset[:, None] * NUM_HEADS * ROPE_DIM) + (
                head_id * ROPE_DIM
            )
            tl.store(O + o_base_offset + cso_dim_offset_l[None, :], o_l, mask=seq_mask)
            tl.store(O + o_base_offset + cso_dim_offset_h[None, :], o_h, mask=seq_mask)


def rotary_pos_emb_qr_triton(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    nope_dim: int,
    rope_dim: int,
):
    assert x.is_contiguous()
    assert rope_dim == cos.size(-1) and rope_dim == sin.size(-1)
    assert x.size(-1) == nope_dim + rope_dim
    total_seq_len = cos.numel() // cos.size(-1)
    assert total_seq_len == x.numel() // x.size(-1) // x.size(-2)
    num_heads = x.size(-2)
    if len(x.shape) == 4:
        x_embed = torch.empty(
            (x.shape[0], x.shape[1], num_heads, rope_dim),
            dtype=x.dtype,
            device=x.device,
        )
    elif len(x.shape) == 3:
        x_embed = torch.empty(
            (total_seq_len, num_heads, rope_dim), dtype=x.dtype, device=x.device
        )
    else:
        raise RuntimeError("not support")
    rotary_pos_emb_qr_kernel[(NUM_CORES,)](
        x,
        cos,
        sin,
        x_embed,
        total_seq_len=total_seq_len,
        NOPE_DIM=nope_dim,
        ROPE_DIM=rope_dim,
        BLOCK=128,
        NUM_HEADS=num_heads,
        NUM_CORES=NUM_CORES,
    )
    return x_embed


@triton.jit
def fused_matmul_qn_and_rotary_pos_emb_qr_kernel(
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
    NUM_HEADS: tl.constexpr,
    # rotary pos emb
    COS,
    SIN,
    O,
    total_seq_len,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    with dl.async_task(scope=dl.async_task.vector):
        rotary_pos_emb_qr_kernel(
            X=Q,
            COS=COS,
            SIN=SIN,
            O=O,
            total_seq_len=total_seq_len,
            NOPE_DIM=NOPE_DIM,
            ROPE_DIM=ROPE_DIM,
            BLOCK=BLOCK,
            NUM_HEADS=NUM_HEADS,
            NUM_CORES=NUM_CORES,
        )
    with dl.async_task(scope=dl.async_task.cube):
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
            Q_HEAD_DIM=NOPE_DIM + ROPE_DIM,
            NUM_HEADS=NUM_HEADS,
        )


def fused_matmul_qn_and_rotary_pos_emb_qr_triton(
    q: Tensor, weight_uk: Tensor, cos: Tensor, sin: Tensor
):
    rope_dim = cos.size(-1)
    assert rope_dim == sin.size(-1)
    assert q.is_contiguous()
    num_heads, nope_dim, kv_lora_rank = weight_uk.shape
    b, seq_len = q.shape[0], q.shape[1]
    q = q.view(b, seq_len, num_heads, rope_dim + nope_dim)
    total_seq_len = cos.numel() // cos.size(-1)
    assert total_seq_len == q.numel() // q.size(-1) // q.size(-2)
    mat_c = torch.empty(
        (b, seq_len, num_heads, kv_lora_rank), dtype=q.dtype, device=q.device
    )
    q_embed = torch.empty(
        (b, seq_len, num_heads, rope_dim), dtype=q.dtype, device=q.device
    )
    fused_matmul_qn_and_rotary_pos_emb_qr_kernel[(NUM_CORES,)](
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
        NUM_HEADS=num_heads,
        # rotary pos emb
        COS=cos,
        SIN=sin,
        O=q_embed,
        total_seq_len=total_seq_len,
        NOPE_DIM=nope_dim,
        ROPE_DIM=rope_dim,
        BLOCK=128,
    )
    return mat_c, q_embed


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
def fused_matmul_and_rms_norm_and_rotary_pos_emb_kernel(
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
    # rotary pos emb kr
    COS,
    SIN,
    KV_EMB,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_PE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
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
        rotary_pos_emb_qr_kernel(
            X=KV,
            COS=COS,
            SIN=SIN,
            O=KV_EMB,
            total_seq_len=total_seq_len,
            NOPE_DIM=NOPE_DIM,
            ROPE_DIM=ROPE_DIM,
            BLOCK=BLOCK_PE,
            NUM_HEADS=NUM_HEADS,
            NUM_CORES=NUM_CORES,
        )


def fused_matmul_and_rms_norm_and_rotary_pos_emb_triton(
    mat_a: Tensor,
    weight_uq_qr: Tensor,
    kv: Tensor,
    rmsnormGammaCkv: Tensor,
    cos: Tensor,
    sin: Tensor,
):
    mat_b = weight_uq_qr
    assert mat_a.is_contiguous()
    assert mat_a.size(0) == kv.size(0)
    assert mat_a.size(1) == kv.size(1)
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
    mat_c = torch.empty(m, n, dtype=mat_a.dtype, device=mat_a.device)
    # rms_norm_ckv
    assert kv.is_contiguous()
    kv_lora_rank = rmsnormGammaCkv.size(-1)
    assert kv_lora_rank <= kv.size(-1)
    rms_norm_output = torch.empty(
        (b, seq_len, 1, kv_lora_rank),
        dtype=mat_a.dtype,
        device=mat_a.device,
    )
    # rotary pos emb kr
    rope_dim = sin.size(-1)
    nope_dim = kv_lora_rank
    assert cos.size(-1) == sin.size(-1)
    assert kv.size(-1) == nope_dim + rope_dim
    num_heads_kv = 1
    x_embed = torch.empty(
        (b, seq_len, num_heads_kv, rope_dim), dtype=mat_a.dtype, device=mat_a.device
    )
    fused_matmul_and_rms_norm_and_rotary_pos_emb_kernel[(NUM_CORES,)](
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
        # rotary pos emb kr
        COS=cos,
        SIN=sin,
        KV_EMB=x_embed,
        NOPE_DIM=nope_dim,
        ROPE_DIM=rope_dim,
        BLOCK_PE=32,
        NUM_HEADS=num_heads_kv,
        NUM_CORES=NUM_CORES,
    )
    return mat_c.view(b, seq_len, n), rms_norm_output, x_embed
