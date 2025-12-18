import torch
import triton
import triton.language as tl
import triton.language.extra.deeplink as dl
from torch import Tensor
from dlblas.kernels.ascend.apply_rotary_pos_emb import partial_rotary_emb_kernel
from dlblas.kernels.ascend.fused_rmsnorm_partial_rope import (
    fused_single_norm_and_partial_rope_kernel,
)
from dlblas.utils.op_helper import grouped_launch_diagonal
from dlblas.kernels.ascend.rms_norm import rms_norm_block_kernel
from dlblas.utils.device_utils import NUM_CORES


@triton.jit
def partial_matmul_kernel(
    mat_a,
    mat_b,
    mat_c,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    B_STRIDE_K: tl.constexpr,
    B_N_START: tl.constexpr,
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
        b_n_start = B_N_START + task_n_idx * BLOCK_N
        c_n_start = task_n_idx * BLOCK_N
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
            mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * B_STRIDE_K)[:, None] + (
                b_n_start + tl.arange(0, BLOCK_N)
            )[None, :]
            mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & (
                (b_n_start + tl.arange(0, BLOCK_N)) < (B_N_START + N)
            )[None, :]
            mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
            dl.compile_hint(mat_b_block, "dot_pad_only_k")
            mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
        mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (
            c_n_start + tl.arange(0, BLOCK_N)
        )[None, :]
        mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
            (c_n_start + tl.arange(0, BLOCK_N)) < N
        )[None, :]
        tl.store(
            mat_c + mat_c_offset,
            mat_c_block.to(mat_c.dtype.element_ty),
            mask=mat_c_mask,
        )


def partial_matmul_triton(mat_a: Tensor, mat_b: Tensor, b_n_start: int, n: int):
    assert mat_a.is_contiguous()
    assert mat_b.is_contiguous()

    assert mat_a.size(-1) == mat_b.size(0)
    assert len(mat_a.shape) == 2 and len(mat_b.shape) == 2
    m = mat_a.shape[0]
    k = mat_a.shape[1]
    assert b_n_start + n <= mat_b.shape[1]
    mat_c = torch.empty(m, n, dtype=mat_a.dtype, device=mat_a.device)

    """
    NPU芯片更加亲和512B对齐场景,如下分块通用性能较好,可以使用autotune选取最优
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256    
    """
    partial_matmul_kernel[(NUM_CORES,)](
        mat_a=mat_a,
        mat_b=mat_b,
        mat_c=mat_c,
        M=m,
        N=n,
        K=k,
        B_STRIDE_K=mat_b.stride(0),
        B_N_START=b_n_start,
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=256,
        BLOCK_TRESHHOLD=6,
        NUM_CORES=NUM_CORES,
    )
    return mat_c


# @triton.jit
# def fused_partial_matmul_and_sigmoid_kernel(
#     mat_a,
#     mat_b,
#     mat_c,
#     M,
#     N: tl.constexpr,
#     K: tl.constexpr,
#     B_STRIDE_K: tl.constexpr,
#     B_N_START: tl.constexpr,
#     NUM_CORES: tl.constexpr,
#     BLOCK_M: tl.constexpr,
#     BLOCK_N: tl.constexpr,
#     BLOCK_K: tl.constexpr,
#     BLOCK_TRESHHOLD: tl.constexpr,
# ):
#     pid = tl.program_id(axis=0)
#     NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
#     NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
#     NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
#     for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
#         task_m_idx, task_n_idx = grouped_launch_diagonal(
#             block_idx, NUM_BLOCKS_M, NUM_BLOCKS_N, BLOCK_TRESHHOLD
#         )
#         m_start = task_m_idx * BLOCK_M
#         b_n_start = B_N_START + task_n_idx * BLOCK_N
#         c_n_start = task_n_idx * BLOCK_N
#         mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
#         for k_start in range(0, K, BLOCK_K):
#             mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (
#                 k_start + tl.arange(0, BLOCK_K)
#             )[None, :]
#             mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
#                 (k_start + tl.arange(0, BLOCK_K)) < K
#             )[None, :]
#             mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)
#             dl.compile_hint(mat_a_block, "dot_pad_only_k")
#             mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * B_STRIDE_K)[:, None] + (
#                 b_n_start + tl.arange(0, BLOCK_N)
#             )[None, :]
#             mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & (
#                 (b_n_start + tl.arange(0, BLOCK_N)) < (B_N_START + N)
#             )[None, :]
#             mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
#             dl.compile_hint(mat_b_block, "dot_pad_only_k")
#             mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
#         mat_c_sigmoid_block = tl.sigmoid(mat_c_block)
#         mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (
#             c_n_start + tl.arange(0, BLOCK_N)
#         )[None, :]
#         mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
#             (c_n_start + tl.arange(0, BLOCK_N)) < N
#         )[None, :]
#         tl.store(
#             mat_c + mat_c_offset,
#             mat_c_sigmoid_block.to(mat_c.dtype.element_ty),
#             mask=mat_c_mask,
#         )


# def partial_matmul_and_sigmoid_triton(
#     mat_a: Tensor, mat_b: Tensor, b_n_start: int, n: int
# ):
#     assert mat_a.is_contiguous()
#     assert mat_b.is_contiguous()

#     assert mat_a.size(-1) == mat_b.size(0)
#     assert len(mat_a.shape) == 2 and len(mat_b.shape) == 2
#     m = mat_a.shape[0]
#     k = mat_a.shape[1]
#     assert b_n_start + n <= mat_b.shape[1]
#     mat_c = torch.empty(m, n, dtype=mat_a.dtype, device=mat_a.device)

#     """
#     NPU芯片更加亲和512B对齐场景,如下分块通用性能较好,可以使用autotune选取最优
#     BLOCK_M = 128
#     BLOCK_N = 256
#     BLOCK_K = 256
#     """
#     partial_matmul_and_sigmoid_triton[(NUM_CORES,)](
#         mat_a=mat_a,
#         mat_b=mat_b,
#         mat_c=mat_c,
#         M=m,
#         N=n,
#         K=k,
#         B_STRIDE_K=mat_b.stride(0),
#         B_N_START=b_n_start,
#         BLOCK_M=128,
#         BLOCK_N=256,
#         BLOCK_K=256,
#         BLOCK_TRESHHOLD=6,
#         NUM_CORES=NUM_CORES,
#     )
#     return mat_c


@triton.jit
def fused_rmsnorm_and_sigmoid_kernel(
    QQ,
    W,
    O_NORM,
    O_SIGMOID,
    n_rows,
    eps: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    # 拆分QQ，一半做rmsnorm，另一半做sigmoid
    pid = tl.program_id(0)
    HALF_DIM: tl.constexpr = DIM // 2
    offsets = tl.arange(0, HALF_DIM)
    w = tl.load(W + offsets)
    w = tl.expand_dims(w, 0)
    w = tl.broadcast_to(w, (BLOCK, HALF_DIM))
    NUM_BLOCKS = tl.cdiv(n_rows, BLOCK)
    for row_block_id in range(pid, NUM_BLOCKS, NUM_CORES):
        pos_offset = row_block_id * BLOCK + tl.arange(0, BLOCK)
        pos_mask = (pos_offset < n_rows)[:, None]
        # rms_norm
        q = tl.load(QQ + pos_offset[:, None] * DIM + offsets[None, :], mask=pos_mask)
        qf = q.to(tl.float32)
        var = tl.sum(qf * qf, 1) / HALF_DIM

        qrt = tl.expand_dims(tl.math.rsqrt(var + eps), 1)
        out = qf * tl.broadcast_to(qrt, (BLOCK, HALF_DIM))
        out = w * out.to(q.dtype)
        tl.store(
            O_NORM + pos_offset[:, None] * HALF_DIM + offsets[None, :],
            out,
            mask=pos_mask,
        )
        # sigmoid
        q = tl.load(
            QQ + pos_offset[:, None] * DIM + offsets[None, :] + HALF_DIM, mask=pos_mask
        )
        out = tl.sigmoid(q.to(tl.float32))
        tl.store(
            O_SIGMOID + pos_offset[:, None] * HALF_DIM + offsets[None, :],
            out.to(q.dtype),
            mask=pos_mask,
        )


@triton.jit
def fused_matmul_norm_sigmoid_kernel(
    MAT_A,
    MAT_B,
    MAT_C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    B_STRIDE_K: tl.constexpr,
    B_N_START: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
    QQ,
    W,
    O_NORM,
    O_SIGMOID,
    n_rows,
    eps: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_VEC: tl.constexpr,
):
    with dl.async_task(scope=dl.async_task.cube):
        partial_matmul_kernel(
            mat_a=MAT_A,
            mat_b=MAT_B,
            mat_c=MAT_C,
            M=M,
            N=N,
            K=K,
            B_STRIDE_K=B_STRIDE_K,
            B_N_START=B_N_START,
            NUM_CORES=NUM_CORES,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            BLOCK_TRESHHOLD=BLOCK_TRESHHOLD,
        )
    with dl.async_task(scope=dl.async_task.vector):
        fused_rmsnorm_and_sigmoid_kernel(
            QQ=QQ,
            W=W,
            O_NORM=O_NORM,
            O_SIGMOID=O_SIGMOID,
            n_rows=n_rows,
            eps=eps,
            DIM=DIM,
            BLOCK=BLOCK_VEC,
            NUM_CORES=NUM_CORES,
        )


def fused_matmul_norm_sigmoid_triton(
    hidden_states: Tensor,
    weight: Tensor,
    b_n_start: int,
    n: int,
    num_q_heads: int,
    head_dim: int,
    qq: Tensor,
    rmsnorm_gamma_q: Tensor,
    eps: float = 1e-6,
):
    # k matmul + q norm + gate_sigmoid
    assert hidden_states.is_contiguous()
    assert qq.is_contiguous()
    assert hidden_states.size(-1) == weight.size(0)
    assert len(hidden_states.shape) == 2 and len(weight.shape) == 2
    m = hidden_states.shape[0]
    k = hidden_states.shape[1]
    assert b_n_start + n <= weight.shape[1]
    mat_c = torch.empty(m, n, dtype=hidden_states.dtype, device=hidden_states.device)
    total_seq_len = hidden_states.numel() // hidden_states.size(-1)
    o_norm = torch.empty(
        (total_seq_len, num_q_heads, head_dim), dtype=qq.dtype, device=qq.device
    )
    o_sigmoid = torch.empty(
        (total_seq_len, num_q_heads, head_dim), dtype=qq.dtype, device=qq.device
    )
    fused_matmul_norm_sigmoid_kernel[(NUM_CORES,)](
        MAT_A=hidden_states,
        MAT_B=weight,
        MAT_C=mat_c,
        M=m,
        N=n,
        K=k,
        B_STRIDE_K=weight.stride(0),
        B_N_START=b_n_start,
        NUM_CORES=NUM_CORES,
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=256,
        BLOCK_TRESHHOLD=6,
        QQ=qq,
        W=rmsnorm_gamma_q,
        O_NORM=o_norm,
        O_SIGMOID=o_sigmoid,
        n_rows=total_seq_len * num_q_heads,
        eps=eps,
        DIM=head_dim * 2,
        BLOCK_VEC=32,
    )
    return mat_c, o_norm, o_sigmoid


# @triton.jit
# def fused_single_norm_rope_matmul_kernel(
#     # partial_matmul
#     MAT_A,
#     MAT_B,
#     MAT_C,
#     M,
#     N: tl.constexpr,
#     K: tl.constexpr,
#     B_STRIDE_K: tl.constexpr,
#     B_N_START: tl.constexpr,
#     NUM_CORES: tl.constexpr,
#     BLOCK_M: tl.constexpr,
#     BLOCK_N: tl.constexpr,
#     BLOCK_K: tl.constexpr,
#     BLOCK_TRESHHOLD: tl.constexpr,
#     # rmsnorm
#     INPUT,
#     NORM_WEIGHT,
#     seq_len,
#     eps: tl.constexpr,
#     NUM_HEADS: tl.constexpr,
#     TOTAL_HEAD_DIM: tl.constexpr,
#     # rotary_pos_emb
#     COS,
#     SIN,
#     OUTPUT,
#     ROPE_HEAD_DIM: tl.constexpr,
# ):
#     # matmul + norm + rotary_pos_emb
#     with dl.async_task(scope=dl.async_task.cube):
#         partial_matmul_kernel(
#             mat_a=MAT_A,
#             mat_b=MAT_B,
#             mat_c=MAT_C,
#             M=M,
#             N=N,
#             K=K,
#             B_STRIDE_K=B_STRIDE_K,
#             B_N_START=B_N_START,
#             NUM_CORES=NUM_CORES,
#             BLOCK_M=BLOCK_M,
#             BLOCK_N=BLOCK_N,
#             BLOCK_K=BLOCK_K,
#             BLOCK_TRESHHOLD=BLOCK_TRESHHOLD,
#         )
#     with dl.async_task(scope=dl.async_task.vector):
#         fused_single_norm_and_partial_rope_kernel(
#             # rmsnorm
#             INPUT=INPUT,
#             NORM_WEIGHT=NORM_WEIGHT,
#             seq_len=seq_len,
#             eps=eps,
#             NUM_HEADS=NUM_HEADS,
#             TOTAL_HEAD_DIM=TOTAL_HEAD_DIM,
#             # rotary_pos_emb
#             COS=COS,
#             SIN=SIN,
#             OUTPUT=OUTPUT,
#             ROPE_HEAD_DIM=ROPE_HEAD_DIM,
#             NUM_CORES=NUM_CORES,
#         )


# def fused_single_norm_rope_matmul_triton(
#     # partial_matmul
#     hidden_states: Tensor,
#     weight: Tensor,
#     b_n_start: int,
#     n: int,
#     mat_num_heads: int,
#     mat_head_dim: int,
#     # fused norm and partial rope
#     x: Tensor,
#     norm_weight: Tensor,
#     cos: Tensor,
#     sin: Tensor,
#     partial_rotary_factor: float,
#     eps: float = 1e-6,
#     inplace: bool = True,
# ):
#     # partial_matmul
#     assert hidden_states.is_contiguous()
#     assert hidden_states.size(-1) == weight.size(0)
#     assert len(hidden_states.shape) == 2 and len(weight.shape) == 2
#     m = hidden_states.shape[0]
#     k = hidden_states.shape[1]
#     assert b_n_start + n <= weight.shape[1]
#     assert n == mat_num_heads * mat_head_dim
#     mat_c = torch.empty(
#         (m, mat_num_heads, mat_head_dim), dtype=weight.dtype, device=weight.device
#     )
#     # fused norm and partial rope
#     assert x.is_contiguous()
#     head_dim = norm_weight.shape[0]
#     seq_len = cos.numel() // cos.size(-1)
#     assert seq_len == x.numel() // x.size(-1) // x.size(-2)
#     assert head_dim == x.size(-1)
#     rotary_dim = int(head_dim * partial_rotary_factor)
#     assert rotary_dim == cos.size(-1) and rotary_dim == sin.size(-1)
#     num_heads = x.size(-2)
#     if inplace:
#         x_embed = x
#     else:
#         x_embed = torch.empty_like(x)
#     fused_matmul_norm_rotary_emb_kernel[(NUM_CORES,)](
#         # partial_matmul
#         MAT_A=hidden_states,
#         MAT_B=weight,
#         MAT_C=mat_c,
#         M=m,
#         N=n,
#         K=k,
#         B_STRIDE_K=weight.stride(0),
#         B_N_START=b_n_start,
#         NUM_CORES=NUM_CORES,
#         BLOCK_M=128,
#         BLOCK_N=256,
#         BLOCK_K=256,
#         BLOCK_TRESHHOLD=6,
#         # rmsnorm
#         INPUT=x,
#         NORM_WEIGHT=norm_weight,
#         seq_len=seq_len,
#         eps=eps,
#         NUM_HEADS=num_heads,
#         TOTAL_HEAD_DIM=head_dim,
#         # rotary_pos_emb
#         COS=cos,
#         SIN=sin,
#         OUTPUT=x_embed,
#         ROPE_HEAD_DIM=rotary_dim,
#         NUM_CORES=NUM_CORES,
#     )
#     return mat_c, x_embed


@triton.jit
def fused_matmul_norm_rotary_emb_kernel(
    # v_partial_matmul
    MAT_A,
    MAT_B,
    MAT_C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    B_STRIDE_K: tl.constexpr,
    B_N_START: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
    # k_rmsnorm
    K_INPUT,
    K_WEIGHT,
    K_NORM,
    n_rows,
    eps: tl.constexpr,
    K_DIM: tl.constexpr,
    # q_k_rotary_pos
    Q_INPUT,
    COS,
    SIN,
    Q_EMBED,
    K_EMBED,
    seq_len,
    stride_qs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kh: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    BLOCK_NORM: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
):
    # v matmul + k_norm + q_k_rotary_pos
    with dl.async_task(scope=dl.async_task.cube):
        partial_matmul_kernel(
            mat_a=MAT_A,
            mat_b=MAT_B,
            mat_c=MAT_C,
            M=M,
            N=N,
            K=K,
            B_STRIDE_K=B_STRIDE_K,
            B_N_START=B_N_START,
            NUM_CORES=NUM_CORES,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            BLOCK_TRESHHOLD=BLOCK_TRESHHOLD,
        )
    with dl.async_task(scope=dl.async_task.vector):
        rms_norm_block_kernel(
            input=K_INPUT,
            weight=K_WEIGHT,
            output=K_NORM,
            n_rows=n_rows,
            input_row_stride=K_DIM,
            eps=eps,
            N_COLS=K_DIM,
            BLOCK=BLOCK_NORM,
            NUM_CORES=NUM_CORES,
        )
        partial_rotary_emb_kernel(
            Q=Q_INPUT,
            K=K_INPUT,
            COS=COS,
            SIN=SIN,
            Q_EMBED=Q_EMBED,
            K_EMBED=K_EMBED,
            seq_len=seq_len,
            stride_qs=stride_qs,
            stride_qh=stride_qh,
            stride_ks=stride_ks,
            stride_kh=stride_kh,
            DIM=DIM_ROPE,
            BLOCK=BLOCK_ROPE,
            NUM_Q_HEADS=NUM_Q_HEADS,
            NUM_K_HEADS=NUM_K_HEADS,
            NUM_CORES=NUM_CORES,
        )


def fused_matmul_norm_rotary_emb_triton(
    # v_partial_matmul
    hidden_states: Tensor,
    weight: Tensor,
    b_n_start: int,
    n: int,
    # k_rmsnorm
    k_input: Tensor,
    rmsnorm_gamma_k: Tensor,
    eps: float,
    num_kv_heads: int,
    head_dim: int,
    # q_k_rotary_pos
    q_input: Tensor,
    cos: Tensor,
    sin: Tensor,
    partial_rotary_factor: float,
    inplace: bool = False,
):
    # v_partial_matmul
    assert hidden_states.is_contiguous()
    assert hidden_states.size(-1) == weight.size(0)
    assert len(hidden_states.shape) == 2 and len(weight.shape) == 2
    m = hidden_states.shape[0]
    k = hidden_states.shape[1]
    assert b_n_start + n <= weight.shape[1]
    assert n == num_kv_heads * head_dim
    v = torch.empty(
        (m, num_kv_heads, head_dim), dtype=weight.dtype, device=weight.device
    )
    # k_rmsnorm
    assert k_input.is_contiguous()
    k_dim = k_input.size(-1)
    n_rows = k_input.numel() // k_input.size(-1)
    # q_k_rotary_pos
    assert q_input.is_contiguous() and len(q_input.shape) >= 3
    assert q_input.shape[0] == k_input.shape[0]
    assert q_input.size(-1) == cos.size(-1)
    assert q_input.size(-1) == sin.size(-1)
    if inplace:
        q_embed, k_embed, k_norm = q_input, k_input, k_input
    else:
        q_embed = torch.empty_like(q_input)
        k_embed = torch.empty_like(k_input)
        k_norm = torch.empty_like(k_input)
    seq_len = cos.numel() // cos.size(-1)
    assert seq_len == q_input.numel() // q_input.size(-1) // q_input.size(-2)
    rotary_dim = int(q_input.size(-1) * partial_rotary_factor)
    fused_matmul_norm_rotary_emb_kernel[(NUM_CORES,)](
        # v_partial_matmul
        MAT_A=hidden_states,
        MAT_B=weight,
        MAT_C=v,
        M=m,
        N=n,
        K=k,
        B_STRIDE_K=weight.stride(0),
        B_N_START=b_n_start,
        NUM_CORES=NUM_CORES,
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=256,
        BLOCK_TRESHHOLD=6,
        # k_rmsnorm
        K_INPUT=k_input,
        K_WEIGHT=rmsnorm_gamma_k,
        K_NORM=k_norm,
        n_rows=n_rows,
        eps=eps,
        K_DIM=k_dim,
        # q_k_rotary_pos
        Q_INPUT=q_input,
        COS=cos,
        SIN=sin,
        Q_EMBED=q_embed,
        K_EMBED=k_embed,
        seq_len=seq_len,
        stride_qs=q_input.stride(-3),
        stride_qh=q_input.stride(-2),
        stride_ks=k_input.stride(-3),
        stride_kh=k_input.stride(-2),
        DIM_ROPE=rotary_dim,
        BLOCK_ROPE=64,
        BLOCK_NORM=32,
        NUM_Q_HEADS=q_input.size(-2),
        NUM_K_HEADS=k_input.size(-2),
    )
    return v, q_embed, k_embed


def attention_prolog_triton_v1(
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
    # q matmul | k matmul + q norm + gate_sigmoid | v matmul + k_norm + q_k_rotary_pos
    seq_len = hidden_states.numel() // hidden_states.size(-1)
    qq = partial_matmul_triton(
        hidden_states, weight, b_n_start=0, n=2 * num_q_heads * head_dim
    )
    k, q_norm, q_sigmoid = fused_matmul_norm_sigmoid_triton(
        hidden_states,
        weight,
        b_n_start=2 * num_q_heads * head_dim,
        n=num_kv_heads * head_dim,
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        qq=qq,
        rmsnorm_gamma_q=rmsnorm_gamma_q,
        eps=eps,
    )
    v, q_rope, k_rope = fused_matmul_norm_rotary_emb_triton(
        # v_partial_matmul
        hidden_states=hidden_states,
        weight=weight,
        b_n_start=2 * num_q_heads * head_dim + num_kv_heads * head_dim,
        n=num_kv_heads * head_dim,
        # k_rmsnorm
        k_input=k.view(seq_len, num_kv_heads, head_dim),
        rmsnorm_gamma_k=rmsnorm_gamma_k,
        eps=eps,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        # q_k_rotary_pos
        q_input=q_norm.view(seq_len, num_q_heads, head_dim),
        cos=cos,
        sin=sin,
        partial_rotary_factor=partial_rotary_factor,
        inplace=True,
    )
    return q_rope, k_rope, v, q_sigmoid


def partial_matmul():
    pass


def rmsnorm_rotary_pos_emb():
    pass


def sigmoid():
    pass


def attention_prolog_mega(
    hidden_states,
    weight,
    qq,
    gate,
    key,
    value,
    rmsnorm_gamma_q,
    rmsnorm_gamma_k,
    cos,
    sin,
    query0_n_start,
    query0_n_size,
    query1_n_start,
    query1_n_size,
    key_n_start,
    key_n_size,
    value_n_start,
    value_n_size,
):
    with dl.async_task(scope=dl.async_task.cube):
        partial_matmul(hidden_states, weight, qq, query0_n_start, query0_n_size)
        partial_matmul(hidden_states, weight, key, key_n_start, key_n_size)
        partial_matmul(hidden_states, weight, value, value_n_start, value_n_size)

    with dl.async_task(scope=dl.async_task.vector):
        dl.sync_block_all()
        sigmoid(qq, gate)
        rmsnorm_rotary_pos_emb(qq, rmsnorm_gamma_q, cos, sin)
        dl.sync_block_all()
        rmsnorm_rotary_pos_emb(key, rmsnorm_gamma_k, cos, sin)
