import torch
import triton
import triton.language as tl
import triton.language.extra.deeplink as dl
from torch import Tensor
from dlblas.kernels.ascend.apply_rotary_pos_emb import partial_rotary_emb_kernel
from dlblas.kernels.ascend.fused_rmsnorm_partial_rope import (
    fused_single_norm_and_partial_rope_kernel,
)
from dlblas.kernels.ascend.qwen3_next_atten_prolog_v1 import (
    partial_matmul_kernel,
    partial_matmul_triton,
)
from dlblas.utils.op_helper import grouped_launch_diagonal
from dlblas.kernels.ascend.rms_norm import rms_norm_block_kernel
from dlblas.utils.device_utils import NUM_CORES


@triton.jit
def sigmoid_kernel(
    QQ,
    O_SIGMOID,
    n_rows,
    in_stride_row: tl.constexpr,
    SIGMOID_DIM_START: tl.constexpr,
    SIGMOID_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    # 最低维拆分QQ，一半做sigmoid
    pid = tl.program_id(0)
    offsets = tl.arange(0, SIGMOID_DIM)
    NUM_BLOCKS = tl.cdiv(n_rows, BLOCK)
    for row_block_id in range(pid, NUM_BLOCKS, NUM_CORES):
        pos_offset = row_block_id * BLOCK + tl.arange(0, BLOCK)
        pos_mask = (pos_offset < n_rows)[:, None]
        # sigmoid
        q = tl.load(
            QQ
            + pos_offset[:, None] * in_stride_row
            + offsets[None, :]
            + SIGMOID_DIM_START,
            mask=pos_mask,
        )
        out = tl.sigmoid(q.to(tl.float32))
        tl.store(
            O_SIGMOID + pos_offset[:, None] * SIGMOID_DIM + offsets[None, :],
            out.to(q.dtype),
            mask=pos_mask,
        )


@triton.jit
def fused_sigmoid_single_norm_rope_matmul_kernel(
    # partial_matmul
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
    # rmsnorm
    QQ,
    NORM_WEIGHT,
    seq_len,
    qq_stride_s: tl.constexpr,
    qq_stride_h: tl.constexpr,
    eps: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    TOTAL_HEAD_DIM: tl.constexpr,
    # sigmoid
    O_SIGMOID,
    BLOCK_SIGMOID: tl.constexpr,
    # rotary_pos_emb
    COS,
    SIN,
    OUTPUT,
    ROPE_HEAD_DIM: tl.constexpr,
):
    # matmul + norm + rotary_pos_emb
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
        sigmoid_kernel(
            QQ=QQ,
            O_SIGMOID=O_SIGMOID,
            n_rows=seq_len * NUM_HEADS,
            in_stride_row=qq_stride_h,
            SIGMOID_DIM_START=TOTAL_HEAD_DIM,
            SIGMOID_DIM=TOTAL_HEAD_DIM,
            BLOCK=BLOCK_SIGMOID,
            NUM_CORES=NUM_CORES,
        )
        fused_single_norm_and_partial_rope_kernel(
            # rmsnorm
            INPUT=QQ,
            NORM_WEIGHT=NORM_WEIGHT,
            seq_len=seq_len,
            eps=eps,
            input_stride_s=qq_stride_s,
            input_stride_h=qq_stride_h,
            NUM_HEADS=NUM_HEADS,
            TOTAL_HEAD_DIM=TOTAL_HEAD_DIM,
            # rotary_pos_emb
            COS=COS,
            SIN=SIN,
            OUTPUT=OUTPUT,
            ROPE_HEAD_DIM=ROPE_HEAD_DIM,
            NUM_CORES=NUM_CORES,
        )


def fused_sigmoid_single_norm_rope_matmul_triton(
    # partial_matmul
    hidden_states: Tensor,
    weight: Tensor,
    b_n_start: int,
    n: int,
    # fused norm and partial rope
    qq: Tensor,
    norm_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
    partial_rotary_factor: float,
    eps: float = 1e-6,
):
    # partial_matmul
    assert hidden_states.is_contiguous()
    assert hidden_states.size(-1) == weight.size(0)
    assert len(hidden_states.shape) == 2 and len(weight.shape) == 2
    m = hidden_states.shape[0]
    k = hidden_states.shape[1]
    assert b_n_start + n <= weight.shape[1]
    mat_c = torch.empty((m, n), dtype=weight.dtype, device=weight.device)
    # fused norm and partial rope
    assert qq.is_contiguous()
    head_dim = norm_weight.shape[0]
    seq_len = cos.numel() // cos.size(-1)
    assert seq_len == qq.numel() // qq.size(-1) // qq.size(-2)
    assert head_dim * 2 == qq.size(-1)
    assert partial_rotary_factor < 1.0
    rotary_dim = int(head_dim * partial_rotary_factor)
    assert rotary_dim == triton.next_power_of_2(rotary_dim)
    assert rotary_dim == cos.size(-1) and rotary_dim == sin.size(-1)
    num_heads = qq.size(-2)
    q_rope = torch.empty(
        (seq_len, num_heads, head_dim), dtype=qq.dtype, device=qq.device
    )
    gate = torch.empty((seq_len, num_heads, head_dim), dtype=qq.dtype, device=qq.device)
    fused_sigmoid_single_norm_rope_matmul_kernel[(NUM_CORES,)](
        # partial_matmul
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
        # rmsnorm
        QQ=qq,
        NORM_WEIGHT=norm_weight,
        seq_len=seq_len,
        qq_stride_s=qq.stride(-3),
        qq_stride_h=qq.stride(-2),
        eps=eps,
        NUM_HEADS=num_heads,
        TOTAL_HEAD_DIM=head_dim,
        # sigmoid
        O_SIGMOID=gate,
        BLOCK_SIGMOID=64,
        # rotary_pos_emb
        COS=cos,
        SIN=sin,
        OUTPUT=q_rope,
        ROPE_HEAD_DIM=rotary_dim,
    )
    return mat_c, q_rope, gate


@triton.jit
def fused_single_norm_rope_matmul_kernel(
    # partial_matmul
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
    # rmsnorm
    INPUT,
    NORM_WEIGHT,
    seq_len,
    eps: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    TOTAL_HEAD_DIM: tl.constexpr,
    # rotary_pos_emb
    COS,
    SIN,
    OUTPUT,
    ROPE_HEAD_DIM: tl.constexpr,
):
    # matmul + norm + rotary_pos_emb
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
        fused_single_norm_and_partial_rope_kernel(
            # rmsnorm
            INPUT=INPUT,
            NORM_WEIGHT=NORM_WEIGHT,
            seq_len=seq_len,
            eps=eps,
            input_stride_s=NUM_HEADS * TOTAL_HEAD_DIM,
            input_stride_h=TOTAL_HEAD_DIM,
            NUM_HEADS=NUM_HEADS,
            TOTAL_HEAD_DIM=TOTAL_HEAD_DIM,
            # rotary_pos_emb
            COS=COS,
            SIN=SIN,
            OUTPUT=OUTPUT,
            ROPE_HEAD_DIM=ROPE_HEAD_DIM,
            NUM_CORES=NUM_CORES,
        )


def fused_single_norm_rope_matmul_triton(
    # partial_matmul
    hidden_states: Tensor,
    weight: Tensor,
    b_n_start: int,
    n: int,
    # fused norm and partial rope
    x: Tensor,
    norm_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
    partial_rotary_factor: float,
    eps: float = 1e-6,
    inplace: bool = True,
):
    # partial_matmul
    assert hidden_states.is_contiguous()
    assert hidden_states.size(-1) == weight.size(0)
    assert len(hidden_states.shape) == 2 and len(weight.shape) == 2
    m = hidden_states.shape[0]
    k = hidden_states.shape[1]
    assert b_n_start + n <= weight.shape[1]
    mat_c = torch.empty((m, n), dtype=weight.dtype, device=weight.device)
    # fused norm and partial rope
    assert x.is_contiguous()
    head_dim = norm_weight.shape[0]
    seq_len = cos.numel() // cos.size(-1)
    assert seq_len == x.numel() // x.size(-1) // x.size(-2)
    assert head_dim == x.size(-1)
    assert partial_rotary_factor < 1.0
    rotary_dim = int(head_dim * partial_rotary_factor)
    assert rotary_dim == triton.next_power_of_2(rotary_dim)
    assert rotary_dim == cos.size(-1) and rotary_dim == sin.size(-1)
    num_heads = x.size(-2)
    if inplace:
        x_embed = x
    else:
        x_embed = torch.empty_like(x)
    fused_single_norm_rope_matmul_kernel[(NUM_CORES,)](
        # partial_matmul
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
        # rmsnorm
        INPUT=x,
        NORM_WEIGHT=norm_weight,
        seq_len=seq_len,
        eps=eps,
        NUM_HEADS=num_heads,
        TOTAL_HEAD_DIM=head_dim,
        # rotary_pos_emb
        COS=cos,
        SIN=sin,
        OUTPUT=x_embed,
        ROPE_HEAD_DIM=rotary_dim,
    )
    return mat_c, x_embed


def attention_prolog_triton_v2(
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
    # qa matmul + sigmoid qb matmul | k matmul + q norm + q rope | v matmul + k_norm + k_rope
    seq_len = hidden_states.numel() // hidden_states.size(-1)
    qq = partial_matmul_triton(
        hidden_states, weight, b_n_start=0, n=2 * num_q_heads * head_dim
    )
    # query = partial_matmul_triton(
    #     hidden_states, weight, b_n_start=0, n=num_q_heads * head_dim
    # )
    # query_tmp = query.clone().view(seq_len, num_q_heads, head_dim)
    key, q_rope, gate = fused_sigmoid_single_norm_rope_matmul_triton(
        # partial_matmul
        hidden_states=hidden_states,
        weight=weight,
        b_n_start=2 * num_q_heads * head_dim,
        n=num_kv_heads * head_dim,
        # fused sigmoid, norm and partial rope
        qq=qq.view(seq_len, num_q_heads, head_dim * 2),
        norm_weight=rmsnorm_gamma_q,
        cos=cos,
        sin=sin,
        partial_rotary_factor=partial_rotary_factor,
        eps=eps,
    )
    v, k_rope = fused_single_norm_rope_matmul_triton(
        # partial_matmul
        hidden_states=hidden_states,
        weight=weight,
        b_n_start=2 * num_q_heads * head_dim + num_kv_heads * head_dim,
        n=num_kv_heads * head_dim,
        # fused norm and partial rope
        x=key.view(seq_len, num_kv_heads, head_dim),
        norm_weight=rmsnorm_gamma_k,
        cos=cos,
        sin=sin,
        partial_rotary_factor=partial_rotary_factor,
        eps=eps,
        inplace=True,
    )
    return q_rope, k_rope, v.view(seq_len, num_kv_heads, head_dim), gate


def partial_matmul():
    pass


def rmsnorm_rotary_pos_emb():
    pass


def sigmoid():
    pass


def attention_prolog_mega(
    hidden_states,
    weight,
    query_a,
    gate,
    query_b,
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
        partial_matmul(hidden_states, weight, query_a, query0_n_start, query0_n_size)
        partial_matmul(hidden_states, weight, query_b, query1_n_start, query1_n_size)
        partial_matmul(hidden_states, weight, key, key_n_start, key_n_size)
        partial_matmul(hidden_states, weight, value, value_n_start, value_n_size)

    with dl.async_task(scope=dl.async_task.vector):
        sigmoid(query_a, gate)
        dl.sync_block_all()
        rmsnorm_rotary_pos_emb(query_b, rmsnorm_gamma_q, cos, sin)
        dl.sync_block_all()
        rmsnorm_rotary_pos_emb(key, rmsnorm_gamma_k, cos, sin)
