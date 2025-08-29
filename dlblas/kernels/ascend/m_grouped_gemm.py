
import torch
from torch import Tensor
import triton
import triton.language as tl
from dlblas.utils.op_helper import grouped_lanuch_diagonal
from dlblas.utils.device_utils import get_number_cores


def get_autotune_config():
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 4}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 5}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 6}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 7}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 4}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 5}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 6}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 7}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}),
    ]

@triton.autotune(configs=get_autotune_config(), key=['N', 'K'])
@triton.jit
def m_grouped_gemm_bKmajor_kernel(
    A,
    B,
    C,
    pad_starts,
    pad_ends,
    group_starts,
    group_ends,
    m_indices_pad,
    M_pad_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    dtype_a: tl.constexpr,
    dtype_b: tl.constexpr, 
    dtype_c: tl.constexpr,
    strideBN: tl.constexpr,
    strideBK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # GROUP_M: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
):
    
    dtypeA = tl.bfloat16 if dtype_a == 0 else tl.float16
    dtypeB = tl.bfloat16 if dtype_b == 0 else tl.float16
    dtypeC = tl.bfloat16 if dtype_c == 0 else tl.float16

    """gemm fp8 kernel."""
    BLOCKS = tl.num_programs(axis=0)
    start_pid = tl.program_id(axis=0)
    M_pad = tl.load(M_pad_ptr)
    num_pid_m = tl.cdiv(M_pad, BLOCK_M).to(tl.int32)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, BLOCKS):
        # pid_m, pid_n = grouped_launch(tile_id, M_pad, N, BLOCK_M, BLOCK_N, GROUP_M)
        pid_m, pid_n = grouped_lanuch_diagonal(tile_id, num_pid_m, num_pid_n, BLOCK_TRESHHOLD)
        group = tl.load(m_indices_pad + pid_m)
        pad_off = tl.load(pad_starts + group)
        group_start = (tl.load(group_starts + group) + (pid_m * BLOCK_M - pad_off))
        group_end = tl.load(group_ends + group)
        offs_am = group_start + tl.arange(0, BLOCK_M)
        offs_bn = (group * N + pid_n * BLOCK_N) + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs_base = A + (offs_am[:, None]*K + offs_k[None, :])
        b_ptrs_base = B + (offs_bn[:, None]*K + offs_k[None, :])
        msk_m = offs_am < group_end
        msk_n = offs_bn < (group * N +N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
            a_ptrs = a_ptrs_base + k * BLOCK_K
            b_ptrs = b_ptrs_base + k * BLOCK_K
            a = tl.load(
                a_ptrs,
                mask=msk_m[:, None] and (offs_k[None, :] < K - k * BLOCK_K),
                other=0.0,
            )
            tl.compile_hint(a, "dot_pad_only_k")
            b = tl.load(
                b_ptrs,
                mask=msk_n[:, None] and (offs_k[None, :] < K - k * BLOCK_K),
                other=0.0,
            )
            tl.compile_hint(b, "dot_pad_only_k")
            accumulator = tl.dot(a, b.T, acc=accumulator)
    
        c = accumulator.to(dtypeC)
        offs_cm = group_start
        offs_cn = (pid_n * BLOCK_N)
        offs_cm_ = offs_cm + tl.arange(0, BLOCK_M)
        offs_cn_ = offs_cn + tl.arange(0, BLOCK_N)
        c_ptrs = C + N * offs_cm_[:, None].to(tl.int64) + offs_cn_[None, :]
        c_mask = (offs_cm_[:, None] < group_end) & (offs_cn_[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

@triton.autotune(configs=get_autotune_config(), key=['N', 'K'])
@triton.jit
def m_grouped_gemm_bNmajor_kernel(
    A,
    B,
    C,
    group_size_ptr,
    num_groups: tl.constexpr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    strideBN: tl.constexpr,
    strideBK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
):
    total_cores = tl.num_programs(axis=0)
    core_idx = tl.program_id(axis=0)
    num_block_n = tl.cdiv(N, BLOCK_N)
    last_count = 0
    group_start = 0
    group_end = 0
    # group_size_m = tl.load(group_size_ptr + tl.arange(0, num_groups)).to(tl.int32)
    for group_idx in range(num_groups):
        # m = tl.extract_slice(group_size_m, [group_idx], [1], [1])
        m = tl.load(group_size_ptr + group_idx).to(tl.int32)
        group_end = group_start + m
        num_block_m = tl.cdiv(m, BLOCK_M)
        cur_count = last_count + num_block_m * num_block_n
        cur_block = core_idx if core_idx >= last_count else (core_idx + total_cores)
        while cur_block < cur_count:
            task_m_idx, task_n_idx = grouped_lanuch_diagonal(cur_block-last_count, num_block_m, num_block_n, BLOCK_TRESHHOLD)
            # matmul begin
            offs_am = (group_start + task_m_idx * BLOCK_M) + tl.arange(0, BLOCK_M)
            offs_bn = (task_n_idx * BLOCK_N) + tl.arange(0, BLOCK_N)
            offs_ak = tl.arange(0, BLOCK_K)
            offs_bk = (group_idx * K) + tl.arange(0, BLOCK_K)
            a_ptrs_base = A + (offs_am[:, None]*K + offs_ak[None, :])
            b_ptrs_base = B + (offs_bk[:, None]*strideBK + offs_bn[None, :]) * strideBN
            msk_m = offs_am < group_end
            msk_n = offs_bn < N
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
                a_ptrs = a_ptrs_base + k * BLOCK_K
                b_ptrs = b_ptrs_base + k * BLOCK_K * strideBK
                a = tl.load(
                    a_ptrs,
                    mask=msk_m[:, None] and (offs_ak[None, :] < K - k * BLOCK_K),
                    other=0.0,
                )
                tl.compile_hint(a, "dot_pad_only_k")
                b = tl.load(
                    b_ptrs,
                    mask=(offs_bk[:, None] < (group_idx * K + K - k * BLOCK_K)) and msk_n[None, :],
                    other=0.0,
                )
                tl.compile_hint(b, "dot_pad_only_k")
                # mma
                accumulator = tl.dot(a, b, acc=accumulator)
        
            c = accumulator.to(C.dtype.element_ty)
            offs_cm = (group_start + task_m_idx * BLOCK_M) + tl.arange(0, BLOCK_M)
            offs_cn = (task_n_idx * BLOCK_N) + tl.arange(0, BLOCK_N)
            c_ptrs = C + N * offs_cm[:, None] + offs_cn[None, :]
            c_mask = (offs_cm[:, None] < group_end) and (offs_cn[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)
            # matmul end
            cur_block = cur_block + total_cores
        last_count = cur_count % total_cores
        group_start = group_end


@torch.library.custom_op("moe::m_grouped_gemm", mutates_args=())
def m_grouped_gemm(A: Tensor,
                     B: Tensor,
                     size_per_group: torch.Tensor,
                     trans_b: bool = False,
                     numSM: int = -1) -> Tensor:
    assert A.dim() == 2
    assert B.dim() == 3
    M, K = A.shape
    assert A.stride(-1) == 1, "Please make sure A is K-major"
    if trans_b:
        num_groups, N, BK = B.shape
        strideBN, strideBK = B.stride(1), B.stride(2)
    else:
        num_groups, BK, N = B.shape
        strideBK, strideBN = B.stride(1), B.stride(2)
    assert BK == K, "K of A should be equal to K of B"
    assert num_groups == size_per_group.numel()
    C = A.new_empty(M, N)
    num_cores = get_number_cores()
    m_grouped_gemm_kernel = m_grouped_gemm_bKmajor_kernel if trans_b else m_grouped_gemm_bNmajor_kernel
    m_grouped_gemm_kernel[(num_cores,)](
        A,
        B,
        C,
        size_per_group,
        num_groups,
        M,
        N,
        K,
        strideBN,
        strideBK,
    )
    return C