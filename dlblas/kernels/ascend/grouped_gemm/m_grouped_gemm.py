import torch
from torch import Tensor
import triton
import triton.language as tl
import triton.language.extra.deeplink as dl
from dlblas.utils.op_helper import grouped_launch_diagonal
from dlblas.utils.device_utils import get_number_cores


def get_autotune_config():
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 4}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 5}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 6}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 7}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 9}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 4}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 5}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 6}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 7}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 8}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_TRESHHOLD": 9}),
    ]

@triton.autotune(configs=get_autotune_config(), key=['N', 'K'])
@triton.jit
def m_grouped_gemm_bKmajor_kernel(
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
    # should use tl.static_range on NV
    for group_idx in range(num_groups):
        # m = tl.extract_slice(group_size_m, [group_idx], [1], [1])
        m = tl.load(group_size_ptr + group_idx).to(tl.int32)
        group_end = group_start + m
        num_block_m = tl.cdiv(m, BLOCK_M)
        cur_count = last_count + num_block_m * num_block_n
        cur_block = core_idx if core_idx >= last_count else (core_idx + total_cores)
        while cur_block < cur_count:
            task_m_idx, task_n_idx = grouped_launch_diagonal(cur_block-last_count, num_block_m, num_block_n, BLOCK_TRESHHOLD)
            # matmul begin
            offs_am = (group_start + task_m_idx * BLOCK_M) + tl.arange(0, BLOCK_M)
            offs_bn = (group_idx * N + task_n_idx * BLOCK_N) + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            a_ptrs_base = A + (offs_am[:, None]*K + offs_k[None, :])
            b_ptrs_base = B + (offs_bn[:, None]*strideBN + offs_k[None, :])
            msk_m = offs_am < group_end
            msk_n = offs_bn < (group_idx + 1) * N
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
                a_ptrs = a_ptrs_base + k * BLOCK_K
                b_ptrs = b_ptrs_base + k * BLOCK_K
                a = tl.load(
                    a_ptrs,
                    mask=msk_m[:, None] and (offs_k[None, :] < K - k * BLOCK_K),
                    other=0.0,
                )
                dl.compile_hint(a, "dot_pad_only_k")
                b = tl.load(
                    b_ptrs,
                    mask=msk_n[:, None] and (offs_k[None, :] < (K - k * BLOCK_K)),
                    other=0.0,
                )
                b = tl.trans(b)
                dl.compile_hint(b, "dot_pad_only_k")
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
    group_idx = 0
    # group_size_m = tl.load(group_size_ptr + tl.arange(0, num_groups)).to(tl.int32)
    # should use tl.static_range on NV
    for group_idx in range(num_groups):
        # m = tl.extract_slice(group_size_m, [group_idx], [1], [1])
        m = tl.load(group_size_ptr + group_idx).to(tl.int32)
        group_end = group_start + m
        num_block_m = tl.cdiv(m, BLOCK_M)
        cur_count = last_count + num_block_m * num_block_n
        cur_block = core_idx if core_idx >= last_count else (core_idx + total_cores)
        while cur_block < cur_count:
            task_m_idx, task_n_idx = grouped_launch_diagonal(cur_block-last_count, num_block_m, num_block_n, BLOCK_TRESHHOLD)
            # matmul begin
            offs_am = (group_start + task_m_idx * BLOCK_M) + tl.arange(0, BLOCK_M)
            offs_bn = (task_n_idx * BLOCK_N) + tl.arange(0, BLOCK_N)
            offs_ak = tl.arange(0, BLOCK_K)
            offs_bk = (group_idx * K) + tl.arange(0, BLOCK_K)
            a_ptrs_base = A + (offs_am[:, None]*K + offs_ak[None, :])
            b_ptrs_base = B + (offs_bk[:, None]*strideBK + offs_bn[None, :])
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
                dl.compile_hint(a, "dot_pad_only_k")
                b = tl.load(
                    b_ptrs,
                    mask=(offs_bk[:, None] < (group_idx * K + K - k * BLOCK_K)) and msk_n[None, :],
                    other=0.0,
                )
                dl.compile_hint(b, "dot_pad_only_k")
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


def m_grouped_gemm(A: Tensor, B: Tensor, size_per_group: torch.Tensor, trans_b: bool = False) -> Tensor:
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
    # print(f"m_grouped_gemm_kernel best config {m_grouped_gemm_kernel.best_config}", flush = True)
    return C