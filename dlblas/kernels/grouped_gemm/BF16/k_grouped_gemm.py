import torch
from torch import Tensor

import triton
import triton.language as tl

def get_cuda_autotune_config():
    return [
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, "GROUP_M": 12}, num_stages=3, num_warps=8),
            # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, "GROUP_M": 12}, num_stages=5, num_warps=8),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 12}, num_stages=3, num_warps=8),
            # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 12}, num_stages=5, num_warps=8),
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, "GROUP_M": 6}, num_stages=3, num_warps=8),
            # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, "GROUP_M": 6}, num_stages=5, num_warps=8),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 6}, num_stages=3, num_warps=8),
            # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 6}, num_stages=5, num_warps=8),
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, "GROUP_M": 10}, num_stages=3, num_warps=8),
            # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, "GROUP_M": 10}, num_stages=5, num_warps=8),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 10}, num_stages=3, num_warps=8),
            # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 10}, num_stages=5, num_warps=8),
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, "GROUP_M": 14}, num_stages=3, num_warps=8),
            # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, "GROUP_M": 14}, num_stages=5, num_warps=8),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 14}, num_stages=3, num_warps=8),
            # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, "GROUP_M": 14}, num_stages=5, num_warps=8),
            ]


@triton.jit
def grouped_launch(pid,
                m, n,
                block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr):
    
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)
    remian_pid = pid - group_id * width
    pid_m = group_id * group_m + (remian_pid % group_size)

    pid_n = (pid % width) // group_size

    return pid_m, pid_n


@triton.autotune(configs=get_cuda_autotune_config(), key=['M', 'N'])
@triton.jit
def k_grouped_gemm_general_kernel(
    A, B, C,
    group_starts,
    group_ends,
    num_groups: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K,
    dtype_a: tl.constexpr,
    dtype_b: tl.constexpr, 
    dtype_c: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    
    dtypeA = tl.bfloat16 if dtype_a == 0 else tl.float16
    dtypeB = tl.bfloat16 if dtype_b == 0 else tl.float16
    dtypeC = tl.bfloat16 if dtype_c == 0 else tl.float16

    BLOCKS = tl.num_programs(axis=0)
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n * num_groups

    for tile_id in tl.range(start_pid, num_tiles, BLOCKS):
        
        group = tile_id // (num_pid_m * num_pid_n)
        group_start = tl.load(group_starts + group).to(tl.int32)
        group_end = tl.load(group_ends + group).to(tl.int32)

        id_tmp = tile_id % (num_pid_m * num_pid_n)

        if GROUP_M == 1:
            num_pid_m = tl.cdiv(M, BLOCK_M)
            pid_m = id_tmp % num_pid_m
            pid_n = id_tmp // num_pid_m
        else:
            pid_m, pid_n = grouped_launch(id_tmp, M, N, BLOCK_M, BLOCK_N, GROUP_M)

        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N
        
        tokens = group_end - group_start

        num_pid_k = tl.cdiv(tokens, BLOCK_K)
        offs_k = group_start + tl.arange(0, BLOCK_K)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for kk in range(0, num_pid_k-1):
            a_ptrs = A + ((offs_am + tl.arange(0, BLOCK_M))[None, :] + offs_k[:, None].to(tl.int64) * M)
            b_ptrs = B + ((offs_bn + tl.arange(0, BLOCK_N))[None, :] + offs_k[:, None].to(tl.int64) * N)
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator = tl.dot(a.T, b, acc=accumulator, input_precision = "tf32x3")
            offs_k += BLOCK_K
        
        if tokens > 0:
            offs_k_final = group_start + (num_pid_k - 1) * BLOCK_K + tl.arange(0, BLOCK_K)
            a_ptrs = A + ((offs_am + tl.arange(0, BLOCK_M))[None, :] + offs_k_final[:, None].to(tl.int64) * M)
            b_ptrs = B + ((offs_bn + tl.arange(0, BLOCK_N))[None, :] + offs_k_final[:, None].to(tl.int64) * N)
            maskA = (offs_k_final[:, None] < group_end)
            maskB = (offs_k_final[:, None] < group_end)
            a = tl.load(a_ptrs, mask=maskA, other=0.0)
            b = tl.load(b_ptrs, mask=maskB, other=0.0)
            accumulator = tl.dot(a.T, b, acc=accumulator, input_precision = "tf32x3")    
            c = accumulator.to(dtypeC)
            off_row = offs_am + group * M + tl.arange(0, BLOCK_M)
            off_col = offs_bn + tl.arange(0, BLOCK_N)
            c_ptrs = C + N * off_row[:, None] + off_col[None, :]
            c_mask = (off_row[:, None] < (group+1) * M) & (off_col[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)
            

def k_grouped_gemm_general(A: Tensor,
                     B: Tensor,
                     size_per_group: torch.Tensor) -> Tensor:
    assert A.dim() == 2
    assert B.dim() == 2

    K, M = A.shape
    K_, N = B.shape

    assert A.stride(-1) == 1, "Please make sure A is K-major"
    assert B.stride(-1) == 1, "Please make sure B is K-major"
    assert K == K_, "Please make sure that A and B have the same seqlen"
    # assert K * A.element_size() % 128 == 0, "A and B should be 128-byte aligned"
    num_groups = size_per_group.shape[0]

    C = A.new_empty(num_groups, M, N)
    group_end = size_per_group.cumsum(0) - size_per_group + size_per_group
    group_start = size_per_group.cumsum(0) - size_per_group

    dtype_mapping = {
        torch.bfloat16: 0,
        torch.float16: 1
    }
    dtype_a = dtype_mapping.get(A.dtype, -1)
    dtype_b = dtype_mapping.get(B.dtype, -1)
    dtype_c = dtype_mapping.get(C.dtype, -1)

    assert dtype_a >= 0, f"data type {A.dtype} not supported" 
    assert dtype_b >= 0, f"data type {B.dtype} not supported" 
    assert dtype_c >= 0, f"data type {C.dtype} not supported" 

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    
    def grid(META):
        assert M % META["BLOCK_M"] == 0, "Only support when M is a multiple of BLOCK_M"

        return (NUM_SMS, )

    k_grouped_gemm_general_kernel[grid](
            A, B, C,
            group_start,
            group_end,
            num_groups,
            M,
            N,
            K,
            dtype_a, dtype_b, dtype_c,
        )
    # print(f"best config {k_grouped_gemm_general_kernel.best_config}", flush = True)
    return C


@triton.autotune(configs=get_cuda_autotune_config(), key=['M', 'N'])
@triton.jit
def k_grouped_gemm_TMA_kernel(
    a_desc_ptr, b_desc_ptr, c_desc_ptr,
    A, B, C,
    group_starts,
    group_ends,
    num_groups: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K,
    dtype_a: tl.constexpr,
    dtype_b: tl.constexpr, 
    dtype_c: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    
    dtypeA = tl.bfloat16 if dtype_a == 0 else tl.float16
    dtypeB = tl.bfloat16 if dtype_b == 0 else tl.float16
    dtypeC = tl.bfloat16 if dtype_c == 0 else tl.float16

    BLOCKS = tl.num_programs(axis=0)
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n * num_groups

    for tile_id in tl.range(start_pid, num_tiles, BLOCKS):
        
        group = tile_id // (num_pid_m * num_pid_n)
        group_start = tl.load(group_starts + group).to(tl.int32)
        group_end = tl.load(group_ends + group).to(tl.int32)

        id_tmp = tile_id % (num_pid_m * num_pid_n)

        if GROUP_M == 1:
            num_pid_m = tl.cdiv(M, BLOCK_M)
            pid_m = id_tmp % num_pid_m
            pid_n = id_tmp // num_pid_m
        else:
            pid_m, pid_n = grouped_launch(id_tmp, M, N, BLOCK_M, BLOCK_N, GROUP_M)

        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N
        
        tokens = group_end - group_start

        num_pid_k = tl.cdiv(tokens, BLOCK_K)
        offs_k = group_start

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for kk in range(0, num_pid_k-1):
            a = tl._experimental_descriptor_load(a_desc_ptr, [offs_k, offs_am], [BLOCK_K, BLOCK_M], dtypeA)
            b = tl._experimental_descriptor_load(b_desc_ptr, [offs_k, offs_bn], [BLOCK_K, BLOCK_N], dtypeB)
            accumulator = tl.dot(a.T, b, acc=accumulator, input_precision = "tf32x3")
            offs_k += BLOCK_K
        if tokens > 0:
            offs_k_final = group_start + (num_pid_k - 1) * BLOCK_K + tl.arange(0, BLOCK_K)
            a_ptrs = A + ((offs_am + tl.arange(0, BLOCK_M))[None, :] + offs_k_final[:, None].to(tl.int64) * M)
            b_ptrs = B + ((offs_bn + tl.arange(0, BLOCK_N))[None, :] + offs_k_final[:, None].to(tl.int64) * N)
            maskA = (offs_k_final[:, None] < group_end)
            maskB = (offs_k_final[:, None] < group_end)
            a = tl.load(a_ptrs, mask=maskA, other=0.0)
            b = tl.load(b_ptrs, mask=maskB, other=0.0)
            accumulator = tl.dot(a.T, b, acc=accumulator, input_precision = "tf32x3")

            c = accumulator.to(dtypeC)
            off_row = offs_am + group * M
            off_col = offs_bn
            tl._experimental_descriptor_store(c_desc_ptr, c, [off_row, off_col])


def k_grouped_gemm_TMA(A: Tensor,
                     B: Tensor,
                     size_per_group: torch.Tensor) -> Tensor:
    assert A.dim() == 2
    assert B.dim() == 2

    K, M = A.shape
    K_, N = B.shape

    assert A.stride(-1) == 1, "Please make sure A is K-major"
    assert B.stride(-1) == 1, "Please make sure B is K-major"
    assert K == K_, "Please make sure that A and B have the same seqlen"
    # assert K * A.element_size() % 128 == 0, "A and B should be 128-byte aligned"
    num_groups = size_per_group.shape[0]

    C = A.new_empty(num_groups, M, N)
    group_end = size_per_group.cumsum(0) - size_per_group + size_per_group
    group_start = size_per_group.cumsum(0) - size_per_group

    dtype_mapping = {
        torch.bfloat16: 0,
        torch.float16: 1
    }
    dtype_a = dtype_mapping.get(A.dtype, -1)
    dtype_b = dtype_mapping.get(B.dtype, -1)
    dtype_c = dtype_mapping.get(C.dtype, -1)

    assert dtype_a >= 0, f"data type {A.dtype} not supported" 
    assert dtype_b >= 0, f"data type {B.dtype} not supported" 
    assert dtype_c >= 0, f"data type {C.dtype} not supported" 

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    from .utils import TmaAutoTuneHelper
    desc_helper = TmaAutoTuneHelper()
    desc_helper.init_tma_descriptor("a")
    desc_helper.init_tma_descriptor("b")
    desc_helper.init_tma_descriptor("c")

    def grid(META):
        assert (N * B.element_size()) % 16 == 0, "TMA required 16-byte alignment"
        assert M % META["BLOCK_M"] == 0, "Only support when M is a multiple of BLOCK_M"
        nonlocal desc_helper
        desc_helper.fill_2d_tma_descriptor(
            "a",
            A.data_ptr(),
            K,
            M,
            META["BLOCK_K"],
            META["BLOCK_M"],
            A.element_size(),
        )

        desc_helper.fill_2d_tma_descriptor(
            "b",
            B.data_ptr(),
            K,
            N,
            META["BLOCK_K"],
            META["BLOCK_N"],
            B.element_size(),
        )
        desc_helper.fill_2d_tma_descriptor(
            "c",
            C.data_ptr(),
            C.shape[0] * C.shape[1],
            C.shape[2],
            META["BLOCK_M"],
            META["BLOCK_N"],
            C.element_size(),
        )

        return (NUM_SMS, )

    desc_a = desc_helper.get_tma_descriptor_kernel_param("a")
    desc_b = desc_helper.get_tma_descriptor_kernel_param("b")
    desc_c = desc_helper.get_tma_descriptor_kernel_param("c")

    k_grouped_gemm_TMA_kernel[grid](
            desc_a, desc_b, desc_c,
            A, B, C,
            group_start,
            group_end,
            num_groups,
            M,
            N,
            K,
            dtype_a, dtype_b, dtype_c,
        )
    # print(f"best config {k_grouped_gemm_kernel.best_config}", flush = True)
    return C
import torch_npu
cuda_arch = torch_npu.npu.get_device_capability()
if cuda_arch is None:
    cuda_arch = (9, 0)
k_grouped_gemm = k_grouped_gemm_TMA if cuda_arch[0] >= 9 else k_grouped_gemm_general
# k_grouped_gemm = k_grouped_gemm_general
