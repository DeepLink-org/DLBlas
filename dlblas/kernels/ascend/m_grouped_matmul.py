
import torch
from torch import Tensor
import triton
import triton.language as tl
import triton.runtime.driver as driver

# get device properties of npu
def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)

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

@triton.jit
def grouped_lanuch_npu(pid, num_pid_m, num_pid_n, BLOCK_TRESHHOLD):
    if (num_pid_m >= BLOCK_TRESHHOLD) and (num_pid_n >= BLOCK_TRESHHOLD):
        # 对角线分核代码实现 
        curThresholdM = BLOCK_TRESHHOLD if pid < (num_pid_m // BLOCK_TRESHHOLD * BLOCK_TRESHHOLD) * num_pid_n else num_pid_m % BLOCK_TRESHHOLD
        curThresholdM_thresholdN = curThresholdM * BLOCK_TRESHHOLD
        curThresholdN = BLOCK_TRESHHOLD if pid % (num_pid_n * BLOCK_TRESHHOLD) < (curThresholdM * num_pid_n) // curThresholdM_thresholdN * curThresholdM_thresholdN else num_pid_n % BLOCK_TRESHHOLD
        localRelativeBlock = pid % (BLOCK_TRESHHOLD * num_pid_n) % (BLOCK_TRESHHOLD * curThresholdM)
        task_m_idx = localRelativeBlock % curThresholdM + pid // (BLOCK_TRESHHOLD * num_pid_n) * BLOCK_TRESHHOLD
        # 求最小公倍数，方便求基本块的坐标
        x, y = curThresholdM, curThresholdN if curThresholdM > curThresholdN else curThresholdN, curThresholdM
        while y != 0:
            x, y = y, x % y
        lcm = curThresholdM * curThresholdN // x
        task_n_idx = (localRelativeBlock + (localRelativeBlock // lcm)) % curThresholdN + pid % (BLOCK_TRESHHOLD * num_pid_n) // curThresholdM_thresholdN * BLOCK_TRESHHOLD
    else:
        task_m_idx = pid // num_pid_n
        task_n_idx = pid % num_pid_n
    return task_m_idx, task_n_idx


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
        pid_m, pid_n = grouped_lanuch_npu(tile_id, num_pid_m, num_pid_n, BLOCK_TRESHHOLD)
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
    # GROUP_M: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
):

    """gemm fp8 kernel."""
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
            task_m_idx, task_n_idx = grouped_lanuch_npu(cur_block-last_count, num_block_m, num_block_n, BLOCK_TRESHHOLD)
            # task_m_idx = (cur_block-last_count) // num_block_n
            # task_n_idx = (cur_block-last_count) % num_block_n
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



@triton.jit
def repeat_interleave_kernel(
    group_ptr,
    repeats_ptr,
    repeat_cum_ptr,
    output_ptr,
    # BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    repeat = tl.load(repeats_ptr + pid)
    start = tl.load(repeat_cum_ptr + pid) - repeat
    group = tl.load(group_ptr + pid)
    for r in range(repeat):
        tl.store(output_ptr + start + r, group)

def repeat_interleave(
    group_indices: Tensor, 
    repeats: Tensor, 
    repeat_cum: Tensor, 
    m_indices_pad: Tensor, 
) -> None:
    grid = lambda args: (len(repeats), )
    repeat_interleave_kernel[grid](group_indices, repeats, repeat_cum, m_indices_pad)
    return


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
    num_cores = 24 # get_npu_properties()["num_aicore"] # 24
    # print(f"num_cores={num_cores}")

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


def triton_gmm(A, B, batch_sizes, trans_b=False):
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
        C = A.new_empty(M, N)
        batch_sizes = batch_sizes.numpy()

       
        C = A.new_empty(M, N)
        from dlblas.kernels.ascend.matmul_v2 import triton_matmul as triton_matmul
        start = 0
        for i, size in enumerate(batch_sizes):
            rhs = B[i, :, :].t() if trans_b else B[i, :, :]
            triton_matmul(a[start:start + size, :], rhs, C[start:start + size, :])
            start += size
        return C


def torch_grouped_matmul(a, b, size_per_group, trans_b):
    assert trans_b == False
    return torch.ops.npu.npu_grouped_matmul(
        [a],
        [b],
        bias=None,
        group_list=size_per_group.cumsum(0),
        split_item=2,
        group_type=0,
        group_list_type=0,
    )[0]


@m_grouped_gemm.register_fake
def _(A: Tensor,
        B: Tensor,
        size_per_group: torch.Tensor,
        trans_b: bool = False,
        numSM: int = -1) -> Tensor:
    M, K = A.shape
    if trans_b:
        num_groups, N, BK = B.shape
    else:
        num_groups, BK, N = B.shape
    C = A.new_empty(M, N)
    return C


def generate_random_list(length, total_sum):
    # 生成一个长度为length的列表，元素之和为total_sum
    # 先生成一个平均分配的列表
    avg = total_sum // length
    lst = [0] * length
    # 随机调整数值，确保总和不变
    for i in range(length):
        # 随机选择两个不同的位置
        lst[i] = random.randint(0, 2*int(avg))
    ratio = total_sum / sum(lst)
    lst = [int(x * ratio) for x in lst]

    diff = total_sum - sum(lst)
    lst[-1] += diff
    return lst

def row_max_normalization(tensor):
    row_maxs = tensor.abs().max(dim = -1).values + 1e-9
    tensor_normalized = tensor / row_maxs.unsqueeze(dim = -1)
    return tensor_normalized

if __name__=='__main__':
    from typing import Tuple
    import random

    def gmm(a, b, batch_sizes, trans_b=False):
        batch_sizes = batch_sizes.numpy()

        out = []
        start = 0
        for i, size in enumerate(batch_sizes):
            rhs = b[i, :, :].t() if trans_b else b[i, :, :]
            out.append(a[start:start + size, :] @ rhs)
            start += size
        return torch.cat(out)
    

    groups = 128
    z = groups
    trans_b = False; print(f"{trans_b = }")
    device = 'npu'
    batch_sizes = torch.Tensor(generate_random_list(groups, groups*5120)).to(device).to(torch.int64)
    batch_sizes_cpu = batch_sizes.cpu()
    M = batch_sizes.sum().item()

    for (n, k) in ((4096, 4096), (512, 512), (768*2, 2048), (2048, 768), (1536*2, 4096), (4096, 1536)):
        a = torch.randn(M, k, dtype = torch.bfloat16, device = device).view(-1, k)
        b = torch.randn(z, n, k, dtype = torch.bfloat16, device = device) if trans_b else torch.randn(z, k, n, dtype = torch.bfloat16, device = device)
        print(f"M={M}, z={z}, k={k}, n={n}")
        # golden = gmm(a, b, batch_sizes.cpu(), trans_b)
        golden = torch_grouped_matmul(a, b, batch_sizes, trans_b)
        # result = triton_gmm(a, b, batch_sizes.cpu(), trans_b)
        result = m_grouped_gemm(a, b, batch_sizes, trans_b)
     
        configs = []
        configs.append(
            triton.testing.Benchmark(
                x_names=['cnt'],  # Argument names to use as an x-axis for the plot
                # x_vals=[128 * i for i in range(10, 15)],  # Different possible values for `x_name`
                x_vals=[1],  # NOTE: the tunning framework specialized to one shape
                line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
                # Possible values for `line_arg`
                # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
                line_vals=['triton_gmm' , 'torch'] ,  # Label name for the lines
                line_names=['Triton_gmm', 'Torch'] ,  # Line styles
                styles=[('green', '-'), ('blue', '-')],
                ylabel='TFLOPS',  # Label name for the y-axis
                plot_name='matmul-performance-' +
                (f'bf16-[Batch={z} M={M} N={n} k={k}]'),  # Name for the plot, used also as a file name for saving the plot.
                args={},
            ))
        @triton.testing.perf_report(configs)
        def benchmark(cnt, provider):
            warmup = 500
            rep = 500
            quantiles = [0.5, 0.2, 0.8]
            if provider == 'torch':
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_grouped_matmul(a, b, batch_sizes, trans_b),
                                                            quantiles=quantiles,
                                                            warmup=warmup,
                                                            rep=rep)
            if provider == 'triton_gmm':
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: m_grouped_gemm(a, b, batch_sizes, trans_b),
                                                            quantiles=quantiles,
                                                            warmup=warmup,
                                                            rep=rep)

            return ms, max_ms, min_ms

        benchmark.run(show_plots=False, print_data=True)

        mask = golden.abs() < 1.0
        tmpatol = tmprtol = 2 ** -6
        torch.testing.assert_close(result[mask], golden[mask], atol = tmpatol, rtol = 0)
        torch.testing.assert_close(result[~mask], golden[~mask], atol = 0, rtol = tmprtol)
        print("run matmul success")


        # out_triton = row_max_normalization(out_triton)
        # out_ref = row_max_normalization(out_ref)
        # torch.testing.assert_close(out_triton, out_ref, rtol = 1e-02, atol = 1e-02)
        # print(f"n={n}, k={k}, gmm success")
