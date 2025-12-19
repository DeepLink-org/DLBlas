import torch
import torch_npu
import triton
import triton.language as tl
from torch import Tensor

NUM_CORES=24

@triton.jit
def rms_norm_block_kernel(
    input,
    weight,
    output,
    n_rows,
    input_row_stride: tl.constexpr,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    cols_offsets = tl.arange(0, N_COLS)
    w = tl.load(weight + cols_offsets)
    w = tl.expand_dims(w, 0)
    w = tl.broadcast_to(w, (BLOCK, N_COLS))
    NUM_BLOCKS = tl.cdiv(n_rows, BLOCK)
    for row_block_id in range(pid, NUM_BLOCKS, NUM_CORES):
        pos_offset = row_block_id * BLOCK + tl.arange(0, BLOCK)
        pos_mask = (pos_offset < n_rows)[:, None]
        base_offset = pos_offset[:, None] * input_row_stride + cols_offsets[None, :]
        x = tl.load(input + base_offset, mask=pos_mask)
        xf = x.to(tl.float32)
        var = tl.sum(xf * xf, 1) / N_COLS
        qrt = tl.expand_dims(tl.math.rsqrt(var + eps), 1)
        out = xf * tl.broadcast_to(qrt, (BLOCK, N_COLS))
        out = w * out.to(x.dtype)
        tl.store(output + base_offset, out, mask=pos_mask)


def rms_norm_block_triton(
    hidden_states: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
):
    assert hidden_states.is_contiguous()
    feat_size = weight.shape[0]
    seq_len = hidden_states.numel() // hidden_states.size(-1)
    input_stride = hidden_states.stride(-2)
    out = torch.empty_like(hidden_states)
    rms_norm_block_kernel[(NUM_CORES,)](
        hidden_states,
        weight,
        out,
        n_rows=seq_len,
        input_row_stride=input_stride,
        eps=eps,
        N_COLS=feat_size,
        BLOCK=16,
        NUM_CORES=NUM_CORES,
    )
    return out


@triton.jit
def matmul_kernel(
        mat_a, mat_b, mat_c,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        num_cores: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_TRESHHOLD: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    task_m_idx = 0
    task_n_idx = 0
    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    #当任务量较多时，可以使能对角线分核策略进行优化
    for block_idx in range (
        pid, NUM_BLOCKS, num_cores
    ):
        #8 * 8 对角线分核代码实现 
        curThresholdM = BLOCK_TRESHHOLD if block_idx < (NUM_BLOCKS_M // BLOCK_TRESHHOLD * BLOCK_TRESHHOLD) * NUM_BLOCKS_N else NUM_BLOCKS_M % BLOCK_TRESHHOLD
        curThresholdM_thresholdN = curThresholdM * BLOCK_TRESHHOLD
        curThresholdN = BLOCK_TRESHHOLD if block_idx % (NUM_BLOCKS_N * BLOCK_TRESHHOLD) < (curThresholdM * NUM_BLOCKS_N) // curThresholdM_thresholdN * curThresholdM_thresholdN else NUM_BLOCKS_N % BLOCK_TRESHHOLD
        localRelativeBlock = block_idx % (BLOCK_TRESHHOLD * NUM_BLOCKS_N) % (BLOCK_TRESHHOLD * curThresholdM)
        task_m_idx = localRelativeBlock % curThresholdM + block_idx // (BLOCK_TRESHHOLD * NUM_BLOCKS_N) * BLOCK_TRESHHOLD
        #求最小公倍数，方便求基本块的坐标
        x, y = curThresholdM, curThresholdN if curThresholdM > curThresholdN else curThresholdN, curThresholdM
        while y != 0:
            x, y = y, x % y
        lcm = curThresholdM * curThresholdN // x
        task_n_idx = (localRelativeBlock + (localRelativeBlock // lcm)) % curThresholdN + block_idx % (BLOCK_TRESHHOLD * NUM_BLOCKS_N) // curThresholdM_thresholdN * BLOCK_TRESHHOLD
        
        m_start = task_m_idx * BLOCK_M
        n_start = task_n_idx * BLOCK_N
        
        mat_c_block = tl.zeros((BLOCK_M, BLOCK_N),dtype = tl.float32)
        for k_start in range(0, K, BLOCK_K):
            mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (
                k_start + tl.arange(0, BLOCK_K)
            )[None, :]
            mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                (k_start + tl.arange(0, BLOCK_K)) < K
            )[None, :]
            mat_a_block = tl.load(mat_a + mat_a_offset, mask = mat_a_mask, other = 0.0)
            # tl.compile_hint(mat_a_block, "dot_pad_only_k")
            mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + ( 
                n_start + tl.arange(0, BLOCK_N)
            )[None, :]
            mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & (
                (n_start + tl.arange(0, BLOCK_N)) < N
            )[None, :]
            mat_b_block = tl.load(mat_b + mat_b_offset, mask = mat_b_mask, other = 0.0)
            # tl.compile_hint(mat_b_block, "dot_pad_only_k")
            mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
        mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (
            n_start + tl.arange(0, BLOCK_N)
        )[None, :]
        mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
            (n_start + tl.arange(0, BLOCK_N)) < N
        )[None, :]
        tl.store(mat_c + mat_c_offset, mat_c_block.to(tl.bfloat16), mask = mat_c_mask)
   

def triton_matmul(
    mat_a,
    mat_b,
):
    m = mat_a.shape[0]
    k = mat_a.shape[1]
    n = mat_b.shape[1]
    mat_c = torch.empty(m, n, dtype=mat_a.dtype, device=mat_a.device)
    
    '''
    NPU芯片更加亲和512B对齐场景,如下分块通用性能较好,可以使用autotune选取最优
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256    
    '''

    matmul_kernel[(NUM_CORES,)] (
        mat_a,
        mat_b,
        mat_c,
        m,
        n,
        k,
        NUM_CORES,
        BLOCK_M = 128,
        BLOCK_N = 256,
        BLOCK_K = 256,
        BLOCK_TRESHHOLD = 6,
    )
    return mat_c


@triton.jit
def fused_matmul_rmsnorm_kernel(
    # for matmul
        mat_a, mat_b, mat_c,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_TRESHHOLD: tl.constexpr,
        # for rmsnorm
        input,
        weight,
        output,
        n_rows,
        input_row_stride: tl.constexpr,
        eps: tl.constexpr,
        N_COLS: tl.constexpr,
        BLOCK: tl.constexpr,
        NUM_CORES: tl.constexpr,
):
    matmul_kernel(
        mat_a = mat_a,
        mat_b = mat_b, 
        mat_c = mat_c,
        M = M,
        N = N,
        K = K,
        num_cores=NUM_CORES,
        BLOCK_M = BLOCK_M,
        BLOCK_N = BLOCK_N,
        BLOCK_K = BLOCK_K,
        BLOCK_TRESHHOLD = BLOCK_TRESHHOLD,
    )
    rms_norm_block_kernel(
        input = input,
        weight = weight,
        output = output,
        n_rows = n_rows,
        input_row_stride = input_row_stride,
        eps = eps,
        N_COLS = N_COLS,
        BLOCK = BLOCK,
        NUM_CORES = NUM_CORES,
    )


def fused_matmul_rmsnorm_triton(
    mat_a,
    mat_b,
    hidden_states: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
):
    m = mat_a.shape[0]
    k = mat_a.shape[1]
    n = mat_b.shape[1]
    mat_c = torch.empty(m, n, dtype=mat_a.dtype, device=mat_a.device)
    # 
    assert hidden_states.is_contiguous()
    feat_size = weight.shape[0]
    seq_len = hidden_states.numel() // hidden_states.size(-1)
    input_stride = hidden_states.stride(-2)
    out_norm = torch.empty_like(hidden_states)
    fused_matmul_rmsnorm_kernel[(NUM_CORES,)] (
        # matmul
        mat_a = mat_a,
        mat_b = mat_b, 
        mat_c = mat_c,
        M = m,
        N = n,
        K = k,
        BLOCK_M = 128,
        BLOCK_N = 256,
        BLOCK_K = 256,
        BLOCK_TRESHHOLD = 6,
        # rmsnorm
        input=hidden_states,
        weight=weight,
        output=out_norm,
        n_rows=seq_len,
        input_row_stride=input_stride,
        eps=eps,
        N_COLS = feat_size,
        BLOCK = 16,
        NUM_CORES = NUM_CORES,
    )
    return mat_c, out_norm

def rms_norm_ref(input: torch.Tensor, weight: torch.Tensor, eps: float):
    return torch.rms_norm(input, normalized_shape=weight.shape, weight=weight, eps=eps)



dtype_ = torch.float16
b = 128
seq_len = 4096
kv_lora_rank = 512  
norm_input = torch.randn(
    (b, seq_len, 1, kv_lora_rank), dtype=dtype_, device='npu'
)
rmsnormGammaCkv = torch.randn((kv_lora_rank), dtype=dtype_, device='npu')
norm_out_ref = rms_norm_ref(norm_input, rmsnormGammaCkv, eps=1e-06)
def test_rms_norm():
    kv_cache_triton = rms_norm_block_triton(norm_input, rmsnormGammaCkv, eps=1e-06)
    torch.testing.assert_close(norm_out_ref, kv_cache_triton, rtol=1e-02, atol=1e-02)
   
M = 2048
K = 7168
N = 16384
mat_a = torch.randn([M, K], dtype = torch.bfloat16, device = "npu")
mat_b = torch.randn([K, N], dtype = torch.bfloat16, device = "npu")
golden = torch.matmul(mat_a, mat_b)
def test_matmul():
    result = triton_matmul(mat_a, mat_b)   
    mask = golden.abs() < 1.0
    tmpatol = tmprtol = 2 ** -6
    try:
        torch.testing.assert_close(result[mask], golden[mask], atol = tmpatol, rtol = 0)
        torch.testing.assert_close(result[~mask], golden[~mask], atol = 0, rtol = tmprtol)
        print("run matmul success")
    except:
        print(f"[ERROR] M={M} ,K={K}, N={N}存在精度问题")

def test_fused_matmul_rmsnorm():
    mat_c_triton, norm_out_triton = fused_matmul_rmsnorm_triton(mat_a,mat_b, norm_input, rmsnormGammaCkv, eps=1e-6)
    torch.testing.assert_close(norm_out_ref, norm_out_triton, rtol=1e-02, atol=1e-02)
    mask = golden.abs() < 1.0
    tmpatol = tmprtol = 2 ** -6
    try:
        torch.testing.assert_close(mat_c_triton[mask], golden[mask], atol = tmpatol, rtol = 0)
        torch.testing.assert_close(mat_c_triton[~mask], golden[~mask], atol = 0, rtol = tmprtol)
        print("run matmul success")
    except:
        print(f"[ERROR] M={M} ,K={K}, N={N}存在精度问题")

def profile():
    stream = torch.npu.current_stream()
    experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
            data_simplification=False
        )
    LOOP=20
    data_path = "./prof_data"
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.NPU
            ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=LOOP, repeat=1, skip_first=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(data_path),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config) as prof:
       
        stream.synchronize()
        prof.step()
        for i in range(LOOP+2):
            kv_cache_triton = rms_norm_block_triton(norm_input, rmsnormGammaCkv, eps=1e-06)
            result = triton_matmul(mat_a, mat_b)   
            mat_c_triton, norm_out_triton = fused_matmul_rmsnorm_triton(mat_a,mat_b, norm_input, rmsnormGammaCkv, eps=1e-6)
            prof.step()
            if i == 0:
                stream.synchronize()
        stream.synchronize()

if __name__ == "__main__":
    test_rms_norm()
    test_matmul()
    test_fused_matmul_rmsnorm()
    profile()