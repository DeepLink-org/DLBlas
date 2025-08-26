import torch
import triton
from dlblas.utils.device_utils import infer_device
from dlblas.kernels.ascend.matmul_v2 import triton_matmul as triton_matmul


if __name__ == "__main__":
    # M = 2048
    # K = 7168
    # N = 16384
    M, K, N = 8246, 2048, 768
    DEV = torch.device(infer_device())
    mat_a = torch.randn([M, K], dtype = torch.bfloat16, device = DEV)
    mat_b = torch.randn([K, N], dtype = torch.bfloat16, device = DEV)
    
    result = triton_matmul(mat_a, mat_b)
    # print(result)
    golden = torch.matmul(mat_a, mat_b)
    
    mask = golden.abs() < 1.0
    tmpatol = tmprtol = 2 ** -6
    # try:
    torch.testing.assert_close(result[mask], golden[mask], atol = tmpatol, rtol = 0)
    torch.testing.assert_close(result[~mask], golden[~mask], atol = 0, rtol = tmprtol)
    # excep
        # print(f"[ERROR] M={M} ,K={K}, N={N}存在精度问题")
    
    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=['cnt'],  # Argument names to use as an x-axis for the plot
            # x_vals=[128 * i for i in range(10, 15)],  # Different possible values for `x_name`
            x_vals=[1],  # NOTE: the tunning framework specialized to one shape
            line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=['triton', 'torch'] ,  # Label name for the lines
            line_names=['Triton', 'Torch'] ,  # Line styles
            styles=[('green', '-'), ('blue', '-')],
            ylabel='TFLOPS',  # Label name for the y-axis
            plot_name='matmul-performance-' +
            ('bf16'),  # Name for the plot, used also as a file name for saving the plot.
            args={
               
                'M': 2048,
                'N': 16384,
                'K': 7168,
            },
        ))

    @triton.testing.perf_report(configs)
    def benchmark(cnt, M, N, K, provider):
        warmup = 500
        rep = 500
        a = torch.randn((M, K), device=DEV, dtype=torch.bfloat16)
        b = torch.randn((K, N), device=DEV, dtype=torch.bfloat16)
        
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b),
                                                         quantiles=quantiles,
                                                         warmup=warmup,
                                                         rep=rep)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_matmul(a, b),
                                                         quantiles=quantiles,
                                                         warmup=warmup,
                                                         rep=rep)

        return ms, max_ms, min_ms

    benchmark.run(show_plots=False, print_data=True)
    print("run matmul success")