import torch
import triton
from dlblas.utils.device_utils import infer_device
from dlblas.kernels.ascend.matmul import call as triton_matmul


if __name__ == "__main__":
    M = 2048 * 64
    DEV = torch.device(infer_device())
    for (N, K) in ((4096, 4096), (512, 512), (768*2, 2048), (2048, 768), (1536*2, 4096), (4096, 1536), (7168, 18432)):
        mat_a = torch.randn([M, K], dtype = torch.bfloat16, device = DEV)
        mat_b = torch.randn([K, N], dtype = torch.bfloat16, device = DEV)
        # result = triton_matmul(mat_a, mat_b)
        # golden = torch.matmul(mat_a, mat_b)
        # mask = golden.abs() < 1.0
        # tmpatol = tmprtol = 2 ** -6
        # torch.testing.assert_close(result[mask], golden[mask], atol = tmpatol, rtol = 0)
        # torch.testing.assert_close(result[~mask], golden[~mask], atol = 0, rtol = tmprtol)
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
                plot_name='matmul-performance-' + f'bf16-[M={M} N={N} k={K}]',
                args={},
            ))
        @triton.testing.perf_report(configs)
        def benchmark(cnt, provider):
            warmup = 500
            rep = 500
            quantiles = [0.5, 0.2, 0.8]
            if provider == 'torch':
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(mat_a, mat_b),
                                                            quantiles=quantiles,
                                                            warmup=warmup,
                                                            rep=rep)
            if provider == 'triton':
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_matmul(mat_a, mat_b),
                                                            quantiles=quantiles,
                                                            warmup=warmup,
                                                            rep=rep)
            return ms, max_ms, min_ms

        benchmark.run(show_plots=False, print_data=True)
        print("run matmul success")