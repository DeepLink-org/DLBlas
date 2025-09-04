import torch
import triton
from dlblas.utils.device_utils import infer_device
from dlblas.kernels.ascend.grouped_gemm import k_grouped_gemm
from tests.kernels.ascend.test_grouped_matmul import generate_random_list, k_grouped_matmul_torch

if __name__=='__main__':
    groups = 8
    z = groups
    DEV = infer_device()
    dtype_ = torch.bfloat16
    batch_sizes = torch.Tensor(generate_random_list(groups, groups*2560)).to(DEV).to(torch.int64).abs()
    K = batch_sizes.sum().item()
    for (M, N) in ((4096, 4096), (512, 512), (768*2, 2048), (2048, 768), (1536*2, 4096)):
        a = torch.randn(K, M, dtype = dtype_, device = DEV)
        b = torch.randn(K, N, dtype = dtype_, device = DEV)
        golden = k_grouped_matmul_torch(a, b, batch_sizes.cpu())
        result = k_grouped_gemm(a, b, batch_sizes)
        mask = golden.abs() < 1.0
        tmpatol = tmprtol = 2 ** -6
        torch.testing.assert_close(result[mask], golden[mask], atol = tmpatol, rtol = 0)
        torch.testing.assert_close(result[~mask], golden[~mask], atol = 0, rtol = tmprtol)
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
                plot_name='k_grouped_matmul-performance-' +
                (f'bf16-[Batch={z} M={M} N={N} k={K}]'),  # Name for the plot, used also as a file name for saving the plot.
                args={},
            ))
        @triton.testing.perf_report(configs)
        def benchmark(cnt, provider):
            warmup = 500
            rep = 500
            quantiles = [0.5, 0.2, 0.8]
            if provider == 'torch':
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: k_grouped_matmul_torch(a, b, batch_sizes),
                                                            quantiles=quantiles,
                                                            warmup=warmup,
                                                            rep=rep)
            if provider == 'triton_gmm':
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: k_grouped_gemm(a, b, batch_sizes),
                                                            quantiles=quantiles,
                                                            warmup=warmup,
                                                            rep=rep)

            return ms, max_ms, min_ms

        benchmark.run(show_plots=False, print_data=True)
        print("run matmul success")