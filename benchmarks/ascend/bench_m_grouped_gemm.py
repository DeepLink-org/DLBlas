import torch
import triton
from dlblas.utils.device_utils import infer_device
from dlblas.kernels.ascend.m_grouped_gemm import m_grouped_gemm
from tests.kernels.ascend.test_m_grouped_matmul import generate_random_list, torch_grouped_matmul

if __name__=='__main__':
    groups = 128
    z = groups
    trans_b = False; print(f"{trans_b = }")
    device = infer_device()
    batch_sizes = torch.Tensor(generate_random_list(groups, groups*2560)).to(device).to(torch.int64)
    M = batch_sizes.sum().item()
    for (n, k) in ((4096, 4096), (512, 512), (768*2, 2048), (2048, 768), (1536*2, 4096)): # (4096, 1536)
        a = torch.randn(M, k, dtype = torch.bfloat16, device = device).view(-1, k)
        b = torch.randn(z, n, k, dtype = torch.bfloat16, device = device) if trans_b else torch.randn(z, k, n, dtype = torch.bfloat16, device = device)
        print(f"M={M}, z={z}, k={k}, n={n}")
        golden = torch_grouped_matmul(a, b, batch_sizes, trans_b)
        result = m_grouped_gemm(a, b, batch_sizes, trans_b)
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
                plot_name='m_grouped_matmul-performance-' +
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
        print("run matmul success")