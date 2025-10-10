import torch
import triton
from dlblas.utils.device_utils import infer_device
from dlblas.kernels.ascend.matmul_relu import matmul as triton_matmul

LEAKY_RELU_CUSTOM = "leaky_relu_custom"
def torch_matmul(a, b, activation=""):
    c = torch.matmul(a, b)
    if activation == LEAKY_RELU_CUSTOM:
        c = torch.where(c >= 0, c, 0.01 * c) + 1.0
    return c

def main():
    torch.manual_seed(0)
    DEV = torch.device(infer_device())
    activation = LEAKY_RELU_CUSTOM
    a = torch.randn((512, 512), device=DEV, dtype=torch.float16)
    b = torch.randn((512, 512), device=DEV, dtype=torch.float16)
    triton_output = triton_matmul(a, b, activation)
    torch_output = torch_matmul(a, b, activation)
    torch.testing.assert_close(triton_output, torch_output, rtol=0.01, atol=1e-03)


    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 6)],  # Different possible values for `x_name`
            line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=['triton_no_parallel','triton-parallel'],  # Label name for the lines
            line_names=['triton_no_parallel', 'triton-parallel'],  # Line styles
            styles=[('green', '-'), ('blue', '-')],
            ylabel='TFLOPS',  # Label name for the y-axis
            plot_name='matmul-performance-fp16',
            args={},
        ))

    @triton.testing.perf_report(configs)
    def benchmark(M, N, K, provider):
        warmup = 500
        rep = 500
        a = torch.randn((M, K), device=DEV, dtype=torch.float16)
        b = torch.randn((K, N), device=DEV, dtype=torch.float16)

        quantiles = [0.5, 0.2, 0.8]
        if provider == 'triton_no_parallel':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_matmul(a, b, activation, 1),
                                                         quantiles=quantiles,
                                                         warmup=warmup,
                                                         rep=rep)
        if provider == 'triton-parallel':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_matmul(a, b, activation, 2),
                                                         quantiles=quantiles,
                                                         warmup=warmup,
                                                         rep=rep)
        return ms, max_ms, min_ms

    benchmark.run(show_plots=False, print_data=True)


if __name__ == '__main__':
    main()