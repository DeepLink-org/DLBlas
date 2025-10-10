import torch
import triton
from dlblas.utils.device_utils import infer_device
from dlblas.kernels.layer_norm.layernorm_normal_loop import call as layernorm_normal_loop
from dlblas.kernels.layer_norm.layernorm_normal import call as layernorm_normal
from dlblas.kernels.layer_norm.layernorm_opt_2D import call as layernorm_opt_2D
from dlblas.kernels.layer_norm.layernorm_opt_mask_2D_tma import call as layernorm_opt_mask_2D_tma
from dlblas.kernels.layer_norm.layernorm_opt_mask import call as layernorm_opt_mask
from dlblas.kernels.layer_norm.layernorm_opt import call as layernorm_opt
from dlblas.kernels.layer_norm.layernorm_torch import call as layernorm_torch
from dlblas.kernels.layer_norm import call as layernorm_dlblas

device = infer_device()

def bench_layernorm(dlblas_op, plot_name):
    @triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['hidden_size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[256, 512, 1024, 2048, 4096, 8192, 16 * 1024, 17*1024, 18*1024, 24*1024, 32* 1024, 48*1024, 64 * 1024], 
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals= ['DLBlas OP', 'Torch OP'],  # Possible values for `line_arg`.
        line_names=['DLBlas OP', 'Torch OP'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name=plot_name,  # Name for the plot. Used also as a file name for saving the plot.
        args={"batch_size":1, "seq_len":32*1024, "dtype":torch.bfloat16},  # Values for function arguments not in `x_names` and `y_name`.
    ))
    def benchmark(hidden_size, provider, batch_size, seq_len, dtype):
        x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
        if hidden_size >= 32*1024 and dlblas_op == layernorm_normal_loop:
            return 0, 0, 0
        triton_x = x.clone().requires_grad_(False)
        torch_x = x.clone().requires_grad_(False)
        weight = torch.ones(hidden_size, dtype=dtype, device=device)
        bias = torch.randn(hidden_size, dtype=dtype, device=device)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'DLBlas OP':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: dlblas_op(triton_x, weight, bias, eps=1e-6), quantiles=quantiles)
        elif provider == 'Torch OP':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.layer_norm(torch_x, (hidden_size,), weight, bias, eps=1e-6), quantiles=quantiles)
        else:
            raise RuntimeError(f'unknown provider:{provider}')

        gbps = lambda ms: (2 * x.numel() * x.element_size() * 1e-9 + 2*weight.numel()*weight.element_size()*1e-9) / (ms * 1e-3)
        return gbps(ms), gbps(max_ms), gbps(min_ms)
    benchmark.run(print_data=True, show_plots=True, save_path='./')


if __name__ == "__main__":
    bench_layernorm(layernorm_torch, 'layernorm_torch-performance')
    bench_layernorm(layernorm_normal, 'layernorm-normal-performance')
    bench_layernorm(layernorm_normal_loop, 'layernorm-normal-loop-performance')
    bench_layernorm(layernorm_opt_2D, 'layernorm_opt_2D-performance')
    bench_layernorm(layernorm_opt_mask_2D_tma, 'layernorm_opt_mask_2D_tma-performance')
    bench_layernorm(layernorm_opt_mask, 'layernorm_opt_mask-performance')
    bench_layernorm(layernorm_opt, 'layernorm_opt-performance')
    bench_layernorm(layernorm_dlblas, 'layernorm-dlblas-performance')